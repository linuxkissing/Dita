from close_loop_eval_diffusion_calvin import PytorchDiffInference
from libero.libero import benchmark
import tqdm

from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action, 
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
import os
import json
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id






def get_task_suite_name(dataset_name):

    if dataset_name == 'libero_10_no_noops':
        return 'libero_10'
    if dataset_name == 'libero_spatial_no_noops':
        return 'libero_spatial'
    if dataset_name == 'libero_object_no_noops':
        return 'libero_object'
    if dataset_name == 'libero_goal_no_noops':
        return 'libero_goal'



def close_loop_eval_libero(
    test_episodes_num=100,
    model=None,
    args=None,
    stride=1,
    root_folder = None,
    cfg=None, 
    dataset_statistics = None, 
):
    # import ipdb;ipdb.set_trace()
    if root_folder != None:
        json_folder = os.path.join(root_folder, 'statistic')
        video_folder = os.path.join(root_folder, 'videos')

        os.makedirs(json_folder, exist_ok=True)
        os.makedirs(video_folder, exist_ok=True)    
    
    device = torch.device('cuda', int(os.environ["LOCAL_RANK"]))
    cuda_id = device.index if device.type == "cuda" else 0
    try:
        egl_id = get_egl_device_id(cuda_id)
    except EglDeviceNotFoundError:
        egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)

    num_trails_per_task = 50
    num_steps_wait = 10

    model = PytorchDiffInference(
        model = model,
        sequence_length=cfg.dataset.traj_length,
        use_wrist_img = cfg.model.use_wrist_img,
        num_pred_action = cfg.num_pred_action,
        stride = stride,
        use_action_head_diff = cfg.use_action_head_diff,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = get_task_suite_name(cfg.dataname)
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    # num_tasks_in_suite = 2
    if args.rank==0:
        print(f"Task suite: {task_suite_name}", flush = True)

    resize_size = 224

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, None, resolution = 256, cuda_id=args.local_rank)

        model.set_natural_instruction(task_description)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(int(num_trails_per_task // args.world_size))):
            
            model.reset_observation()
            if args.rank==0:
                print(f"\nTask: {task_description}", flush = True)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx + args.rank * (num_trails_per_task // args.world_size)])

            # Setup
            t = 0
            replay_images = []
            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            if args.rank==0:
                print(f"Starting episode {task_episodes+1}...", flush = True)

            while t < max_steps + num_steps_wait:

                # try:

                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(None))
                        t += 1
                        continue
                    
                    img = get_libero_image(obs, resize_size)
                    state = np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    )
                    state = torch.tensor(state).to(torch.float32)
                    replay_images.append(img)
                    
                    model.set_observation(img)
                    
                    model_output = model.inference(
                        trajectory_dim = cfg.trajectory_dim, # set trajectory = 7
                        ret_7 = True,
                    )
                    # import ipdb;ipdb.set_trace()
                    model_output[...,-1] = np.where(model_output[...,-1] > 0.5, np.ones_like(model_output[...,-1]), np.ones_like(model_output[...,-1])*(-1))
                    normalized_actions = model_output
                    action_norm_stats = dataset_statistics[cfg.dataname]['action']
                    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                    actions = np.where(
                        mask,
                        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                        normalized_actions,
                    )
                    actions[...,-1] = actions[..., -1] * -1.0
                    
                    # for ii in range(model.num_pred_action):
                    for ii in range(1):

                        action = actions[ii]

                        obs, reward, done, info = env.step(action.tolist())

                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        
                        t += 1

                    if done:
                        break
                
                # except Exception as e:

                #     print(f"Caught exception: {e}", flush = True)
                #     break
            
            task_episodes += 1
            total_episodes += 1
            model.save_video(os.path.join(video_folder, f'rank{args.rank}_{total_episodes}_{done}.mp4'))


            # save_rollout_video(
            #     replay_images, total_episodes, success=done, task_description=task_description
            # )
            if args.rank==0:
                print(f"Success: {done}", flush = True)
                print(f"# episodes completed so far: {total_episodes}", flush = True)
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", flush = True)

        if args.rank==0:
            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}", flush = True)
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}", flush = True)
    if args.rank==0:
        print(f"success_rate/total: {float(total_successes) / float(total_episodes)}", flush = True)
        print(f"num_episodes/total: {total_episodes}", flush = True)

    with open(os.path.join(json_folder, f'rank_{args.rank}.json'),'w') as file:
        json.dump({'success_rate': float(total_successes) / float(total_episodes), 'num_episodes': total_episodes}, file)

    return None
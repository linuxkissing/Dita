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
import gymnasium as gym
import numpy as np
import sapien.core as sapien
import torch
import torch.nn as nn
import torchvision
from Dataset_Sim.SimDataset import process_traj_v3
from moviepy.editor import ImageSequenceClip
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat
from moviepy.editor import ImageSequenceClip





def get_task_suite_name(dataset_name):

    if dataset_name == 'libero_10_no_noops':
        return 'libero_10'
    if dataset_name == 'libero_spatial_no_noops':
        return 'libero_spatial'
    if dataset_name == 'libero_object_no_noops':
        return 'libero_object'
    if dataset_name == 'libero_goal_no_noops':
        return 'libero_goal'


class PytorchDiffInference(nn.Module):
    def __init__(self, model, prediction_type='epsilon',sequence_length = 15, 
                 use_wrist_img=False, device="cuda", 
                 stride=1, num_pred_action=4,
                 use_action_head_diff=0):
        super().__init__()

        self.device = torch.device(device)

        # use_wrist_img = use_wrist_img
        use_depth_img = False

        self.use_wrist_img = use_wrist_img
        self.use_depth_img = use_depth_img

        self.sequence_length = sequence_length
        self.num_pred_action = num_pred_action
        self.use_action_head_diff = use_action_head_diff

        self.stride = stride

        self.model = model

        print('sequence_length:', self.sequence_length)
        try:
            if hasattr(self.model, 'module'):
                self.use_wrist_img = self.model.module.use_wrist_img
                self.use_depth_img = self.model.module.use_depth_img
            else:
                self.use_wrist_img = self.model.use_wrist_img
                self.use_depth_img = self.model.use_depth_img
        except:
            self.use_wrist_img = False
            self.use_depth_img = False

        self.model.eval()
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.instruction = ""
        self.stride = stride
        self.data_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ToTensor(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224,224), antialias=True),
                torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.clip_tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14/", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14/"
        ).text_model

        self.to(self.device)
        self.frame = 0
        
        # self.base_episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_0.pkl", "rb"))
        # self.episode = pkl.load(open("/xxx/xxx/Anonymous/project/embodied_foundation/PickCube-v0_traj_0_camera_1.pkl", "rb"))
        self.eef_pose = None
        self.empty_instruction = None
        # model_output: dx dy dz dqw dqx dqy dqz terminate

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            text_embeddings = self.clip_text_encoder(**inputs)[0].squeeze(0)
        self.instruction = text_embeddings

    def set_eef_pose(self, eef_pose):
        self.eef_pose = eef_pose

    def set_observation(self, rgb, depth=None, wrist=None):
        assert (rgb >= 0).all() and (rgb <= 255).all()
        self.observation.append(rgb)
        if self.model_input == []:

            # rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = rgb_data.unsqueeze(0)
            if len(self.model_input) < self.sequence_length:
                self.model_input = self.model_input.repeat(self.sequence_length, 1, 1, 1)
        else:

            # rgb = torch.tensor(rgb).to(self.device, non_blocking=True)
            rgb_data = self.data_transform(rgb).to(self.device, non_blocking=True)
            self.model_input = torch.cat((self.model_input, rgb_data.unsqueeze(0)), dim=0)
            self.model_input = self.model_input[-self.sequence_length :]

        if wrist is not None and self.use_wrist_img:
            if self.model_input_wrist == []:

                # wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                wrist_data = self.data_transform(wrist).to(self.device, non_blocking=True)

                self.model_input_wrist = wrist_data.unsqueeze(0)
                if len(self.model_input_wrist) < self.sequence_length:
                    self.model_input_wrist = self.model_input_wrist.repeat(self.sequence_length, 1, 1, 1)
            else:

                # wrist_data = torch.tensor(wrist).to(self.device, non_blocking=True)

                # wrist_data = self.data_transform((wrist_data / 255.0).permute(2, 0, 1).contiguous())
                wrist_data = self.data_transform(wrist).to(self.device, non_blocking=True)

                self.model_input_wrist = torch.cat((self.model_input_wrist, wrist_data.unsqueeze(0)), dim=0)
                self.model_input_wrist = self.model_input_wrist[-self.sequence_length :]

        if depth is not None and self.use_depth_img:
            if self.model_input_depth == []:

                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = depth_data.unsqueeze(0)
            else:
                depth_data = torch.tensor(depth / 10).to(self.device, non_blocking=True)
                self.model_input_depth = torch.cat((self.model_input_depth, depth_data.unsqueeze(0)), dim=0)
                self.model_input_depth = self.model_input_depth[-self.sequence_length :]
            depth = torch.tensor(depth / 10 * 255).repeat(1, 1, 3).byte().cpu().numpy()
            self.observation[-1] = np.concatenate([self.observation[-1], depth], axis=1)

    def reset_observation(self):
        self.model_input = []
        self.observation = []
        self.model_input_wrist = []
        self.model_input_depth = []
        self.frame = 0

                
    def save_video(self, fpath):

        clip = ImageSequenceClip(self.observation, fps=10 / self.stride)
        clip.write_videofile(fpath, codec="libx264", audio=False, logger=None)  # Use 'libx264' for the H.264 codec

    def calc_act(self, base_episode, camera_extrinsic_cv, current_frame_idx):
        try:
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][current_frame_idx + self.stride]["prev_ee_pose"]).clone()
        except:
            current_frame_idx = min(current_frame_idx, len(base_episode["step"]) - 1)
            pose1 = torch.tensor(base_episode["step"][current_frame_idx]["prev_ee_pose"]).clone()
            # pose2 = torch.tensor(base_episode["step"][current_frame_idx]["target_ee_pose"]).clone()
            pose2 = torch.tensor(base_episode["step"][-1]["prev_ee_pose"]).clone()

        pose1[0] -= 0.615  # base to world
        pose2[0] -= 0.615  # base to world
        action = {}
        action["world_vector"], action["rotation_delta"] = process_traj_v3(
            (camera_extrinsic_cv),
            pose1,
            pose2,
        )

        if base_episode["step"][current_frame_idx]["is_terminal"] == True:
            action["terminate_episode"] = torch.tensor([1, 0, 0], dtype=torch.int32)
        else:
            action["terminate_episode"] = torch.tensor([0, 1, 0], dtype=torch.int32)
        action["gripper_closedness_action"] = torch.tensor(
            base_episode["step"][current_frame_idx]["action"][-1],
            dtype=torch.float32,
        ).unsqueeze(-1)

        return action

    def get_target_pose(self, delta_pos, delta_rot):
        target_ee_pose_at_camera = sapien.Pose(p=self.eef_pose.p + delta_pos)
        r_prev = quat2mat(self.eef_pose.q)
        r_diff = quat2mat(delta_rot)
        r_target = r_diff @ r_prev
        target_ee_pose_at_camera.set_q(mat2quat(r_target))

        return target_ee_pose_at_camera
 
    def inference(self, extrinsics=None, abs_pose=0, abs_seq_pose=0, horizon=-1, set_pose=False, trajectory_dim=11, reg_prediction_nums=0, pad_diff_nums=0, obs_pose=None, cfg=0, ret_7=False, dim=0):
        # import ipdb;ipdb.set_trace()
        obs = {"image": self.model_input[-self.sequence_length :].unsqueeze(0)}
        if self.use_wrist_img:
            obs["wrist_image"] = self.model_input_wrist[-self.sequence_length :].unsqueeze(0)
        if self.use_depth_img:
            obs["depth_image"] = self.model_input_depth[-self.sequence_length :].unsqueeze(0)
        obs["natural_language_embedding"] = self.instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)

        if cfg != 0:
            # classifier free guidance
            if self.empty_instruction is None:
                inputs = self.clip_tokenizer(text='', return_tensors="pt", max_length=77, padding="max_length")
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                with torch.no_grad():
                    self.empty_instruction = self.clip_text_encoder(**inputs)[0].squeeze(0)
                self.empty_instruction = self.empty_instruction[None, None, ...].repeat(1, obs["image"].shape[1], 1, 1)
            obs['natural_language_embedding'] = torch.cat([self.empty_instruction, obs['natural_language_embedding'], ], dim=0)
            obs["image"] = torch.cat([obs["image"], obs["image"]], dim=0)

        if obs_pose is not None:
            obs['poses'] = obs_pose.to(obs["natural_language_embedding"].device)  # B 1 11
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # 1 x T x L x C
                if self.use_action_head_diff == 2:
                    if hasattr(self.model, 'module'):
                        model_output = self.model.module.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                    else:
                        model_output = self.model.inference_withfeats(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                else:
                    if hasattr(self.model, 'module'):
                        model_output = self.model.module.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                    else:
                        model_output = self.model.inference(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon, reg_prediction_nums=reg_prediction_nums, pad_diff_nums=pad_diff_nums, cfg=cfg)
                # model_output = self.model.module.inference1(obs, num_pred_action=self.num_pred_action, abs_pose=abs_pose, horizon=horizon)
                # 
        if ret_7:
            # this is for openx data
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                ], dim=-1
            )[0]
            self.frame += self.stride
            return output.cpu().numpy()
        elif trajectory_dim == 7:
            assert set_pose
            # this is for openx data
            rot_delta = model_output["rotation_delta"][0]
            def euler_to_quaternion(eulers):
                import scipy.spatial.transform as st
                quaternion = st.Rotation.from_euler('xyz', eulers).as_quat()
                return torch.tensor([quaternion[-1], quaternion[0], quaternion[1], quaternion[2]])
            quat_list = torch.stack([euler_to_quaternion(rot_delta[i].cpu().numpy()) for i in range(len(rot_delta))])[None,...]
            import numpy as np
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    quat_list.cpu(),
                    # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            self.frame += self.stride
            return output.cpu().numpy()
            pass
        else:
            output = torch.cat(
                [
                    model_output["world_vector"].cpu(),
                    model_output["rotation_delta"].cpu(),
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            pass

        # add 1 to quat
        output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1

        output[..., -1] = output[..., -1] > 0.5

        self.frame += self.stride

        return output.cpu().numpy()



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
    # from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
    # try:
    #     egl_id = get_egl_device_id(cuda_id)
    # except EglDeviceNotFoundError:
    #     egl_id = 0
    #     os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)

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
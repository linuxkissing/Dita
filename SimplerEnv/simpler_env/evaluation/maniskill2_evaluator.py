"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_interval_video, write_video


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    # __import__('ipdb').set_trace()
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)

    if 'DEBUG' in os.environ:
        import ipdb;ipdb.set_trace()
    # env.env.env.env._cameras['base_camera'].get_params()
        # env.env.env.env._cameras['overhead_camera'].get_params()
        # env.env.env.env._cameras['3rd_view_camera'].get_params()
# google robot
# {'extrinsic_cv': array([[ 9.2726380e-02,  9.9569201e-01, -2.2351742e-08, -2.4154928e-01],
#        [ 7.0387590e-01, -6.5550283e-02, -7.0729196e-01,  7.8472352e-01],
#        [-7.0424509e-01,  6.5584615e-02, -7.0692158e-01,  1.0430859e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), 'cam2world_gl': array([[ 0.09272623, -0.7038759 ,  0.70424485,  0.2046381 ],
#        [ 0.9956918 ,  0.06555015, -0.06558463,  0.22353707],
#        [ 0.        ,  0.7072921 ,  0.70692146,  1.2924082 ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]],
#       dtype=float32), 'intrinsic_cv': array([[425.,   0., 320.],
#        [  0., 425., 256.],
#        [  0.,   0.,   1.]], dtype=float32)}
        
# {'extrinsic_cv': array([[ 9.2726380e-02,  9.9569201e-01, -2.2351742e-08, -2.4154928e-01],
    #    [ 7.0387590e-01, -6.5550283e-02, -7.0729196e-01,  7.8472352e-01],
    #    [-7.0424509e-01,  6.5584615e-02, -7.0692158e-01,  1.0430859e+00],
    #    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
    #   dtype=float32), 'cam2world_gl': array([[ 0.09272623, -0.7038759 ,  0.70424485,  0.2046381 ],
    #    [ 0.9956918 ,  0.06555015, -0.06558463,  0.22353707],
    #    [ 0.        ,  0.7072921 ,  0.70692146,  1.2924082 ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=float32), 'intrinsic_cv': array([[425.,   0., 320.],
    #    [  0., 425., 256.],
    #    [  0.,   0.,   1.]], dtype=float32)}
        
    # windx
    #     {'extrinsic_cv': array([[-0.48393318,  0.8751056 ,  0.        , -0.09338154],
    #    [ 0.6025578 ,  0.33321443, -0.72518444,  0.74075735],
    #    [-0.63461304, -0.35094085, -0.68855476,  1.0061874 ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=float32), 'cam2world_gl': array([[-0.48393357, -0.6025578 ,  0.6346128 ,  0.14699998],
    #    [ 0.87510526, -0.33321476,  0.35094088,  0.188     ],
    #    [ 0.        ,  0.72518474,  0.68855447,  1.2300001 ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]],
    #   dtype=float32), 'intrinsic_cv': array([[623.588,   0.   , 319.501],
    #    [  0.   , 623.588, 239.545],
    #    [  0.   ,   0.   ,   1.   ]], dtype=float32)}
    # {'extrinsic_cv': array([[-4.1036496e-01,  9.1192150e-01, -1.3411045e-07, -1.9287115e-01],
    #    [ 6.3068348e-01,  2.8380769e-01, -7.2228217e-01,  7.3701322e-01],
    #    [-6.5866470e-01, -2.9639938e-01, -6.9159847e-01,  1.0232365e+00],
    #    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
    #   dtype=float32), 'cam2world_gl': array([[-4.1036487e-01, -6.3068348e-01,  6.5866458e-01,  1.3000000e-01],
    #    [ 9.1192144e-01, -2.8380764e-01,  2.9639941e-01,  2.6999998e-01],
    #    [-1.1920929e-07,  7.2228223e-01,  6.9159842e-01,  1.2400002e+00],
    #    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
    #   dtype=float32), 'intrinsic_cv': array([[623.588,   0.   , 319.501],
    #    [  0.   , 623.588, 239.545],
    #    [  0.   ,   0.   ,   1.   ]], dtype=float32)}

    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    env_save_name = env_name

    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]

    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_dir = f"{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}"

    video_tag = ''
    if obj_variation_mode == "xy":
        video_tag = f"_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_tag = f"_obj_episode_{obj_episode_id}"
    if 'DEBUG1' in os.environ and os.path.exists(os.path.join(logging_dir, video_dir)) and \
        len([item for item in os.listdir(os.path.join(logging_dir, video_dir)) if item.__contains__(video_tag)]) > 0:
        return False
    print('process:', os.path.join(logging_dir, video_dir), flush=True)
    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    # action_ensemble = model.action_ensemble_temp  if hasattr(model, "action_ensemble") else "none"

    # Step the environment
    task_descriptions = []
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        # print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        images.append(image)
        task_descriptions.append(task_description)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name

    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)
    print('write', video_path, flush=True)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            print(obj_init_x, obj_init_y, )
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        print(obj_episode_id, )
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr

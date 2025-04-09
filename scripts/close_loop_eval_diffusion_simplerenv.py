import os
from typing import Optional, Sequence


os.environ["MS2_ASSET_DIR"] = "/mnt/petrelfs/share_data/zhaochengyang/maniskill2/assets"
import pickle
import sys
import time
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torchvision
from transforms3d.euler import euler2axangle

# from moviepy.editor import ImageSequenceClip
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
from transforms3d.quaternions import mat2quat, quat2mat
norms_dict = {
        'google_robot': {
    "action": { "mean": [
        0.006987511646002531,
        0.0062658516690135,
        -0.012625154107809067,
        0.04333285614848137,
        -0.005756264552474022,
        0.0009130426915362477,
        0.5354204773902893
      ],
      "std": [
        0.06921114027500153,
        0.059708621352910995,
        0.07353121042251587,
        0.15610602498054504,
        0.13164429366588593,
        0.14593781530857086,
        0.49711522459983826
      ],
      "max": [
        2.9984593391418457,
        22.09052848815918,
        2.7507524490356445,
        1.570636510848999,
        1.5321086645126343,
        1.5691522359848022,
        1.0
      ],
      "min": [
        -2.0204520225524902,
        -5.497899532318115,
        -2.031663417816162,
        -1.569917917251587,
        -1.569892168045044,
        -1.570419430732727,
        0.0
      ],
      "q01": [
        -0.22453527510166169,
        -0.14820013284683228,
        -0.231589707583189,
        -0.3517994859814644,
        -0.4193011274933815,
        -0.43643461108207704,
        0.0
      ],
      "q99": [
        0.17824687153100965,
        0.14938379630446405,
        0.21842354819178575,
        0.5892666035890578,
        0.35272657424211445,
        0.44796681255102094,
        1.0
      ],
      "mask": np.array([1, 1, 1, 1, 1, 1, 0.0])
    },
    "proprio": {
      "mean": [
        0.5598950386047363,
        -0.08333975076675415,
        0.7770934700965881,
        -1.501692295074463,
        0.6251405477523804,
        -1.6193324327468872,
        0.42613422870635986
      ],
      "std": [
        0.12432791292667389,
        0.11558851599693298,
        0.24595724046230316,
        2.3793978691101074,
        0.4169624149799347,
        0.8072276711463928,
        0.4554515480995178
      ],
      "max": [
        1.0534898042678833,
        0.48018959164619446,
        1.6896663904190063,
        3.1415927410125732,
        1.5707963705062866,
        3.1415889263153076,
        1.0
      ],
      "min": [
        -0.4436439275741577,
        -0.9970501065254211,
        -0.006579156965017319,
        -3.141592264175415,
        -1.5690358877182007,
        -3.1415863037109375,
        0.0
      ],
      "q01": [
        0.32481380939483645,
        -0.28334290891885755,
        0.14107070609927178,
        -3.130769968032837,
        -0.23744970083236697,
        -2.965691123008728,
        0.0
      ],
      "q99": [
        0.8750156319141384,
        0.21247054174542404,
        1.0727112340927123,
        3.1300241947174072,
        1.4914317286014553,
        2.842233943939208,
        1.0
      ]
    },
    "num_transitions": 3786400,
    "num_trajectories": 87212
  },
        'bridge_orig':{
    "action": {
      "mean": [
        0.00023341945779975504,
        0.0001300470030400902,
        -0.00012762592814397067,
        -0.0001556538773002103,
        -0.0004039329360239208,
        0.00023557659005746245,
        0.5764578580856323
      ],
      "std": [
        0.00976590532809496,
        0.013689203187823296,
        0.01266736350953579,
        0.02853410877287388,
        0.030637994408607483,
        0.07691272348165512,
        0.4973713755607605
      ],
      "max": [
        0.41691166162490845,
        0.25864794850349426,
        0.21218234300613403,
        3.122201919555664,
        1.8618112802505493,
        6.280478477478027,
        1.0
      ],
      "min": [
        -0.4007510244846344,
        -0.13874775171279907,
        -0.22553899884223938,
        -3.2010786533355713,
        -1.8618112802505493,
        -6.279075622558594,
        0.0
      ],
      "q01": [-0.02872725, -0.0417035 , -0.02609386, -0.07763332, -0.09229686,
       -0.20506569,  0.        ],
    #   'q01': array([-0.02872725, -0.0417035 , -0.02609386, -0.08092105, -0.092887  ,
    #    -0.20718276,  0.        ]), 'q99': array([0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.07792851,
    #    0.20382574, 1.     ])
    #    'q01': array([-0.02872725, -0.0417035 , -0.02609386, -0.07763332, -0.09229686,
    #    -0.20506569,  0.        ]), 'q99': array([0.02830968, 0.04085525, 0.04016159, 0.08012086, 0.07817021,
    #    0.2017101 , 1.        ])},

      "q99": [0.02830968, 0.04085525, 0.04016159, 0.08012086, 0.07817021, 0.2017101 , 1.        ],
      "mask": [
        1,
        1,
        1,
        1,
        1,
        1,
        0
      ]
    },
    "proprio": {
      "mean": [
        0.3094126582145691,
        0.030575918033719063,
        0.06454135477542877,
        0.00682411715388298,
        -0.07762681692838669,
        0.10757910460233688,
        0.7083898782730103
      ],
      "std": [
        0.06052854657173157,
        0.09188618510961533,
        0.05159863084554672,
        0.1318272054195404,
        0.1703108251094818,
        0.5767307877540588,
        0.35197606682777405
      ],
      "max": [
        0.5862360596656799,
        0.4034728705883026,
        0.36494991183280945,
        1.514088749885559,
        1.570796251296997,
        3.1415255069732666,
        1.1154625415802002
      ],
      "min": [
        -0.04167502000927925,
        -0.3945816159248352,
        -0.15537554025650024,
        -3.141592502593994,
        -1.4992541074752808,
        -3.14153790473938,
        0.04637829214334488
      ],
      "q01": [
        0.17111587673425674,
        -0.16998695254325866,
        -0.05544630073010921,
        -0.366876106262207,
        -0.5443069756031036,
        -1.3536006283760071,
        0.052190229296684265
      ],
      "q99": [
        0.45320980012416834,
        0.23518154799938193,
        0.1951873075962065,
        0.3806115746498103,
        0.2789784955978382,
        1.8410426235198971,
        1.0105689764022827
      ]
    },
    "num_transitions": 2135463,
    "num_trajectories": 60064
  }
    }
    

# param


class PytorchDiffInference(nn.Module):
    def __init__(self, model, prediction_type='epsilon',sequence_length = 15, 
                 use_wrist_img=False, device="cuda", 
                 stride=1, num_pred_action=4,
                 use_action_head_diff=0, policy_setup='google_robot'):
        super().__init__()

        self.policy_setup = policy_setup
        self.action_scale = 1.
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
                self.text_max_length = self.model.module.text_max_length
            else:
                self.use_wrist_img = self.model.use_wrist_img
                self.use_depth_img = self.model.use_depth_img
                self.text_max_length = self.model.text_max_length
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
            "/mnt/petrelfs/share_data/houzhi/clip-vit-large-patch14/", use_fast=False
        )
        self.clip_text_encoder = CLIPModel.from_pretrained(
            "/mnt/petrelfs/share_data/houzhi/clip-vit-large-patch14/"
        ).text_model

        self.to(self.device)
        self.frame = 0
        
        # self.base_episode = pkl.load(open("/mnt/petrelfs/xiongyuwen/project/embodied_foundation/PickCube-v0_traj_0_camera_0.pkl", "rb"))
        # self.episode = pkl.load(open("/mnt/petrelfs/xiongyuwen/project/embodied_foundation/PickCube-v0_traj_0_camera_1.pkl", "rb"))
        self.eef_pose = None
        self.empty_instruction = None
        # model_output: dx dy dz dqw dqx dqy dqz terminate
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0      
        if policy_setup == "widowx_bridge":
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )

        

    def set_natural_instruction(self, instruction: str):
        inputs = self.clip_tokenizer(text=instruction, return_tensors="pt", max_length=self.text_max_length, padding="max_length")
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
            # wrist = (
            #     nn.functional.interpolate(torch.tensor(wrist).permute(2, 0, 1).unsqueeze(0), size=(224, 224), mode="nearest")
            #     .squeeze()
            #     .permute(1, 2, 0)
            #     .cpu()
            #     .numpy()
            # )
            # self.observation[-1] = np.concatenate([self.observation[-1], wrist], axis=1)
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
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None        

    def reset(self, task_description):
        self.reset_observation()
        self.set_natural_instruction(task_description)
    
    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        global norms_dict
        # Image.fromarray(cv2.resize(np.asarray(Image.open("im_0.jpg")), dsize=(224,224)))
        if self.policy_setup == "google_robot":
            action_norms = norms_dict['google_robot']
            if 'PADRESIZE' in os.environ:
                data_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Pad(padding=(64, 0), fill=0, padding_mode='constant'),  # Pad the image with a border of 10 pixels
                    torchvision.transforms.Resize(size=(224, 224))  # Resize the image to 224x224
                ])

                # 512 640 
                # 640 - 512
            elif 'ORIGSIZE' in os.environ:
                data_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.CenterCrop(size=(512, 512)),
                        torchvision.transforms.Resize((224 , 224), antialias=True)
                    ]
                )
            else:
                data_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.CenterCrop(size=(512, 512)),
                        torchvision.transforms.Resize((224 , 224), antialias=True)
                    ]
                )

                
            pass
        elif self.policy_setup == "widowx_bridge":
            action_norms = norms_dict['bridge_orig']
            if 'PADRESIZE' in os.environ:
                data_transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Pad(padding=(80, 0), fill=0, padding_mode='constant'),  # Pad the image with a border of 10 pixels
                    torchvision.transforms.Resize(size=(224, 224))  # Resize the image to 224x224
                ])
            else:
                data_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        # torchvision.transforms.CenterCrop(size=(480, 480)),
                        torchvision.transforms.Resize((224 , 224), antialias=True)
                    ]
                )

        image = (data_transform(image)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

        self.set_observation(rgb=image, wrist=None)
        self.set_natural_instruction(task_description)     
        model_output = self.inference(None, 
                                    abs_pose= 0, 
                                    set_pose=True, 
                                        trajectory_dim=7, reg_prediction_nums=0, pad_diff_nums=0, 
                                        ret_7=True,
                                        obs_pose=None, cfg = 0)
        normalized_actions = np.asarray(model_output[0])
        # TODO unnormalize
        action_low = np.asarray(action_norms['action']['q01'])
        action_high = np.asarray(action_norms['action']['q99'])


        if 'CAM_COORD' in os.environ:
            action_low = torch.tensor([-0.1261216700077057, -0.35921716690063477, 0.8430755734443665, -1.920664398670192, -1.0045121908187866, -3.1264941692352295, 0.0]).cpu().numpy()
            action_high = torch.tensor([0.31185251474380493, 0.18027661740779877, 1.4179376363754272, 2.614788055419922, 1.0483404397964478, 3.1281654834747314, 1.0]).cpu().numpy()

            pass
        mask = np.asarray(action_norms['action']['mask'])
        # print('before norm:', normalized_actions, action_low)
        
        action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        model_output[0] = action
        # print(action, normalized_actions, action_low, action_high)

        raw_action = {
            "world_vector": np.array(model_output[0, :3]),
            "rotation_delta": np.array(model_output[0, 3:6]),
            "open_gripper": np.array(model_output[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        import cv2 as cv
        image = cv.resize(image, tuple((224, 224)), interpolation=cv.INTER_AREA)
        return image
    
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


    def save_video(self, fpath):
        # height, width, _ = self.observation[0].shape
        # fourcc = cv2.VideoWriter_fourcc(*"AVC1")
        # if os.path.exists(fpath):
        #     os.remove(fpath)
        # out = cv2.VideoWriter(fpath, fourcc, 10.0, (width, height))
        # for image in self.observation:
        #     out.write(image)  # Write out frame to video
        # out.release()
        from moviepy import ImageSequenceClip
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
        from Dataset_Sim.SimDataset import process_traj_v3
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
        import sapien.core as sapien
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
                inputs = self.clip_tokenizer(text='', return_tensors="pt", max_length=self.text_max_length, padding="max_length")
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
        elif abs_seq_pose:
            # convert to quat

            model_output["rotation_delta"][0,:, 0] += 1
            # print(model_output["world_vector"], model_output["rotation_delta"])
            target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                           for tp in range(model_output["world_vector"].shape[1])]
            import numpy as np
            output = torch.cat(
                [
                    # model_output["world_vector"],
                    # model_output["rotation_delta"],
                    torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                    torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                    model_output["gripper_closedness_action"].cpu(),
                    model_output["terminate_episode"][:,:, [0]].cpu()
                ], dim=-1
            )[0]
            pass
        elif not abs_pose:
            if not set_pose:
                model_output["rotation_delta"][0,:, 0] += 1
                # print(model_output["world_vector"], model_output["rotation_delta"])
                target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                            for tp in range(model_output["world_vector"].shape[1])]
                import numpy as np
                output = torch.cat(
                    [
                        # model_output["world_vector"],
                        # model_output["rotation_delta"],
                        torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                        torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                        model_output["gripper_closedness_action"].cpu(),
                        model_output["terminate_episode"][:,:, [0]].cpu()
                    ], dim=-1
                )[0]
            else:
                model_output["rotation_delta"][0,:, 0] += 1
                # print(model_output["world_vector"], model_output["rotation_delta"])
                # target_pose = [self.get_target_pose(model_output["world_vector"][0,tp].cpu().numpy(), model_output["rotation_delta"][0, tp].cpu().numpy()) 
                            # for tp in range(model_output["world_vector"].shape[1])]
                import numpy as np
                output = torch.cat(
                    [
                        model_output["world_vector"].cpu(),
                        model_output["rotation_delta"].cpu(),
                        # torch.tensor(np.stack([tp.p for tp in target_pose]))[None,...],
                        # torch.tensor(np.stack([tp.q for tp in target_pose]))[None,...],
                        model_output["gripper_closedness_action"].cpu(),
                        model_output["terminate_episode"][:,:, [0]].cpu()
                    ], dim=-1
                )[0]
            # output = torch.cat(
            #     [
            #         model_output["world_vector"].cpu(),
            #         model_output["rotation_delta"].cpu(),
            #         model_output["gripper_closedness_action"].cpu(),
            #         model_output["terminate_episode"][:,:, [0]].cpu()
            #     ], dim=-1
            # )[0]
            # print(output[...,:7], output[...,:7].min(), output[...,:7].max(),"output")
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

        # import ipdb; ipdb.set_trace()
        # output = torch.diagonal(model_output, dim1=1, dim2=2).squeeze(0)

        # add 1 to quat
        output[..., -2] = (output[...,-2] > 0.0).float() * 2 - 1
        # output[..., 3] += 1
        # print('traj term:', output[..., -1])
        # import ipdb;ipdb.set_trace()
        output[..., -1] = output[..., -1] > 0.5
        # output[..., -1] = torch.logical_and(torch.logical_and(output[..., -1] > 0.5, model_output["terminate_episode"][0,:, 1].cpu() < 0.5), model_output["terminate_episode"][0,:, 2].cpu() < 0.5).float()
        # print(output[..., -1].shape)
        # gt_output = self.calc_act(self.base_episode, torch.tensor(self.episode["camera_extrinsic_cv"]), self.frame)
        # gt_output = self.calc_act(self.base_episode, torch.eye(4), self.frame)

        # gt_output = torch.cat(
        #     [
        #         gt_output["world_vector"],
        #         gt_output["rotation_delta"],
        #         gt_output["gripper_closedness_action"],
        #         gt_output["terminate_episode"][[0]],
        #     ]
        # )

        # print(list(output.cpu().numpy() - gt_output.cpu().numpy()))
        # print(f"eval frame {self.frame}", flush=True)
        # Image.fromarray((unnormalize(self.model_input[-1].permute(1, 2, 0)).clamp(0, 1) * 255).byte().cpu().numpy()).save(f"obs_{self.frame}.png")

        # self.frame += 1
        self.frame += self.stride

        # import ipdb

        # ipdb.set_trace()

        return output.cpu().numpy()

def analyze_traj_str(traj_str):

    env_id = traj_str.split("-")[0] + "-v0"
    select_camera = f"camera_{traj_str[-5]}"
    seed_start_pos = traj_str.find("traj") + 5
    seed = int(traj_str[seed_start_pos:-13])
    return env_id, select_camera, seed


def close_loop_eval_simplerenv(
    obs_mode="rgbd",
    reward_mode=None,
    render_mode="cameras",
    record_dir=None,
    render_goal_point=True,
    test_episodes_num=100,
    model=None,
    eval_data_list=None,
    args=None,
    rand_seed=0,
    camera_coord=True,
    stride=1,
    root_folder = None,
    data_root_path = None,
    cfg=None,
    eval_dataset=None,
):

    assert cfg is not None
    print('begin ....', root_folder)
    np.set_printoptions(suppress=True, precision=3)


    np.random.seed(0 % 9973)

    i = 0
    print('begin 2....')
    print(cfg)
    time_sequence_length = cfg.num_pred_action if 'wrap_grmg_data' in cfg and cfg.wrap_grmg_data == 1 else cfg.dataset.traj_length
    if 'wrap_grmg_data' in cfg and cfg.wrap_grmg_data == 2:
        time_sequence_length = cfg.num_pred_action + 1
    model = PytorchDiffInference(model=model, sequence_length = time_sequence_length, use_wrist_img=cfg.model.use_wrist_img,
                                 num_pred_action=cfg.num_pred_action, stride=stride, use_action_head_diff=cfg.use_action_head_diff)
    print('begin init model....')
    print('--------------------')


    import random

    print('begin gym....')

    # if record_dir != None:
    #     record_dir = record_dir.format(env_id=env_id)
    #     env = RecordEpisode(env, record_dir, render_mode=render_mode)
    # if render_goal_point and hasattr(env, "goal_site"):
    #     env.goal_site.unhide_visual()
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    import simpler_env

    global norms_dict
    results = {}
    n_trajs = 10
    for env_name in simpler_env.ENVIRONMENTS:
        print(env_name)
        print('move redbull can near apple_google_robot_move_near_v0')
        results[env_name] = []
        env = simpler_env.make(env_name)
        for ep_id in range(n_trajs):
            
            obs, reset_info = env.reset()
            model.reset_observation()
            instruction = env.get_language_instruction()
            print("Reset info", reset_info)
            print("Instruction", instruction)
            if env_name.startswith('google_robot'):
                action_norms = norms_dict['google_robot']
                if 'PADRESIZE' in os.environ:
                    data_transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Pad(padding=(64, 0), fill=0, padding_mode='constant'),  # Pad the image with a border of 10 pixels
                        torchvision.transforms.Resize(size=(224, 224))  # Resize the image to 224x224
                    ])

                    # 512 640 
                    # 640 - 512
                elif 'ORIGSIZE' in os.environ:
                    data_transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            # torchvision.transforms.CenterCrop(size=(512, 512)),
                            torchvision.transforms.Resize((224 , 224), antialias=True)
                        ]
                    )
                else:
                    data_transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.CenterCrop(size=(512, 512)),
                            torchvision.transforms.Resize((224 , 224), antialias=True)
                        ]
                    )

                    
                pass
            elif env_name.startswith('widowx'):
                action_norms = norms_dict['bridge_orig']
                if 'PADRESIZE' in os.environ:
                    data_transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Pad(padding=(80, 0), fill=0, padding_mode='constant'),  # Pad the image with a border of 10 pixels
                        torchvision.transforms.Resize(size=(224, 224))  # Resize the image to 224x224
                    ])
                else:
                    data_transform = torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            # torchvision.transforms.CenterCrop(size=(480, 480)),
                            torchvision.transforms.Resize((224 , 224), antialias=True)
                        ]
                    )


            
            

            done, truncated = False, False
            is_final_subtask = env.is_final_subtask() 
            while not (done or truncated):
                # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
                # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
                # image = get_image_from_maniskill2_obs_dict(env, obs)
                # action = env.action_space.sample() # replace this with your policy inference
                # import ipdb;ipdb.set_trace()
                if env_name.startswith('widowx'):
                    image = obs["image"]["3rd_view_camera"]["rgb"]
                else:
                    image = obs["image"]["overhead_camera"]["rgb"]            
                image = (data_transform(image)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)


                model.set_observation(rgb=image, wrist=None)
                model.set_natural_instruction(instruction)            
                model_output = model.inference(image if cfg.dataset.use_baseframe_action == False else torch.eye(4), 
                                            abs_pose= cfg.abs_pose if 'abs_pose' in cfg else 0, 
                                            set_pose=True, 
                                                trajectory_dim=7, reg_prediction_nums=0, pad_diff_nums=0, 
                                                ret_7=True,
                                                obs_pose=None, cfg = cfg.cfg if 'cfg' in cfg else 0)

                normalized_actions = model_output[0]
                # TODO unnormalize
                action_low = np.asarray(action_norms['action']['q01'])
                action_high = np.asarray(action_norms['action']['q99'])
                mask = np.asarray(action_norms['action']['mask'])
                # print('before norm:', normalized_actions, action_low)
                
                action = np.where(
                    mask,
                    0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                    normalized_actions,
                )

                raw_action = {
                            "world_vector": np.array(action[ :3]),
                            "rotation_delta": np.array(action[3:6]),
                            "open_gripper": np.array(action[ 6:7]),  # range [0, 1]; 1 = open; 0 = close
                        }
                model.action_scale = 1.
                # process raw_action to obtain the action to be sent to the maniskill2 environment
                action = {}
                action["world_vector"] = raw_action["world_vector"] * model.action_scale
                action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
                roll, pitch, yaw = action_rotation_delta
                action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
                action_rotation_axangle = action_rotation_ax * action_rotation_angle
                action["rot_axangle"] = action_rotation_axangle * model.action_scale


                if env_name.startswith('google'):
                    model.sticky_gripper_num_repeat = 15
                    current_gripper_action = raw_action["open_gripper"]
                    if model.previous_gripper_action is None:
                        relative_gripper_action = np.array([0])
                    else:
                        relative_gripper_action = model.previous_gripper_action - current_gripper_action
                    model.previous_gripper_action = current_gripper_action

                    if np.abs(relative_gripper_action) > 0.5 and (not model.sticky_action_is_on):
                        model.sticky_action_is_on = True
                        model.sticky_gripper_action = relative_gripper_action

                    if model.sticky_action_is_on:
                        model.gripper_action_repeat += 1
                        relative_gripper_action = model.sticky_gripper_action

                    if model.gripper_action_repeat == model.sticky_gripper_num_repeat:
                        model.sticky_action_is_on = False
                        model.gripper_action_repeat = 0
                        model.sticky_gripper_action = 0.0

                    action["gripper"] = relative_gripper_action

                else:
                    model.sticky_gripper_num_repeat = 1
                    action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

                action["terminate_episode"] = np.array([0.0])
# 

                # # print('act:', action, model_output[...,-1])
                # print(action)
# odict_keys(['agent', 'extra', 'camera_param', 'image'])
# env.env.env.env.get_obs()['agent']        
    #             ipdb> OrderedDict([('qpos', array([-0.26394573,  0.08319134,  0.50176114,  1.156859  ,  0.02858367,
    #     1.5925982 , -1.080653  ,  0.        ,  0.        , -0.00285961,
    #     0.7851361 ], dtype=float32)), ('qvel', array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)), ('controller', {'gripper': {'target_qpos': array([0., 0.], dtype=float32)}}), ('base_pose', array([0.35      , 0.19999999, 0.07904999, 0.        , 0.        ,
    #    0.        , 1.        ], dtype=float32))])        # 
                # env.env.env.env.get_obs()['extra']
# ipdb> OrderedDict([('tcp_pose', array([-0.08938442,  0.41841692,  1.0409188 ,  0.0324987 , -0.7801727 ,
#        -0.6038523 ,  0.16011547], dtype=float32))])
                action = np.concatenate([action['world_vector'], action['rot_axangle'], action['gripper']])
                obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
                new_instruction = env.get_language_instruction()
                if new_instruction != instruction:
                    # for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
                    instruction = new_instruction
                    model.set_natural_instruction(instruction)
                    print("New Instruction", instruction)

                is_final_subtask = env.is_final_subtask()
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    env.advance_to_next_subtask()

                episode_stats = info.get('episode_stats', {})
            if root_folder != None:
                try:
                    os.makedirs(os.path.join(root_folder, env_name,), exist_ok=True)
                    model.save_video(os.path.join(root_folder, env_name, f'{(i+args.rank * test_episodes_num):04d}_{instruction}_{ep_id}_{done}.mp4'))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            results[env_name].append(done)
        print(env_name, sum(results[env_name]) / len(results[env_name]), )
    print("Episode stats", episode_stats)
    print(simpler_env.ENVIRONMENTS)
    for env_name in results:
        print(env_name, sum(results[env_name]) / len(results[env_name]), )

    return 0, None, 0

# Observation example:
# obs['agent']
# ipdb> OrderedDict([('qpos', array([-0.26394573,  0.08319134,  0.50176114,  1.156859  ,  0.02858367,
#         1.5925982 , -1.080653  ,  0.        ,  0.        , -0.00285961,
#         0.7851361 ], dtype=float32)), ('qvel', array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)), ('controller', {'gripper': {'target_qpos': array([0., 0.], dtype=float32)}}), ('base_pose', array([0.35      , 0.19999999, 0.07904999, 0.        , 0.        ,
#        0.        , 1.        ], dtype=float32))])
# obs['extra']
# ipdb> OrderedDict([('tcp_pose', array([-0.08938442,  0.41841692,  1.0409188 ,  0.0324987 , -0.7801727 ,
    #    -0.6038523 ,  0.16011547], dtype=float32))])
# obs['camera_param']
# ipdb> OrderedDict([('base_camera', {'extrinsic_cv': array([[-2.0994479e-17,  1.0000000e+00, -2.5644997e-16,  1.6016833e-16],
#        [ 7.8086883e-01, -1.4380908e-16, -6.2469506e-01,  1.4055640e-01],
#        [-6.2469506e-01, -2.1336892e-16, -7.8086883e-01,  6.5592980e-01],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), 'cam2world_gl': array([[ 0.        , -0.78086877,  0.62469506,  0.3       ],
#        [ 1.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.62469506,  0.7808688 ,  0.6       ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]],
#       dtype=float32), 'intrinsic_cv': array([[64.,  0., 64.],
#        [ 0., 64., 64.],
#        [ 0.,  0.,  1.]], dtype=float32)}), ('overhead_camera', {'extrinsic_cv': array([[ 2.8595643e-03,  9.9999607e-01, -8.6962245e-08, -2.0099998e-01],
#        [ 7.0691872e-01, -2.0215493e-03, -7.0729196e-01,  7.7029693e-01],
#        [-7.0728922e-01,  2.0224855e-03, -7.0692158e-01,  1.0575197e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), 'cam2world_gl': array([[ 2.8594732e-03, -7.0691866e-01,  7.0728922e-01,  2.0400979e-01],
#        [ 9.9999601e-01,  2.0214319e-03, -2.0225048e-03,  2.0041752e-01],
#        [-5.9604645e-08,  7.0729208e-01,  7.0692146e-01,  1.2924081e+00],
#        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
#       dtype=float32), 'intrinsic_cv': array([[425.,   0., 320.],
#        [  0., 425., 256.],
#        [  0.,   0.,   1.]], dtype=float32)})])
# obs['image']['base_camera'].keys()
# ipdb> odict_keys(['rgb', 'depth', 'Segmentation'])
# obs['image']['overhead_camera'].keys()
# ipdb> odict_keys(['rgb', 'depth', 'Segmentation'])
# Image.fromarray(obs['image']['overhead_camera']['rgb']).save('temp_.png')
# Image.fromarray(obs['image']['base_camera']['rgb']).save('temp_base.png') # wrist

# google_robot_pick_coke_can 0.6
# google_robot_pick_horizontal_coke_can 0.7
# google_robot_pick_vertical_coke_can 0.8
# google_robot_pick_standing_coke_can 0.7
# google_robot_pick_object 0.4
# google_robot_move_near_v0 0.4
# google_robot_move_near_v1 0.4
# google_robot_move_near 0.4
# google_robot_open_drawer 0.0
# google_robot_open_top_drawer 0.0
# google_robot_open_middle_drawer 0.0
# google_robot_open_bottom_drawer 0.0
# google_robot_close_drawer 0.0
# google_robot_close_top_drawer 0.0
# google_robot_close_middle_drawer 0.0
# google_robot_close_bottom_drawer 0.0
# google_robot_place_in_closed_drawer 0.0
# google_robot_place_in_closed_top_drawer 0.0
# google_robot_place_in_closed_middle_drawer 0.0
# google_robot_place_in_closed_bottom_drawer 0.0
# google_robot_place_apple_in_closed_top_drawer 0.0
# widowx_spoon_on_towel 0.0
# widowx_carrot_on_plate 0.0
# widowx_stack_cube 0.0
# widowx_put_eggplant_in_basket 0.0

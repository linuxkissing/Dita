
from builtins import super

from Dataset_Lab.LabDataset import LabDataset




import pickle
import random
import numpy as np
import torch
import os

def convert_data_structure_lab(trajs, batch_transform, dataset_name=''):
    output_dict = {}
            # import ipdb;ipdb.set_trace()
    output_dict['dataset_name'] = dataset_name
    import cv2
    img = trajs['observation']['image'].cpu().numpy() # 180 320 3
    # print(img.min(), img.max())
    # img = np.pad(img, ( (70, 70), (0, 0), (0, 0)), 'constant', constant_values=0)
    output_dict['observation'] = {}
    output_dict['observation']['image_primary'] = img[...]
    
    if dataset_name == 'lab': # stride 4
        low = torch.tensor([-0.07399353384971619, -0.17020349204540253, -0.16437342762947083, -0.514319658279419, -0.28703776001930237, -0.44046279788017273, 0.0]).to(torch.float32)
        high = torch.tensor([0.13233336806297302, 0.17530526220798492, 0.16073314934969085, 0.35203295946121216, 0.2733728289604187, 0.6208988428115845, 1.0]).to(torch.float32)

    
    output_dict['action'] = torch.cat([trajs['action']['world_vector'], trajs['action']['rotation_delta'], 
                                ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)

    output_dict['state'] = torch.cat([trajs['action']['state_pose'], 
                            ( 1 - trajs['action']['gripper_closedness_action']).to(torch.float32)], dim=-1)

    if dataset_name not in ['no_norm', 'no_norm1', 'no_norm2']:
        output_dict['action'][...,:-1] = 2 * (output_dict['action'][...,:-1] - low[:-1]) / (high[:-1] - low[:-1] + 1e-8) -1
    else:
        # this is for Octo
        if dataset_name == 'no_norm1':
            mean = torch.tensor([0.009494006633758545, -0.009273790754377842, -0.023042574524879456, -0.007257933262735605, -0.0023885052651166916, 0.00760976318269968, 0.45395463705062866])
            std = torch.tensor([0.03161812573671341, 0.051145896315574646, 0.06707415729761124, 0.05436154454946518, 0.03937571868300438, 0.12943556904792786, 0.4978216290473938])
        elif dataset_name == 'no_norm2':

            mean = torch.tensor([0.007078998256474733, 0.002239587251096964, -0.011903811246156693, 0.0013399592135101557, -0.004025186412036419, 0.009980602189898491, 0.46560022234916687])
            std = torch.tensor([0.03535059094429016, 0.0555114820599556, 0.05755787715315819, 0.11843094974756241, 0.09467040002346039, 0.15725713968276978, 0.49822336435317993])
        else:
            mean = torch.tensor([0.013636833988130093, -0.0017261075554415584, -0.03187509998679161, -0.0023941127583384514, 0.003163236426189542, 0.017856508493423462, 0.3645598292350769])
            std = torch.tensor([0.02514560893177986, 0.027360064908862114, 0.04232143610715866, 0.04644988849759102, 0.027758773416280746, 0.09074866026639938, 0.48129966855049133])
        output_dict['action'][...,:-1] = (output_dict['action'][...,:-1] - mean[:-1]) / (std[:-1] + 1e-8) 

    output_dict['action'] = output_dict['action'].cpu().numpy()

    output_dict['task'] = {}
    output_dict['task']['language_instruction'] = trajs['instruction']
    
    # print(trajs['action']['loss_weight'].shape)
    output_dict['action_past_goal'] = torch.sum(trajs['action']['loss_weight'], dim=-1) == 0
    output_dict['ep_path'] = trajs['ep_path'] if 'ep_path' in trajs else None
     
    return batch_transform(output_dict)


class LabDataset_warp(LabDataset):

    def __init__(self, dataname='lab', data_path="", language_embedding_path="", traj_per_episode=8, traj_length=15, dataset_type=0, use_baseframe_action=False, 
                 split_type=None, stride=1, img_preprocess_type=2, data_cam_list=None, train_with_droid=False, obs_n_frames=2, include_target=0, euler_delta=0, remove_small_diff=0, selected_list=['left', 'corner', 'right'], cache_in_memory=0, out_size=224, is_training=True, data_aug= False):
        super().__init__(data_path, language_embedding_path, traj_per_episode, traj_length, dataset_type, use_baseframe_action, split_type, stride, img_preprocess_type, data_cam_list, train_with_droid, obs_n_frames, include_target, euler_delta=euler_delta, selected_list=selected_list, remove_small_diff=remove_small_diff, out_size=out_size, cache_in_memory=cache_in_memory, is_training=is_training, data_aug=data_aug)
        self.batch_transform = None
        self.dataname = dataname
        assert dataname!='lab'
        self.tag = 'lab_807'
    
    def set_batch_transform(self, batch_transform):
        self.batch_transform = batch_transform
    
    @torch.no_grad()
    def __getitem__(self, index):

        index = index % (len(self.data_cam_list))
        
        while True:

            try:
                
                data_pkl = self.get_cached_data_pkl(index)
                trajs = self.construct_traj(data_pkl, self.data_cam_list[index])

                # print('aa', self.euler_delta, self.data_transform1, self.data_transform, trajs['observation']['image'].shape, trajs['observation']['image'].cpu().numpy().min(), trajs['observation']['image'].cpu().numpy().max())
                if trajs is not None:
                    return convert_data_structure_lab(trajs, self.batch_transform, self.dataname)
                break

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                print(f"Fail to load data {self.data_cam_list[index]}", flush = True)
                index = random.randint(0, len(self.data_cam_list)-1)

        return trajs 
    
def quaternion_to_euler_radians(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), w**2 + z**2 - (x**2 + y**2))

    sinpitch = 2 * (w * y - z * x)
    pitch = np.arcsin(sinpitch)

    yaw = np.arctan2(2 * (w * z + x * y), w**2 + x**2 - (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], dtype=torch.float32)

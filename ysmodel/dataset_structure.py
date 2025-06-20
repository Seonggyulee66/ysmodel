import argparse
import os
import statistics

import torch
import torch.utils
import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.seg_utils import cal_iou_training
# from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils

yaml_root = './opencood/hypes_yaml/opcamera/Test.yaml'
hypes = yaml_utils.load_yaml(yaml_root)

# Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
# lidar: path, cameras:list of path}}}}
opencood_train_dataset = build_dataset(hypes,visualize=False,train=True)

### base Dataset
# print(opencood_train_dataset.retrieve_base_data(0, True).keys())    ## odict_keys(['650', '641', '659']) : cav_list
# print(opencood_train_dataset.retrieve_base_data(0, True)['650'].keys())    ## odict_keys(['ego', 'time_delay', 'camera_params', 'params', 'camera_np', 'bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png'])
# print(opencood_train_dataset.retrieve_base_data(0, True)['650']['params']['gt_transformation_matrix'].shape) ## odict_keys(['transformation_matrix', 'pairwise_t_matrix', 'camera_data', 'camera_intrinsic', 'camera_extrinsic', 'gt_dynamic', 'gt_static'])
# print(opencood_train_dataset.retrieve_base_data(0, True)['650']['params']['gt_transformation_matrix']) ## odict_keys(['transformation_matrix', 'pairwise_t_matrix', 'camera_data', 'camera_intrinsic', 'camera_extrinsic', 'gt_dynamic', 'gt_static'])
# print(opencood_train_dataset.retrieve_base_data(0, True)['641']['params']['gt_transformation_matrix']) ## odict_keys(['transformation_matrix', 'pairwise_t_matrix', 'camera_data', 'camera_intrinsic', 'camera_extrinsic', 'gt_dynamic', 'gt_static'])

## base camera dataset
# print(opencood_train_dataset.get_sample(0,0).keys())        ## odict_keys(['641', '650', '659'])
# print(opencood_train_dataset.get_sample(0,0)['650'].keys()) ## odict_keys(['ego', 'time_delay', 'camera_params', 'params', 'camera_np', 'bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png', 'object_bbx_cav', 'object_id'])

## cam intermediate fusion dataset   ||   odict_keys(['641', '650', '659'])에서 ego를 찾고 다른 cav와의 데이터를 정리
# print(opencood_train_dataset[0].keys())       ## odict_keys(['ego'])
# print(opencood_train_dataset[0]['ego'].keys())  ## odict_keys(['transformation_matrix', 'pairwise_t_matrix', 'camera_data', 'camera_intrinsic', 'camera_extrinsic', 'gt_dynamic', 'gt_static'])
# print(opencood_train_dataset[0]['ego']['camera_data'].shape) ## (3, 4, 512, 512, 3) ==  (L, M, H ,W , C)
# print(opencood_train_dataset[0]['ego']['gt_static']) ## (1, 256,256)

# train_loader = DataLoader(opencood_train_dataset, batch_size=5, shuffle=True, num_workers=1, collate_fn=opencood_train_dataset.collate_batch)
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
import math


# ##################### For Checking
# import logging
# import os
# log_dir = './logs'
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'output.log')

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,  # 로그 레벨 설정 (INFO 레벨 이상)
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file_path),  # 로그를 파일에 기록
#         logging.StreamHandler()  # 콘솔에 로그 출력
#     ]
# )

data_dir = './train'

save_dir = './merge_image_result'
os.makedirs(save_dir,exist_ok=True)

for root,dirs,files in os.walk(data_dir):
    if dirs:
        for dir in dirs:
            os.makedirs(os.path.join(save_dir,root,dir),exist_ok=True)

scenario_list = os.listdir(data_dir)

def merge_bev_image(ego_img, cav_img, T):
    """
    Merge BEV images of ego and cav vehicles using the transformation matrix T.
    """
    # 이미지 크기
    
    ### CARLA world Coordinate와 BEV Coordinate의 차이때문에 값을 바꿔줘야함
    T[[0,1], 3] = T[[1,0], 3]
    T[1,3] = -T[1,3]

    # 이미지 정보
    h, w = cav_img.shape[:2]
    canvas = np.zeros((2*h, 2*w, 3), dtype=np.uint8)    ## 배경 검은색
    # canvas = np.ones((2*h, 2*w, 3), dtype=np.uint8) * 255    ## 배경 흰색
    center = (w, h)  # canvas 중앙 좌표

    # meter to pixel 변환 스케일 (예: 0.4m/pixel → 1m = 2.5px)
    image_size = cav_img.shape[0]   # 이미지 크기 (가로)
    bev_range = 100.0               # BEV 범위 (m)
    scale = image_size/bev_range    # 한 픽셀당 몇 미터인지 계산

    # 회전 행렬과 이동 벡터 (pixel 단위로 변환)
    R = T[:2, :2]           ## 회전 행렬
    t = T[:2, 3] * scale    ## Transition 행렬 스케일링

    # 최종 affine 행렬 만들기
    affine = np.hstack([R, t.reshape(2, 1)])

    # 중심 정렬 위한 변환 (캔버스 중앙 기준)
    # affine은 canvas의 중앙에서 진행하기 때문에 이에 맞추어 이동 후 affine 적용
    # affine 이후 canvas의 중앙으로 이동
    to_affine_center = np.array([[1, 0, -center[0]//2],
                                [0, 1, -center[1]//2],
                                [0, 0, 1]], dtype=np.float64)
    to_image_center = np.array([[1, 0, center[0]],
                                [0, 1, center[1]],
                                [0, 0, 1]], dtype=np.float64)

    # affine 행렬을 3x3으로 확장해서 곱셈 가능하게
    affine_3x3 = np.vstack([affine, [0, 0, 1]])
    final_affine = to_image_center @ affine_3x3 @ to_affine_center
    final_affine = final_affine[:2]  # warpAffine용 2x3 행렬
    
    # 이미지 변환
    if ego_img.shape[0] == h:
        canvas[center[1] - h//2:center[1] + h//2, center[0] - w//2:center[0] + w//2] = ego_img
    else:
        canvas = ego_img
    warped_cav1 = cv2.warpAffine(cav_img, final_affine, (2*w, 2*h))

    # 두 이미지 합성 (투명도 조절)
    alpha = 0.5
    merged = cv2.addWeighted(canvas, alpha, warped_cav1, 1 - alpha, 0)
    # cv2.circle(canvas, center, 1, (0, 0, 255), -1)

    return merged

# from opencood.pcdet_utils.iou3d_nms import iou3d_nms_utils

yaml_root = './opencood/hypes_yaml/opcamera/Test.yaml'
hypes = yaml_utils.load_yaml(yaml_root)

# Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
# lidar: path, cameras:list of path}}}}
opencood_train_dataset = build_dataset(hypes,visualize=False,train=True)

import cv2
from matplotlib import pyplot as plt
import numpy as np

##### !!!!!!!!!!!!!!!!!!!!!!!
##### 여기서는 basedataset.py line 142에 random을 끄고 사용하였음

# image_types_list = ['bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png']
image_types_list = ['bev_dynamic.png']
distance_list = []
com_range = 100

def cav_distance_cal(selected_cav_base, ego_lidar_pose):
    """
    Calculate a certain cav's distance to the ego vehicle,

    Parameters
    ----------
    selected_cav_base : dict
    ego_lidar_pose : list

    Returns
    -------
    The distance of this two vehicle (float);
    """
    distance = \
        math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                   ego_lidar_pose[0]) ** 2 + (
                          selected_cav_base['params'][
                              'lidar_pose'][1] - ego_lidar_pose[
                              1]) ** 2)

    return distance

for image_type in image_types_list:
    for scenario_id in range(len(scenario_list)):
        
        cav_id_list = list(opencood_train_dataset.scenario_database[scenario_id].keys())
        tick_list = list(opencood_train_dataset.scenario_database[scenario_id][cav_id_list[0]].keys())         ##  ex) odict_keys(['000070', '000072', '000074', '000076', '000078', '000080', '000082', '000084', '000086',
        tick_list.remove('ego')
        num_ticks = len(tick_list)

        check_list = []
        for tick in range(num_ticks):
            ego_img_list = []
            cav_img_list = []
            cav_t_matrix_list = []

            data_sample = opencood_train_dataset.retrieve_base_data((scenario_id,tick),True)
            for cav_id, cav_content in data_sample.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    break
            assert cav_id == list(data_sample.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -999
            assert len(ego_lidar_pose) > 0

            for cav_id, selected_cav_base in data_sample.items():
            ##  selected_cav_base : base dataset format
            ##  odict_keys(['ego', 'time_delay', 'camera_params', 'params', 'camera_np', 'bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png', 'object_bbx_cav', 'object_id', 'object_bbx_ego', 'object_bbx_ego_mask'])
                distance = cav_distance_cal(selected_cav_base,
                                                        ego_lidar_pose)
                distance_list.append(distance)
                if distance > com_range:
                    continue
                
                ##################### For Checking
                if selected_cav_base['ego'] == True:
                    print(f"label dataset Scenario ID : {scenario_id} || Ego : {cav_id}")
                
            #     if selected_cav_base['ego'] == True :
            #         # print(f'ego : {cav_id}')
            #         ego_img = selected_cav_base[image_type]
            #         ego_img_list.append(ego_img)
            #     else:
            #         # print(f'cav : {cav_id}')
            #         cav_img = selected_cav_base[image_type]
            #         Transformation_matrix = selected_cav_base['params']['gt_transformation_matrix']
            #         cav_img_list.append(cav_img)
            #         cav_t_matrix_list.append(Transformation_matrix)
            
            # for i in range(len(cav_img_list)):
            #     if i == 0:
            #         merged_img = merge_bev_image(ego_img=ego_img_list[0], cav_img=cav_img_list[i], T =cav_t_matrix_list[i])
            #     else:
            #         merged_img = merge_bev_image(ego_img=merged_img, cav_img=cav_img_list[i], T=cav_t_matrix_list[i])
            
            # tick = tick_list[tick]
            # image_type_name, _ = os.path.splitext(image_type)
            # ### merge_bev_image는 모든 cav에 대해서 merge하기 때문데 같은 이미지를 모든 cav파일에 저장
            # for iter_cav_id in cav_id_list: 
            #     print(f'senario id : {scenario_id} || cav_id : {iter_cav_id}  || tick : {tick} || image type : {image_type_name} saved ...') 
            #     save_file_path = os.path.join(save_dir, 'train', scenario_list[scenario_id], iter_cav_id, f'{tick}_merged_{image_type_name}.png')

            #     cv2.imwrite(save_file_path, merged_img)

            #     # plt.figure(figsize=(10, 10))
            #     # plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
            #     # plt.title("Merged BEV")
            #     # plt.axis("off")
            #     # plt.show() 




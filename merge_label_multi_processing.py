import os
import math
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset

# ---------------------------- 설정 ----------------------------
data_dir = '/scratch/sglee6/opv2v_dataset/train'
yaml_root = './opencood/hypes_yaml/opcamera/Test.yaml'
image_types_list = ['bev_dynamic.png', 'bev_static.png', 'bev_lane.png', 'bev_visibility.png', 'bev_visibility_corp.png']
# image_types_list = ['bev_lane.png']
com_range = 100  # meters

scenario_list = os.listdir(data_dir)
scenario_list = sorted(scenario_list)
hypes = yaml_utils.load_yaml(yaml_root)
opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)

for scenario_id in range(len(scenario_list)):
    scenario_name = scenario_list[scenario_id]
    cav_id_list = list(opencood_train_dataset.scenario_database[scenario_id].keys())
    for iter_cav_id in cav_id_list:
        save_dir = os.path.join(data_dir, scenario_name, iter_cav_id)
        
# --------------------- BEV 병합 함수 ---------------------
def merge_bev_image(ego_img, cav_img, T):
    T[[0, 1], 3] = T[[1, 0], 3]
    T[1, 3] = -T[1, 3]

    h, w = cav_img.shape[:2]
    canvas = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    center = (w, h)

    image_size = cav_img.shape[0]
    bev_range = 100.0
    scale = image_size / bev_range

    R = T[:2, :2]
    t = T[:2, 3] * scale
    affine = np.hstack([R, t.reshape(2, 1)])

    to_affine_center = np.array([[1, 0, -center[0] // 2],
                                 [0, 1, -center[1] // 2],
                                 [0, 0, 1]], dtype=np.float64)
    to_image_center = np.array([[1, 0, center[0]],
                                [0, 1, center[1]],
                                [0, 0, 1]], dtype=np.float64)
    affine_3x3 = np.vstack([affine, [0, 0, 1]])
    final_affine = to_image_center @ affine_3x3 @ to_affine_center
    final_affine = final_affine[:2]

    if ego_img.shape[0] == h:
        canvas[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2] = ego_img
    else:
        canvas = ego_img

    warped_cav = cv2.warpAffine(cav_img, final_affine, (2 * w, 2 * h))
    alpha = 0.5
    merged = cv2.addWeighted(canvas, alpha, warped_cav, 1 - alpha, 0)
    return merged

# --------------------- 거리 계산 함수 ---------------------
def cav_distance_cal(selected_cav_base, ego_lidar_pose):
    return math.sqrt(
        (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
        (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
    )

# --------------------- 한 Tick 처리 함수 ---------------------
def process_one_tick(args):
    scenario_id, tick_idx, image_type = args
    scenario_name = scenario_list[scenario_id]
    cav_id_list = list(opencood_train_dataset.scenario_database[scenario_id].keys())
    tick_list = list(opencood_train_dataset.scenario_database[scenario_id][cav_id_list[0]].keys())
    tick_list.remove('ego')
    tick = tick_list[tick_idx]

    data_sample = opencood_train_dataset.retrieve_base_data((scenario_id, tick_idx), True)

    cav_list = []
    ego_img_list = []
    cav_img_list = []
    cav_t_matrix_list = []

    for cav_id, cav_content in data_sample.items():
        if cav_content['ego']:
            ego_lidar_pose = cav_content['params']['lidar_pose']
            break

    for cav_id, selected_cav_base in data_sample.items():
        if cav_distance_cal(selected_cav_base, ego_lidar_pose) > com_range:
            continue
        
        cav_list.append(cav_id)
        
        if selected_cav_base['ego']:
            ego_img = selected_cav_base[image_type]
            ego_img_list.append(ego_img)
        else:
            cav_img = selected_cav_base[image_type]
            T = selected_cav_base['params']['gt_transformation_matrix']
            cav_img_list.append(cav_img)
            cav_t_matrix_list.append(T)
    
    if not cav_img_list:
        print(f"[!] Skip: No neighbor CAVs in range for Scenario {scenario_id}, Tick {tick}, Type {image_type}")
        return
    
    for i in range(len(cav_img_list)):
        if i == 0:
            merged_img = merge_bev_image(ego_img_list[0], cav_img_list[i], cav_t_matrix_list[i])
        else:
            merged_img = merge_bev_image(merged_img, cav_img_list[i], cav_t_matrix_list[i])

    image_type_name, _ = os.path.splitext(image_type)
    for iter_cav_id in cav_id_list:
        save_dir = os.path.join(data_dir, scenario_name, iter_cav_id)
        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, f'{tick}_merged_{image_type_name}.png')
        cv2.imwrite(save_file_path, merged_img)

    print(f"[✓] Scenario {scenario_id}, Tick {tick}, Type {image_type_name} 저장 완료.")

# --------------------- 메인 실행 ---------------------

task_list = []
for image_type in image_types_list:
    for scenario_id in range(len(scenario_list)):
        tick_list = list(opencood_train_dataset.scenario_database[scenario_id][
            list(opencood_train_dataset.scenario_database[scenario_id].keys())[0]
        ].keys())
        tick_list.remove('ego')
        for tick_idx in range(len(tick_list)):
            task_list.append((scenario_id, tick_idx, image_type))

print(f"총 작업 수: {len(task_list)}")
print(f"CPU 코어 수: {cpu_count()}")
print(f"Working CPU 코어 수: {cpu_count()-4}")

with Pool(processes=max(1, cpu_count() - 4)) as pool:
    pool.map(process_one_tick, task_list)
    
print("All work is done")

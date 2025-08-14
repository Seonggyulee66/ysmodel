import os
import math
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset

"""
!!!! 실행 시 yaml파일에 label_generation 을 true로 설정해야함 !!!!!!!
"""

# ---------------------------- 설정 ----------------------------
data_type = 'train' 
data_dir = '/scratch/sglee6/opv2v_dataset/' + data_type
yaml_root = './opencood/hypes_yaml/opcamera/Test.yaml'
image_types_list = ['bev_dynamic.png', 'bev_static.png', 'bev_lane.png']
# image_types_list = ['bev_lane.png']

scenario_list = os.listdir(data_dir)
scenario_list = sorted(scenario_list)
hypes = yaml_utils.load_yaml(yaml_root)
if data_type == 'train':
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, validate=False)
else:
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True, validate=True)
    
for scenario_id in range(len(scenario_list)):
    scenario_name = scenario_list[scenario_id]
    cav_id_list = list(opencood_train_dataset.scenario_database[scenario_id].keys())
    for iter_cav_id in cav_id_list:
        save_dir = os.path.join(data_dir, scenario_name, iter_cav_id)
        

# --------------------- BEV 병합 함수 ---------------------
# ===================== 설정 =====================
CANVAS_SCALE = 1.5          # <-- 여기만 바꿔서 1.25, 1.5, 2.0 등 원하는 배율로
BEV_RANGE_M = 100.0         # 기존 bev_range (=100m) 유지
ALPHA_BLEND = 0.5           # 합성 비율

# --------------------- BEV 병합 함수 ---------------------
def merge_bev_image(ego_or_canvas_img, cav_img, T,
                    canvas_scale=CANVAS_SCALE,
                    bev_range=BEV_RANGE_M,
                    alpha=ALPHA_BLEND):
    """
    ego_or_canvas_img:
      - 첫 호출: ego의 BEV 이미지 (원본 HxW)
      - 이후 호출: 이미 누적된 캔버스(확대된 Hc x Wc)
    cav_img: 이웃 CAV의 BEV 이미지 (원본 HxW)
    T: cav -> ego 로의 SE(2) (앞서 사용하던 것과 동일. 아래에서 x/y 축 교환 및 부호 조정 포함)
    """
    # 좌표계 보정 (기존 코드 유지)
    T = T.copy()
    T[[0, 1], 3] = T[[1, 0], 3]
    T[1, 3] = -T[1, 3]

    # 소스(각 BEV 타일) 해상도
    h_src, w_src = cav_img.shape[:2]
    assert h_src == w_src, "정사각 BEV를 가정합니다. (필요시 소스 중심만 잘 설정하면 직사각도 동작함)"

    # ----- 캔버스 준비 -----
    # 만약 전달된 ego_or_canvas_img가 이미 큰 캔버스면 그대로 사용,
    # 아니면 첫 호출이므로 새 캔버스를 생성하고 ego 이미지를 중앙 배치.
    if ego_or_canvas_img.shape[:2] == (h_src, w_src):
        # 새 캔버스 생성
        Hc = int(round(canvas_scale * h_src))
        Wc = int(round(canvas_scale * w_src))
        canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)

        # 목적지(캔버스) 중심
        dst_cx, dst_cy = Wc // 2, Hc // 2

        # ego 이미지를 중앙에 삽입
        y1 = dst_cy - h_src // 2
        y2 = y1 + h_src
        x1 = dst_cx - w_src // 2
        x2 = x1 + w_src
        canvas[y1:y2, x1:x2] = ego_or_canvas_img
    else:
        # 이미 누적된 캔버스
        canvas = ego_or_canvas_img
        Hc, Wc = canvas.shape[:2]
        dst_cx, dst_cy = Wc // 2, Hc // 2

    # ----- CAV 타일을 캔버스에 워핑 -----
    # (m -> px) 스케일: 원본 타일 해상도를 기준으로 유지
    px_per_m = (h_src / float(bev_range))

    R = T[:2, :2]
    t = T[:2, 3] * px_per_m
    affine_2x3 = np.hstack([R, t.reshape(2, 1)])

    # '소스 중심'과 '캔버스 중심'을 분리해서 일반화
    src_cx, src_cy = w_src * 0.5, h_src * 0.5
    to_src_center = np.array([[1, 0, -src_cx],
                              [0, 1, -src_cy],
                              [0, 0,    1   ]], dtype=np.float64)
    to_dst_center = np.array([[1, 0,  dst_cx],
                              [0, 1,  dst_cy],
                              [0, 0,    1   ]], dtype=np.float64)

    A3 = np.vstack([affine_2x3, [0, 0, 1]])        # 3x3
    final_affine_3x3 = to_dst_center @ A3 @ to_src_center
    final_affine_2x3 = final_affine_3x3[:2]

    warped_cav = cv2.warpAffine(cav_img, final_affine_2x3, (Wc, Hc))
    merged = cv2.addWeighted(canvas, alpha, warped_cav, 1 - alpha, 0)
    return merged


## ego 주위에 Communication 범위 안에 cav가 없을 경우
def create_canvas_with_ego(ego_img, canvas_scale=CANVAS_SCALE):
    h, w = ego_img.shape[:2]
    Hc = int(round(canvas_scale * h))
    Wc = int(round(canvas_scale * w))
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    cy, cx = Hc // 2, Wc // 2
    y1 = cy - h // 2; y2 = y1 + h
    x1 = cx - w // 2; x2 = x1 + w
    canvas[y1:y2, x1:x2] = ego_img
    return canvas

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

    total_cav_count = 0
    for cav_id, selected_cav_base in data_sample.items():
        if cav_distance_cal(selected_cav_base, ego_lidar_pose) > BEV_RANGE_M:
            continue
        total_cav_count += 1
        ### For Checking
        # if selected_cav_base['ego'] == True:
        #     print(f"label dataset Scenario ID : {scenario_id} || Ego : {cav_id}")
        
        if selected_cav_base['ego']:
            ego_img = selected_cav_base[image_type]
            ego_img_list.append(ego_img)
        else:
            cav_img = selected_cav_base[image_type]
            T = selected_cav_base['params']['gt_transformation_matrix']
            cav_img_list.append(cav_img)
            cav_t_matrix_list.append(T)
    
        if not cav_img_list:
            print(f"[!] Single but named Merged: No neighbor CAVs in range for Scenario {scenario_list[scenario_id]}, Tick {tick}, Type {image_type}")
            merged_img = create_canvas_with_ego(ego_img_list[0], canvas_scale=CANVAS_SCALE)
        else:
            for i in range(len(cav_img_list)):
                if i == 0:
                    merged_img = merge_bev_image(ego_img_list[0], cav_img_list[i], cav_t_matrix_list[i],
                                                canvas_scale=CANVAS_SCALE, bev_range=BEV_RANGE_M)
                else:
                    merged_img = merge_bev_image(merged_img,      cav_img_list[i], cav_t_matrix_list[i],
                                                canvas_scale=CANVAS_SCALE, bev_range=BEV_RANGE_M)


    image_type_name, _ = os.path.splitext(image_type)
    for iter_cav_id in cav_id_list:
        save_dir = os.path.join(data_dir, scenario_name, iter_cav_id)
        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, f'{tick}_merged_{image_type_name}.png')
        cv2.imwrite(save_file_path, merged_img)

    print(f"[✓] Scenario {scenario_list[scenario_id]}, Tick {tick}, Type {image_type_name} 저장 완료. || Total CAV count : {total_cav_count}")
    
    return total_cav_count

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

with Pool(max(1,cpu_count()-4)) as pool:
    total_cav_count = pool.map(process_one_tick, task_list)

max_cav_count = max(total_cav_count)
print(f"All work is done || Max Number of CAV is {max_cav_count}")

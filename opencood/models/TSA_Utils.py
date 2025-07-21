import torch 
import numpy as np
import json
import yaml
import os 
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn as nn 

class PositionalEncodeing(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float,device='cpu') -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model,device=device)   # positional encoding tensor 생성
        position = torch.arange(0,seq_len, dtype=torch.float,device=device).unsqueeze(1) ## shape : [seq_len, 1]
        _2i = torch.arange(0,d_model,2,dtype= torch.float,device=device)   ## 여기서 i는 Embedding 벡터의 차원 인덱스를 의미 2i는 짝수 인덱스

        ## 0::2 => index 0부터 2 step씩 가겠다.
        pe[:,0::2] = torch.sin(position/10000**(_2i/d_model))   ## 모든 sequence에서 짝수열의 값은 sin함수 사용 
        pe[:,1::2] = torch.cos(position/10000**(_2i/d_model))   ## 모든 sequence에서 짝수열의 값은 cos함수 사용 

        pe = pe.unsqueeze(0)    ## 차원 추가 => shape : [1, seq_len, d_model]
        ## register_buffer
        ## 모듈내에서 pe이름으로 접근가능, 학습은 되지 않음, model이 다른 device로 갈때 같이 감
        self.register_buffer('pe',pe)

    def forward(self, x):
        # print('Positional Encoding finished')
        x = x + self.pe[:,:x.shape[1],:].requires_grad_(False)
        return self.dropout(x)

def get_json(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def get_yaml(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_data_path(path, cam):
    data = []
    if 'config' in cam:
        for f_name in os.listdir(path):
            if f_name.endswith('.yaml'):
                data.append(f_name)
    else:
        for f_name in os.listdir(path):
            if f_name.endswith(cam + '.png'):
                data.append(f_name)
    return data

def get_cam_parameter(cam, config_path):
    config = get_yaml(config_path)
    extrinsic_param = np.array(config[cam]['extrinsic'])
    intrinsic_param = np.array(config[cam]['intrinsic'])
    return extrinsic_param, intrinsic_param

def get_agent_pose(config_path, device):
    config = get_yaml(config_path)
    pose_x = np.array(config['true_ego_pos'])[0]
    pose_y = np.array(config['true_ego_pos'])[1]
    return torch.tensor([pose_x, pose_y], device=device, dtype = torch.float32)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0


def grid_sample_1d(feats, offsets):
    """
    feats: (B, C, L)
    offsets: (B, L_out)
    """
    B, C, L = feats.shape
    B_offsets, L_out = offsets.shape

    device = feats.device

    # pos shape must match offsets batch
    pos = torch.linspace(0, L - 1, steps=L_out, device=device).unsqueeze(0).expand(B_offsets, -1)
    print(f"pos shape: {pos.shape} | offsets shape: {offsets.shape}")

    vgrid = pos + offsets

    vgrid_scaled = 2.0 * (vgrid / max(L - 1, 1)) - 1.0

    vgrid_scaled = vgrid_scaled.unsqueeze(-1)
    dummy_y = torch.zeros_like(vgrid_scaled)
    grid = torch.cat([vgrid_scaled, dummy_y], dim=-1)

    feats = feats.unsqueeze(-2)
    out = F.grid_sample(
        feats,
        grid.unsqueeze(2),
        mode='bilinear',
        align_corners=True,
        padding_mode='zeros'
    )
    return out.squeeze(-2).squeeze(-2)



def normalize_grid(arange, dim=1,out_dim=-1):
    ## Normalize 1d sequnece to range between -1 to 1

    n = arange.shape[-1]
    return 2.0 * arange / max(n-1,1) - 1.0

def organise_input(Scene_path, pair_folder_name, tick,cam_name_list, vehicle_type = 'ego',device = 'cpu'):
    dictionary = get_yaml(os.path.join(Scene_path,pair_folder_name,tick))
    img_list = dictionary[vehicle_type]['img_paths_list']
    for i in range(len(img_list)):
        img = Image.open(img_list[i]).convert('RGB')
        tensor_img = ToTensor()(img)
        tensor_img = torch.tensor(tensor_img, dtype = torch.float32).unsqueeze(0)
        tensor_img = tensor_img.to(device)
        img_list[i] = tensor_img

    extrinsic_list = []
    intrinsic_list = []
    for cam in cam_name_list:
        extrinsic_param, intrinsic_param = get_cam_parameter(cam, dictionary[vehicle_type]['cam_feature_path'])
        extrinsic_param = torch.tensor(extrinsic_param, device=device, dtype = torch.float32)
        intrinsic_param = torch.tensor(intrinsic_param, device=device, dtype = torch.float32)
        extrinsic_list.append(extrinsic_param)
        intrinsic_list.append(intrinsic_param)

    global_pose = get_agent_pose(dictionary[vehicle_type]['cam_feature_path'], device=device)

    return img_list, extrinsic_list, intrinsic_list, global_pose 


#############################
# 4. Large BEV Mapping 함수
#############################
def map_bev_to_large(bev_feature, large_size=400, bev_size=200, offset=(0, 0)):
    """
    bev_feature: (B, bev_size, bev_size, embed_dim)
    large_size: 최종 large BEV map의 spatial 크기 (400×400)
    offset: (dx, dy) – BEV feature를 large map에 배치할 때의 추가 offset.
            ego 차량은 (0,0), 다른 차량은 global 좌표 차이를 반영한 offset.
    """
    B, H, W, C = bev_feature.shape
    # large BEV 맵 초기화
    large_bev = torch.zeros(B, large_size, large_size, C, device=bev_feature.device)
    # 기본 배치 위치: large map의 중앙에 bev_feature의 중앙이 오도록 배치.
    base_x = large_size // 2 - bev_size // 2 + offset[0]
    base_y = large_size // 2 - bev_size // 2 + offset[1]
    # 간단하게 복사 (실제 시스템에서는 interpolation이나 warp 적용할 수 있음)
    large_bev[:, base_y:base_y+bev_size, base_x:base_x+bev_size, :] = bev_feature
    return large_bev

# def combine_vehicle_features(ego_bev,other_bev,input_bev_size,large_size,relative_vector,embed_dim,output_seq_len = 80000,device = 'cpu'):
#     # -- (3) 각 차량의 BEV를 400×400 Large BEV Map에 Mapping --
#     # Ego 차량은 ego-centric으로 (offset (0,0))
#     ego_large_bev = map_bev_to_large(ego_bev, large_size=large_size, bev_size=input_bev_size, offset=(0, 0))
#     # 다른 차량은 글로벌 좌표 차이를 offset으로 반영 (여기서는 간단히 픽셀 offset으로 사용)
#     other_large_bev = map_bev_to_large(other_bev, large_size=large_size, bev_size=input_bev_size,
#                                         offset=(relative_vector[0].item(), relative_vector[1].item()))
    
#     # 두 차량의 large BEV map을 단순 합산 (실제에서는 fusion 전략 선택)
#     combined_large_bev = ego_large_bev + other_large_bev
#     # print("Combined large BEV map shape:", combined_large_bev.shape)

#     mask = (combined_large_bev != 0.)
 
#     flatten_input_bev = combined_large_bev.view(1,-1,embed_dim).to(device)          # (Batch, masked sequence, embed_dim)
#     pe_module = PositionalEncodeing(d_model=flatten_input_bev.shape[2],seq_len=flatten_input_bev.shape[1],dropout=0.1,device=device)
#     flatten_input_bev = pe_module(flatten_input_bev)
#     input_bev = flatten_input_bev.reshape_as(combined_large_bev)
#     flatten_input_bev = input_bev[mask].view(1,-1,embed_dim).to(device)    # (Batch, non masked sequence, embed_dim)
    
#     len_sequence = flatten_input_bev.shape[1]
#     if len_sequence > output_seq_len:
#         flatten_input_bev_padded = flatten_input_bev[:,:output_seq_len,:]
#     else:
#         padding_size = output_seq_len - len_sequence
#         flatten_input_bev_padded = F.pad(flatten_input_bev, (0,0,0,padding_size))

#     return flatten_input_bev_padded, mask, combined_large_bev

### Running 시 문제가 잇어서 임의로 고침
def combine_vehicle_features(
    ego_bev,
    other_bev,
    input_bev_size,
    large_size,
    relative_vector,
    embed_dim,
    output_seq_len=80000,
    device='cpu'
):
    # (1) Map ego BEV to large canvas at origin
    ego_large_bev = map_bev_to_large(ego_bev, large_size=large_size, bev_size=input_bev_size, offset=(0, 0))
    
    # (2) Map other vehicle BEV with relative offset
    offset_x, offset_y = int(relative_vector[0].item()), int(relative_vector[1].item())
    other_large_bev = map_bev_to_large(other_bev, large_size=large_size, bev_size=input_bev_size, offset=(offset_x, offset_y))

    # (3) Combine BEV maps (simple addition for now)
    combined_large_bev = ego_large_bev + other_large_bev  # shape: (1, H, W, C)
    
    # (4) Create a mask (where data is not zero)
    mask = (combined_large_bev != 0.)

    # (5) Flatten for positional encoding
    B, H, W, C = combined_large_bev.shape
    assert C == embed_dim, f"Expected embed_dim={embed_dim}, but got {C}"

    flat_input = combined_large_bev.view(B, -1, C)          # (B, H*W, C)
    flat_mask = mask[..., 0]               # (B, H, W)
    flat_mask = flat_mask.view(B, -1)      # (B, H*W)

    # (6) Positional encoding
    pe_module = PositionalEncodeing(d_model=C, seq_len=H * W, dropout=0.1, device=device)
    flat_input_pe = pe_module(flat_input)

    # (7) Masked flattening
    masked_flat_input = flat_input_pe[flat_mask.bool().reshape(B, -1)].view(1, -1, embed_dim)
    # (8) Padding to fixed length
    valid_len = masked_flat_input.shape[1]
    if valid_len > output_seq_len:
        masked_flat_input = masked_flat_input[:, :output_seq_len, :]
    else:
        pad_len = output_seq_len - valid_len
        masked_flat_input = F.pad(masked_flat_input, (0, 0, 0, pad_len))  # pad along sequence dim

    return masked_flat_input, mask, combined_large_bev

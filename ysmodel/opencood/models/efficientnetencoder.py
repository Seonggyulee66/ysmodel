from TSA_Utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision

#############################
# 1. EfficientNet‑b0 Feature Extractor
############################
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        # pretrained EfficientNet‑b0 (torchvision 0.13 이상에서 사용 가능)
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        efficientnet = torchvision.models.efficientnet_b0(weights=weights)
        #efficientnet = models.efficientnet_b0(pretrained=True)
        # classification head 제거: features 만 사용
        self.features = efficientnet.features
        # (예제에서는 별도 avgpool이나 fc는 사용하지 않음)
        
    def forward(self, x):
        """
        x: tensor of shape (B, 3, 600, 800) – 원래 카메라 입력 (800×600)
        → EfficientNet‑b0는 224×224 입력을 기대하므로 리사이즈 후 feature 추출.
        output size : [1, 1280, 7, 7] => 현재 setting으로 
        """
        # 리사이즈 (bilinear interpolation)
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feat = self.features(x_resized)  # (B, C, H_feat, W_feat)
        return feat
    
#############################
# 2. Cross Attention 모듈
#############################
class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        # query: (query_len, B, embed_dim)
        # key, value: (key_len, B, embed_dim)
        attn_output, _ = self.mha(query, key, value)
        query = self.norm1(query + attn_output)
        ff_output = self.ff(query)
        query = self.norm2(query + ff_output)
        return query
    
#############################
# 3. BEV Generator 모듈
#############################
class BEVGenerator(nn.Module):
    def __init__(self, bev_size=200, embed_dim=256, num_heads=8, num_cameras=4):
        """
        bev_size: BEV feature의 spatial size (200×200)
        embed_dim: 임베딩 차원 (예제에서는 256)
        num_heads: cross attention의 head 수
        num_cameras: 각 차량의 카메라 수 (4)
        """
        super(BEVGenerator, self).__init__()
        self.bev_size = bev_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        
        # EfficientNet의 출력 채널 (예제에서는 1280) → embed_dim으로 투영
        self.camera_feat_proj = nn.Conv2d(1280, embed_dim, kernel_size=1)
        # 카메라 위치 및 파라미터 정보를 임베딩할 선형층.
        # 카메라 intrinsic은 (3×3)=9, extrinsic은 (4×4)=16, 그리고 grid 좌표 2 → 총 27 차원.
        self.pos_encoding_fc = nn.Linear(27, embed_dim)
        # learnable한 BEV query (200×200 = 40,000 token)
        self.bev_queries = nn.Parameter(torch.randn(bev_size * bev_size, embed_dim))
        # cross attention layer
        self.cross_attn = CrossAttentionLayer(embed_dim, num_heads)
        
    def forward(self, camera_feats, camera_intrinsics, camera_extrinsics):
        """
        camera_feats: 리스트 길이 num_cameras, 각 원소는 tensor shape (B, 1280, H_feat, W_feat)
        camera_intrinsics: 리스트 길이 num_cameras, 각 원소: (3, 3) tensor
        camera_extrinsics: 리스트 길이 num_cameras, 각 원소: (4, 4) tensor

        Output : (Batch, bev_size, bev_size, emb_dim)
        """
        B = camera_feats[0].shape[0]
        key_list = []
        value_list = []
        
        for i in range(self.num_cameras):
            feat = camera_feats[i]  # (B, 1280, H_feat, W_feat)
            feat_proj = self.camera_feat_proj(feat)  # (B, embed_dim, H_feat, W_feat)
            B, C, H_feat, W_feat = feat_proj.shape
            
            # --- Positional Encoding 추가 ---
            # 1) 생성: 각 픽셀의 grid 좌표 (x, y)
            y_coords = torch.linspace(0, H_feat - 1, H_feat, device=feat_proj.device)
            x_coords = torch.linspace(0, W_feat - 1, W_feat, device=feat_proj.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H_feat, W_feat)
            grid = torch.stack([xx, yy], dim=-1)  # (H_feat, W_feat, 2)
            grid_flat = grid.view(-1, 2)  # (H_feat*W_feat, 2)
            
            # 2) 카메라 파라미터: intrinsic (3×3)와 extrinsic (4×4)를 flatten한 후 각 픽셀마다 동일하게 결합.
            cam_params = torch.cat([camera_intrinsics[i].reshape(-1), camera_extrinsics[i].reshape(-1)], dim=0)  # (9+16=25,)
            cam_params_repeat = cam_params.unsqueeze(0).expand(grid_flat.shape[0], -1)  # (H_feat*W_feat, 25)
            
            # 3) grid와 카메라 파라미터를 concat → (H_feat*W_feat, 27)
            pos_input = torch.cat([grid_flat, cam_params_repeat], dim=-1)
            # 4) 선형투영
            pos_encoding = self.pos_encoding_fc(pos_input)  # (H_feat*W_feat, embed_dim)
            pos_encoding = pos_encoding.transpose(0, 1).view(1, self.embed_dim, H_feat, W_feat)
            
            # 5) positional encoding을 feature에 더하기 (Element-wise)
            feat_proj = feat_proj + pos_encoding  # (B, embed_dim, H_feat, W_feat)
            
            # flatten spatial 차원 → (B, H_feat*W_feat, embed_dim)
            feat_flat = feat_proj.flatten(2).transpose(1, 2)
            key_list.append(feat_flat)
            value_list.append(feat_flat)
            
        # 모든 카메라의 key/value를 concat: (B, total_tokens, embed_dim)
        keys = torch.cat(key_list, dim=1)
        values = torch.cat(value_list, dim=1)
        
        # BEV query 준비: (B, bev_size*bev_size, embed_dim)
        bev_queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        
        # MultiheadAttention는 (seq_len, B, embed_dim) 형태를 기대하므로 전치
        query_t = bev_queries.transpose(0, 1)      # (bev_query_len, B, embed_dim)
        keys_t = keys.transpose(0, 1)                # (key_len, B, embed_dim)
        values_t = values.transpose(0, 1)            # (key_len, B, embed_dim)
        
        # Cross Attention 수행 → (bev_query_len, B, embed_dim)
        bev_attended = self.cross_attn(query_t, keys_t, values_t)
        # 다시 (B, bev_size, bev_size, embed_dim)로 reshape
        bev_features = bev_attended.transpose(0, 1).view(B, self.bev_size, self.bev_size, self.embed_dim)
        return bev_features

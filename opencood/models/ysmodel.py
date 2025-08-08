from datetime import datetime
import sys
import os

from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from TSA_Utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
# from efficientnet_encoder import EfficientNetFeatureExtractor, BEVGenerator
import torchvision
import glob
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models 
import torchvision 
from efficientnet_pytorch import EfficientNet
import json
from PIL import Image
import yaml
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import math
import torchvision.transforms.functional as Func
import contextlib
import opencood
from typing import Iterable, Optional
from torchvision.models.resnet import Bottleneck
from torch.cuda.amp import autocast
from ys_visualize import Visualization
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

#           In ysmodel
## Input Images shape : (agent_nums, Batch, camera_nums, image_H, image_W, channel) || [3,1,4,512,512,3]
## Input Intrinsic shape : (agent_nums, Batch, camera_nums, 3, 3)
## Input Extrinsic shape : (agent_nums, Batch, camera_nums, 4, 4)
## Input Position shape : (agent_nums, Batch, 6) || 6 -> (x,y,z ...)
## visuualize_bev | bev_feat shape : (channel, small_H, small_W) || [64 or 2,200,200]
## get_large_bev | envoded_bev shape : (agent_nums, Batch, Channel, small_H, small_W) || [3,1,64 or 3, 200, 200]
## Canvas shape : (Batch, channel, large_H, large_W) || [1,64,400,400]

BottleneckBlock = lambda x: Bottleneck(x, x//4)
###########################################################################
###################### ENCODER ############################################
###########################################################################
class SafeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, img):
        if isinstance(img, torch.as_tensor):
            # Tensor이면 바로 Normalize
            return self.normalize(img)
        else:
            # PIL.Image이면 ToTensor 후 Normalize
            img = Func.to_tensor(img)
            return self.normalize(img)
        
class SafeDenormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        self.denormalize = transforms.Normalize(mean=mean_inv.tolist(), std=std_inv.tolist())

    def forward(self, img):
        if isinstance(img, torch.as_tensor):
            return self.denormalize(img)
        else:
            raise ValueError("Input must be Tensor for denormalization.")       
    
denormalize_img = SafeDenormalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


normalize_img = SafeNormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        # 3차원: trilinear / 2차원: bilinear
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
    
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# @torch.no_grad()
# def get_pixel_coords_3d(coords_d, depth, cam_param, img_h=256, img_w=256, depth_num=64, depth_start=1, depth_max=61):
#     eps = 1e-5
#     B, N = cam_param.shape[:2]
#     H, W = depth.shape[-2:]
#     scale = img_h // H
#     # coords_h = torch.linspace(scale // 2, img_h - scale//2, H, device=depth.device).float()
#     # coords_w = torch.linspace(scale // 2, img_w - scale//2, W, device=depth.device).float()
#     coords_h = torch.linspace(0, 1, H, device=cam_param.device,dtype=torch.float32) * img_h
#     coords_w = torch.linspace(0, 1, W, device=cam_param.device,dtype=torch.float32) * img_w
#     # coords_d = get_bin_centers(depth_max, depth_start, depth_num).to(depth.device)
#     # coords_d = coords_d * bin_scale + bin_bias

#     D = coords_d.shape[0]
#     coords_d = coords_d.to(cam_param.device,dtype=torch.float32)
#     coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
#     coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
#     coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
#     coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
#     imgs = cam_param.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
#     coords3d = torch.matmul(imgs, coords).squeeze(-1)[..., :3] # B N W H D 3

#     return coords3d, coords_d

@torch.no_grad()
def get_pixel_coords_3d(coords_d, depth, cam_param,
                        img_h=256, img_w=256, depth_num=64, depth_start=1, depth_max=61):
    eps = 1e-5
    B, N = cam_param.shape[:2]
    H, W = depth.shape[-2:]
    scale = img_h // H

    # Define dtype and device from cam_param
    dtype = cam_param.dtype
    device = cam_param.device

    coords_h = torch.linspace(0, 1, H, device=device, dtype=dtype) * img_h
    coords_w = torch.linspace(0, 1, W, device=device, dtype=dtype) * img_w
    coords_d = coords_d.to(device=device, dtype=dtype)

    D = coords_d.shape[0]
    coords_d = coords_d.to(cam_param.device)
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
    coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
    imgs = cam_param.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
    coords3d = torch.matmul(imgs, coords).squeeze(-1)[..., :3] # B N W H D 3

    return coords3d, coords_d


class CamEncode(nn.Module):
    def __init__(self, C, depth_num=64, depth_start=1, depth_end=61, error_tolerance=1.0, img_h=256, img_w=256, in_channels=[256, 512, 1024, 2048],interm_c= 128,out_c: Optional[int] = 3):
        
        super(CamEncode, self).__init__()
        self.C = C 
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.error_tolerance = error_tolerance
        self.img_h = img_h
        self.img_w = img_w
        self.bins = self.init_bin_centers()
        in_c = 3

        with suppress_stdout():
            self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        
        self.up1 = Up(320+112, 512) # Adjusted sizes for ResNet-50
        self.depthnet = nn.Conv2d(512, self.depth_num + self.C, kernel_size=1, padding=0) ## ??

        self.feats = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, out_c, kernel_size=1, padding=0),
        )

        self.depth = nn.Sequential(
            nn.Conv2d(in_c, interm_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(interm_c),
            nn.ReLU(inplace=True),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            BottleneckBlock(interm_c),
            nn.Conv2d(interm_c, depth_num, kernel_size=1, padding=0)
        )

    """
    Mono camera로부터 depth를 예측하는 것은 모호하다. depth 정보가 불충분하기 때문이다. 
    그래서 여기서는 각 픽셀에 대해 이산적인 깊이 집합에 대한 확률 분포를 예측함으로써 깊이의 불확실성을 명시적으로 모델링한다. 
    """
    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1) # 여기서 depth 분포를 바꾸면? 
    
    def init_bin_centers(self):
        depth_range = self.depth_end - self.depth_start
        interval = depth_range / self.depth_num
        interval = interval * torch.ones((self.depth_num+1))
        interval[0] = self.depth_start
        bin_edges = torch.cumsum(interval, dim=0)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) 

        return bin_centers
    
    def pred_depth(self, cam_param, depth, coords_3d=None):
        # b, n, c, h, w = img.shape
        # b, n, c, h, w = depth.shape
        if coords_3d is None:
            coords_3d, coords_d = get_pixel_coords_3d(self.bins, depth, cam_param, depth_num=self.depth_num,depth_start=self.depth_start, depth_max=self.depth_end, img_h=self.img_h, img_w=self.img_w)
            coords_3d = rearrange(coords_3d, 'b n w h d c -> (b n) d h w c')
        
        depth_prob = depth.softmax(1) # (b n) depth h w
        pred_coords_3d = (depth_prob.unsqueeze(-1) * coords_3d).sum(1) # (b n) h w 3

        delta_3d = pred_coords_3d.unsqueeze(1) - coords_3d
        cov = (depth_prob.unsqueeze(-1).unsqueeze(-1) * (delta_3d.unsqueeze(-1) @ delta_3d.unsqueeze(-2))).sum(1)
        scale = (self.error_tolerance ** 2) / 9 
        cov = cov * scale
        return pred_coords_3d, cov

    def get_depth_feat(self, x):
        features = self.feats(x)
        depth = self.depth(x)
        return features, depth 
    
    # Extracts feature map from different stage of the resnet-40 model 
    # and then upsamples and concatnates them to produce a refined feature map.
    def get_eff_depth(self, x):
        endpoints = dict()

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        ############ 수정!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ############ 
        for i, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4']) #Concat last 2 stages 
        return x
    
    def forward (self, x, extrinsic):
        b, n, c, _, _ = x.shape
        image = x.flatten(0, 1).contiguous() # (B*N, C, H, W)

        #features = self.get_eff_depth(image) # (B*N, 512, H/16, W/16)
        #print("FEATURES SHAPE:", features.shape)
        features, depth = self.get_depth_feat(image) # (B*N, D+C, H/16, W/16)
        means3D, cov3D = self.pred_depth(extrinsic, depth)
        # print("FEATURES:", features.shape, "MEANS3D SHAPE:", means3D.shape, "| COV3D SHAPE:", cov3D.shape)
        
        # img_features = features.permute(0, 2, 3, 1).cpu().detach().numpy()
        # means3D = means3D.cpu().detach().numpy()
        # cov3D = cov3D.cpu().detach().numpy()
        # img = img_features.astype(np.float32)
        # step = 10
        # if img.max() > 1.0:
        #     img = img / 255.0

        # for cam in range(4):
        #     H, W, _ = img[cam].shape
        #     fig, ax = plt.subplots(figsize=(8,8), facecolor='white')
        #     ax.set_facecolor('white')
        #     ax.imshow(np.clip(img[cam], 0.0, 1.0))

        #     for i in range(0, H, step):
        #         for j in range(0, W, step):
        #             mu = means3D[cam][i, j, :2]
        #             cov_2d = cov3D[cam][i, j, :2, :2]
        #             try:
        #                 eigvals, eigvecs = np.linalg.eigh(cov_2d)
        #                 if np.any(eigvals <= 0):
        #                     continue
        #                 order = eigvals.argsort()[::-1]
        #                 eigvals = eigvals[order]
        #                 eigvecs = eigvecs[:, order]
        #                 angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        #                 width, height = 2 * np.sqrt(eigvals)

        #                 ellipse = patches.Ellipse(
        #                             (j, i), width, height,
        #                             angle=angle,
        #                             edgecolor='cyan',       # Bright on black or white
        #                             facecolor='yellow',
        #                             linewidth=1.5,
        #                             alpha=1.0
        #                         )

        #                 ax.add_patch(ellipse)

        #             except np.linalg.LinAlgError:
        #                 continue

        #     ax.set_title("Uncertainty Visualization for Camera {}".format(cam)) 
        #     plt.tight_layout()
            #plt.savefig("/home/rina/SG/savefigs/cam_{}_uncertainty_{}.png".format(cam, datetime.now().strftime("%Y%m%d_%H%M%S")))


        #cov3D = cov3D.flatten(-2, -1)
        #cov3D = torch.cat((cov3D[..., 0:3], cov3D[..., 4:6], cov3D[..., 8:9]), dim=-1)

        features = rearrange(features, '(b n) d h w -> b n d h w', b=b, n=n)
        #means3D = rearrange(means3D, '(b n) h w d-> b n d h w', b=b, n=n)
        #cov3D = rearrange(cov3D, '(b n) h w d -> b n d h w',b=b, n=n)

        return features, means3D, cov3D
    
class BEVEncode(nn.Module):
    def __init__(self, inC=3, outC=64):
        super(BEVEncode, self).__init__()
        
        trunk = models.resnet18(pretrained=True)
        trunk.zero_init_residual = True  # 따로 설정 필요
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1 # (batch, 64, 56, 56) > (batch, 256, 56, 56)
        self.layer2 = trunk.layer2 # (batch, 256, 56, 56) > (batch, 512, 28, 28)
        self.layer3 = trunk.layer3 # (batch, 512, 28, 28) > (batch, 1024, 14, 14)

        # 아마 사이즈 수정 필요할 듯!! 
        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)  # Ensure input is float32
        # print("input dtype:", x.dtype)
        # print("model weight dtype:", self.conv1.weight.dtype)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.up1(x3, x1)
        x = self.up2(x) 

        return x

class LSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf):
        super(LSS, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = data_aug_conf['bev_dim']

        xbound = grid_conf['xbound']
        ybound = grid_conf['ybound']
        zbound = grid_conf['zbound']
        dx = torch.as_tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.as_tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)


        self.downsample = 8
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.camC) # (41, 64, 16) > (41, 64, 16, 128, 352, 64)
        self.bevencode = BEVEncode()
        self.use_quickcumsum = True
        self.final_voxel_feature = nn.Sequential(
            nn.Conv3d(self.outC * 2, self.outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.outC),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.outC, self.outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.outC),
            nn.ReLU(inplace=True)
        )

    def create_frustum(self):
        ogfH, ogfW = self.data_aug_conf['final_dim'] # (128, 352)
        fH, fW = ogfH // self.downsample, ogfW // self.downsample # (8, 22)
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # (41, 8, 22)
        D, _, _ = ds.shape # 4
        
        xs = torch.linspace(0, ogfW -1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) # (41, 8, 22)
        ys = torch.linspace(0, ogfH -1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) # (41, 8, 22)

        frustum = torch.stack((xs, ys, ds), -1) # (41, 8, 22, 3)
        return nn.Parameter(frustum, requires_grad=False)

    
    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        # form transformation matrix from camera to ego
        det = torch.det(intrins)
        if torch.any(det==0):
            combine = rots.matmul(torch.linalg.pinv(intrins))
        else:
            combine = rots.matmul(torch.inverse(intrins))
            
        # points = points.to(dtype=combine.dtype)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # get the final (x,y,z) locations in the ego frame
        points += trans.view(B, N, 1, 1, 1, 3)

        return points
    
    def get_cam_feats(self, x, extrinsic):
        B, N, C, imH, imW = x.shape
        #N = 1
        #x = x.view(B*N, C, imH, imW)
        x, mean, var = self.camencode(x, extrinsic)    # X: [4, 3, 256, 256] Mean: [4, 256, 256, 3] Var: [4, 256, 256, 3, 3]
        x = x.view(B, N, C, self.D, imH//8, imW//8)
        x = x.permute(0, 1, 3, 4, 5, 2)

        #mean = mean.view(B, N, C, self.D, imH//8, imW//8)
        #mean = mean.permute(0, 1, 3, 4, 5, 2)

        #var = var.view(B, N, C*2, self.D, imH//8, imW//8)
        #var = var.permute(0, 1, 3, 4, 5, 2)

        return x, mean, var
    
    def cumsum_trick(x, geom__feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        return x, geom__feats
    
    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W #전체 voxel 개수 

        x = x.reshape(Nprime, C) # flatten x
        # Normalize the geometry features to the voxel grid
        # add batch index to each voxel
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime, 1], ix, 
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        # define voxel grid 
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # This part uses the cumulative sum trick to aggregate features from the same voxel grid cell 
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2])\
            + geom_feats[:, 1] * (self.nx[2])\
            + geom_feats[:, 2] \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = self.cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        # This part creates a zero tensor of the final voxel grid and 
        # assigns the aggregated features to the corresponding voxel grid cell
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        # Z 축을 제거하여 최종 voxel grid를 얻는다. 
        #final = torch.cat(final.unbind(dim=2), 1)

        return final
    
    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans, extrinsic):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, mean, var = self.get_cam_feats(x, extrinsic)
        x = self.voxel_pooling(geom, x)
        #mean = self.voxel_pooling(geom, mean)
        #var = self.voxel_pooling(geom, var)

        # print("X SHAPE:", x.shape, "| Mean ", mean.shape, "| VAR ",var.shape)

        return x, mean, var
    
    def forward(self, x, rots, trans, intrins, post_rots, post_trans, extrinsic):
        """
        x:          (N_agents ,B,  Ncam,    C, H, W)
        rots:       (B,  Ncam, 3, 3)
        trans:      (B,  Ncam,   3)
        intrins:    (B,  Ncam, 3, 3)
        post_rots:  (B,  Ncam, 3, 3)
        post_trans: (B,  Ncam,   3)
        """
        N_agent, B, Ncam, C, H, W = x.shape
        # 1) flatten cameras into batch dimension

        all_agent_bev = []
        vis_agent_bev = []
        for i in range(N_agent):
            voxel_feat_x, voxel_feat_mean, voxel_feat_var = self.get_voxels(x[i], rots[i], trans[i], intrins[i], post_rots[i], post_trans[i], extrinsic[i])
            # print("VOXEL FEAT X SHAPE:", voxel_feat_x.shape, "| MEAN SHAPE:", voxel_feat_mean.shape, "| VAR SHAPE:", voxel_feat_var.shape)

            voxel_feat_x = rearrange(voxel_feat_x, 'b c z x y -> b c x y z') # (B, outC, X, Y, Z)
            voxel_feat_x = voxel_feat_x.squeeze(4) # (B, outC, X, Y)
            up_bev_output = self.bevencode(voxel_feat_x)                     # (B, outC, X, Y)
            
            all_agent_bev.append(up_bev_output)
            vis_agent_bev.append(voxel_feat_x)

        all_bev_features = torch.stack(all_agent_bev).to(x.device)
        vis_bev_features = torch.stack(vis_agent_bev).to(x.device)
        # print("ALL BEV FEATURES SHAPE:", all_bev_features.shape)

        return all_bev_features, vis_bev_features

class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None
    
def get_rot(h):
    return torch.as_tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
    img = transforms.Resize(resize_dims)(img)
    img = Func.crop(img, crop[1], crop[0], crop[3] - crop[1], crop[2] - crop[0])
    if flip:
        img = Func.hflip(img)
    img = Func.rotate(img, angle=rotate)

    device = post_rot.device
    dtype = post_rot.dtype

    # Cast scalar to tensor of correct dtype
    resize = torch.tensor(resize, dtype=dtype, device=device)
    rotate = torch.tensor(rotate, dtype=dtype, device=device)

    post_rot = post_rot * resize
    post_tran = post_tran - torch.as_tensor(crop[:2], dtype=dtype, device=device)

    if flip:
        A = torch.tensor([[-1, 0], [0, 1]], dtype=dtype, device=device)
        b = torch.tensor([crop[2] - crop[0], 0], dtype=dtype, device=device)
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

    A = get_rot(rotate / 180 * np.pi).to(dtype=dtype, device=device)
    b = torch.tensor([(crop[2] - crop[0]) / 2, (crop[3] - crop[1]) / 2], dtype=dtype, device=device)
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

def sample_augmentation(data_aug_conf, is_train=True):
    H, W = data_aug_conf['H'], data_aug_conf['W']
    fH, fW = data_aug_conf['final_dim'][0], data_aug_conf['final_dim'][1]
    if is_train:
        resize = float(np.random.uniform(*data_aug_conf['resize_lim']))
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - float(np.random.uniform(*data_aug_conf['bot_pct_lim']))) * newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = bool(data_aug_conf['rand_flip'] and np.random.choice([0, 1]))
        rotate = float(np.random.uniform(*data_aug_conf['rot_lim']))
    else:
        resize = float(max(fH/H, fW/W))
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - float(np.mean(data_aug_conf['bot_pct_lim']))) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0.0
    return resize, resize_dims, crop, flip, rotate


def Encoding(img_inputs, intrinsics, extrinsics, data_aug_conf, grid_conf, is_train=True):
    L, B, M, H, W, C = img_inputs.shape
    device = data_aug_conf['device']
    all_imgs = []
    all_rots = []
    all_trans = []
    all_intrins = []
    all_post_rots = []
    all_post_trans = []

    for i in range(L):
        # print("AGENT", i)
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        ### Batch 
        for k in range(B):
            batch_imgs = []
            batch_rots = []
            batch_trans = []
            batch_intrins = []
            batch_post_rots = []
            batch_post_trans = []
            #print("Batch Input", img_inputs[i][k].shape, "Intrinsics", intrinsics[i][k].shape, "Extrinsics", extrinsics[i][k].shape)

            for j in range(int(data_aug_conf['Ncams'])):
                # print("CAMERA")
                img = img_inputs[i][k][j].float()       ## [H,W,C]
                img = img.permute(2, 0, 1)      ## [C, H, W]
                post_rot = torch.eye(2,dtype=torch.float)
                post_tran = torch.zeros(2,dtype = torch.float)
                extrin, intrin = extrinsics[i][k][j].float(), intrinsics[i][k][j].float()
                # 안전하게 as_tensor를 사용해 장치 정보 유지
                intrin = torch.as_tensor(intrin, dtype=torch.float)
                rot = torch.as_tensor(extrin[:3, :3], dtype=torch.float)
                tran = torch.as_tensor(extrin[:3, 3], dtype=torch.float)

                resize, resize_dims, crop, flip, rotate = sample_augmentation(data_aug_conf, is_train)
                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate)   ## shape || img : [3, 256, 256] post_rot2 : [2,2] post_trans2 : [2]
                post_tran = torch.zeros(3,dtype=torch.float)
                post_rot = torch.eye(3,dtype=torch.float)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                batch_imgs.append(img)
                batch_intrins.append(intrin)
                batch_rots.append(rot)
                batch_trans.append(tran)
                batch_post_rots.append(post_rot)
                batch_post_trans.append(post_tran)
            
            imgs.append(torch.stack(batch_imgs))    ## torch.stack(batch_imgs) : [4, 3, 256, 256]  || [num_cams, C, h, w]
            rots.append(torch.stack(batch_rots))
            trans.append(torch.stack(batch_trans))
            intrins.append(torch.stack(batch_intrins))
            post_rots.append(torch.stack(batch_post_rots))
            post_trans.append(torch.stack(batch_post_trans))

        all_imgs.append(torch.stack(imgs))              ## torch.stack(imgs) : [1, 4, 3, 256,256]  || [Batch, num_cams, C, h, w]
        all_rots.append(torch.stack(rots))              ## torch.Size([1, 4, 3, 3])
        all_trans.append(torch.stack(trans))            ## torch.Size([1, 4, 3])
        all_intrins.append(torch.stack(intrins))        ## torch.Size([1, 4, 3, 3])
        all_post_rots.append(torch.stack(post_rots))    ## torch.Size([1, 4, 3, 3])
        all_post_trans.append(torch.stack(post_trans))  ## torch.Size([1, 4, 3])
    
    encoding_model = LSS(grid_conf, data_aug_conf).to(device)
    
    encoded_bev, vis_encoded_bev = encoding_model(
        torch.stack(all_imgs).to(device), 
        torch.stack(all_rots).to(device),
        torch.stack(all_trans).to(device),
        torch.stack(all_intrins).to(device),
        torch.stack(all_post_rots).to(device),
        torch.stack(all_post_trans).to(device),
        extrinsics.to(device)
    )

    return encoded_bev, vis_encoded_bev

# 간단한 BEV 시각화 함수
def visualize_bev(bev_feat, title='BEV', save=False):
    """
    bev_feat : (C,H,W)
    """
    bev_img = bev_feat.mean(dim=0).detach().cpu().numpy()
    bev_img = (bev_img - bev_img.min()) / (bev_img.max() - bev_img.min() + 1e-5)
    plt.figure(figsize=(6,6))
    plt.imshow(bev_img, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis('off')
    if save == True:
        save_path = f'/home/sglee6/ysmodel/opencood/models/concat_savefigs/{title}.png'
        plt.savefig(save_path)
    else:
        plt.show()

def get_relative_pose(ego_pose, agent_pose):
    dx = agent_pose[..., 0] - ego_pose[..., 0]
    dy = agent_pose[..., 1] - ego_pose[..., 1]
    dyaw = agent_pose[..., 4] - ego_pose[..., 4]
    return dx, dy, dyaw

def warp_and_paste_into_large_bev(canvas, bev_feat, dx, dy, dyaw, voxel_size=0.5):
    C, H, W = bev_feat.shape
    H_big, W_big = canvas.shape[2:]
    device = bev_feat.device
    dtype = bev_feat.dtype  # 모든 텐서의 dtype을 이걸로 맞춰줍니다.

    # 소스 pixel 좌표 (중심 정렬 → meter로 변환)
    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=0)  # (2, H, W)
    coords -= torch.tensor([W / 2, H / 2], device=device, dtype=dtype).view(2, 1, 1)
    coords *= voxel_size  # pixel → meter

    # 회전
    cos_theta = torch.cos(dyaw.clone().detach().to(device=device, dtype=dtype))
    sin_theta = torch.sin(dyaw.clone().detach().to(device=device, dtype=dtype))
    rot_mat = torch.stack([
        torch.stack([cos_theta, -sin_theta]),
        torch.stack([sin_theta, cos_theta])
    ], dim=0)

    coords_rot = rot_mat @ coords.view(2, -1)  # (2, H*W)

    # 이동
    coords_rot[0] += dx.clone().detach().to(device=device, dtype=dtype)
    coords_rot[1] += dy.clone().detach().to(device=device, dtype=dtype)

    # canvas 좌표로 변환 (meter → pixel)
    coords_canvas = coords_rot / voxel_size
    coords_canvas[0] += W_big / 2
    coords_canvas[1] += H_big / 2

    x_canvas = coords_canvas[0].round().long()
    y_canvas = coords_canvas[1].round().long()

    # 유효 좌표만 필터링
    valid = (x_canvas >= 0) & (x_canvas < W_big) & (y_canvas >= 0) & (y_canvas < H_big)
    x_canvas = x_canvas[valid]
    y_canvas = y_canvas[valid]
    src_feat = bev_feat.view(C, -1)[:, valid].to(canvas.dtype)

    # 누적 방식으로 canvas에 삽입
    for c in range(C):
        canvas[0, c].index_put_((y_canvas, x_canvas), src_feat[c], accumulate=True)

    return canvas


# 전체 large BEV 생성
def get_large_bev(data_aug_conf, encoded_bev,vis_encoded_bev, positions, H_big=400, W_big=400, save_vis = False):
    """
    encoded_bev shape : (agent_num, Batch, channel, small_H, small_W)
    position shape : (agent_num, Batch, 6) || 6 : [x,y,z,...]
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    device = data_aug_conf['device']
    positions = positions.permute(1, 0, 2)

    N, B, C, H, W = encoded_bev.shape
    canvas = torch.zeros((B, C, H_big, W_big), device=device)
    N, B, C, H, W = vis_encoded_bev.shape
    vis_canvas = torch.zeros((B, C, H_big, W_big), device=device)
    ego_pose = positions[0][0]
    
    for i in range(N):
        each_agent_bev_feats = encoded_bev[i].squeeze(0)        ## each_agent_bev_feats shape :  (C, small_H, small_W)
        dx, dy, dyaw = get_relative_pose(ego_pose, positions[0][i])
        
        ## visualize the bev_feat
        if save_vis:
            each_vis_agent_bev_feats = vis_encoded_bev[i].squeeze(0)
            visualize_bev(each_vis_agent_bev_feats,title=f'Before_{i}_{timestamp}',save=True)
            vis_canvas = warp_and_paste_into_large_bev(vis_canvas, each_vis_agent_bev_feats, dx, dy, dyaw)
            
        canvas = warp_and_paste_into_large_bev(canvas, each_agent_bev_feats, dx, dy, dyaw)

    if save_vis:
        visualize_bev(vis_canvas[0],title=f'Concated_{timestamp}',save=True)
        
    return canvas      ## canvas shape : (Batch, channel, large_H, large_W)

###########################################################################
###################### DECODER ############################################
###########################################################################
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale  # ✔️ now tracked

class PatchEmbed(nn.Module):
    def __init__(self, img_size=400, patch_size=16, in_channels=64, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)   # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


################################################################################################################################
#                                       Decoder Version 1 - Simple UpBlock1
################################################################################################################################

# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, norm_groups=4):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
#         self.norm = nn.GroupNorm(norm_groups, out_ch)
#         self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def forward(self, x, out_size):
#         x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
#         x = self.conv(x)
#         x = self.norm(x)
#         return self.act(x)

# class PatchDecoder(nn.Module):
#     def __init__(self, emb_dim, out_dim=64):
#         super().__init__()
#         # 입력 채널 줄이기: 64 -> 32
#         self.initial_conv = nn.Sequential(
#             nn.Conv2d(emb_dim, 32, kernel_size=3, padding=1),
#             nn.GroupNorm(4, 32),
#             nn.LeakyReLU(0.1, inplace=True)
#         )

#         self.up1 = UpBlock(32, 24)     # 64x64
#         self.up2 = UpBlock(24, 16)     # 128x128
#         self.up3 = UpBlock(16, 12)     # 256x256
#         self.up4 = UpBlock(12, out_dim)  # 512x512

#     def forward(self, x):
#         B, N, C = x.shape
#         H = W = int(N ** 0.5)
#         assert H * W == N, "Input must be square number of patches"

#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         x = self.initial_conv(x)
#         x = self.up1(x, (64, 64))
#         x = self.up2(x, (128, 128))
#         x = self.up3(x, (256, 256))
#         x = self.up4(x, (512, 512))
#         return x

################################################################################################################################
#                                       Decoder Version 2 - U-net
# ################################################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels, eps=1e-5),   # 🔑 std=0 방지 eps
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels, eps=1e-5),   # 🔑
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        return self.block(x)

class PatchDecoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.input_proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ConvBlock(emb_dim, 128)
        self.down1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.enc2 = ConvBlock(128, 256)
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.enc3 = ConvBlock(256, 512)

        # Decoder
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)

        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, emb_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "N must be a perfect square"

        x = x.transpose(1, 2).reshape(B, C, H, W)  # [B, 64, 25, 25]
        x = self.input_proj(x)                    # [B, 64, 25, 25]

        x1 = self.enc1(x)                         # [B, 128, 25, 25]
        x1_down = self.down1(x1)                  # [B, 128, 13, 13]

        x2 = self.enc2(x1_down)                   # [B, 256, 13, 13]
        x2_down = self.down2(x2)                  # [B, 256, 7, 7]

        x3 = self.enc3(x2_down)                   # [B, 512, 7, 7]

        d2 = self.up2(x3)                         # [B, 256, 14, 14]
        x2 = F.interpolate(x2, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))  # [B, 256, 14, 14]

        d1 = self.up1(d2)                         # [B, 128, 28, 28]
        x1 = F.interpolate(x1, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))  # [B, 128, 28, 28]

        d0 = self.up0(d1)                         # [B, 64, 56, 56]

        out = F.interpolate(d0, size=(512, 512), mode='bilinear', align_corners=False)
        out = self.final_conv(out)               # [B, 64, 512, 512]
        out = self.out_relu(out)
        
        return out


################################################################################################################################
################################################################################################################################
    
## Positional Encoding
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

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 1e-6, ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device

        mean = x.mean(dim=-1,keepdim=True)      ## x shape : [batch, sentence length, dimension]
        std = x.std(dim=-1,keepdim=True)        ## dim = -1 : 젤 마지막을 제거하겟다 -> 열을 제거하겠다 -> 행만 남긴다 -> 행방향 평균 및 분산  
        return self.alpha.to(device) * (x-mean) / (std+self.eps) + self.bias.to(device)
    
class FeedforwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff :int, dropout : float,device) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff,device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model,device=device)

    def forward(self, x):
        # print('Feed Forward Finished')
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# Residual Connection
class ResidualConnection(nn.Module):
    def __init__(self, dropout, normalized_shape):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape=normalized_shape)  # 실제 shape 지정 필요

    def forward(self, x, sublayer, return_with_aux=False):
        if return_with_aux:
            out, aux = sublayer(self.norm(x))
            return x + self.dropout(out), aux
        else:
            return x + self.dropout(sublayer(self.norm(x)))

class DeformableAttention1D(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_points=2,
        dropout=0.,
        device=None,
    ):
        super().__init__()
        self.heads = heads
        self.num_points = num_points
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        # LayerNorm
        self.norm_q = nn.LayerNorm(dim * 2)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)


        # Q, K, V projection
        self.to_q = nn.Linear(dim * 2, inner_dim, bias=False, device=device)                    ## in_features (*,H_in), out_features (*,H_out) || nn.Linear(H_in, H_out)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
      

        # Offset predictor: (B, H, N, P)
        self.offset_predictor = nn.Conv1d(inner_dim, heads * num_points, 1, device=device)      ## in_features (batch, feature_dim, time_step) || nn.Conv1d(feature_dim, out_feature_dim)
        self.offset_predictor.bias.data.uniform_(-1.0, 1.0)
        self.attn_weight_predictor = nn.Conv1d(inner_dim, heads * num_points, 1, device=device)

        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1, device=device),
            nn.Dropout(dropout)
        )
        
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

    def forward(self, x, prev_x, use_learned_weights=False):
        """
        x, prev_x: (B, N, D)
        """
        B, N, D = x.shape
        H, P = self.heads, self.num_points
        device = x.device

        # ===== Query =====
        q_input = torch.cat([x, prev_x], dim=-1)  # (B, N, 2D)
        q_input = self.norm_q(q_input)            # (B, N, 2D)
        q = self.to_q(q_input).transpose(1, 2)  # (B, inner_dim, N)

        # ===== Offset + Attention weights =====
        # offset / attn weight 예측
        offsets = self.offset_predictor(q)                                                      # (B, H*P, N)
        offsets = rearrange(offsets, 'b (h p) n -> b h p n', h=self.heads, p=self.num_points)   # (B, H, P, N)

        attn_weights = self.attn_weight_predictor(q)                                                        # (B, H*P, N)
        attn_weights = rearrange(attn_weights, 'b (h p) n -> b h p n', h=self.heads, p=self.num_points)     # (B, H, P, N)
        attn_weights = attn_weights.softmax(dim=2)                                                          # (B, H, P, N) || softmax를 통해 attention weight를 구함

        # ===== KV =====
        kv_input = torch.cat([x, prev_x], dim=0)                # (2B, N, D)
        kv_input_k = self.norm_k(kv_input)                       # (2B, N, D)
        kv_input_v = self.norm_v(kv_input)                       # (2B, N, D)
        k = self.to_k(kv_input_k)
        v = self.to_v(kv_input_v)
        # k, v = kv.chunk(2, dim=-1)                              # (2B, N, inner_dim) 
        
        k = k.view(2, B, N, H, -1).permute(1, 3, 4, 0, 2).reshape(B, H, -1, 2 * N)  # (2B,N,inner_dim) -> (2,B,N,H,Dim_head) -> (B,H,Dim_head,2,N) -> (B,H,Dim_head,2N)
        v = v.view(2, B, N, H, -1).permute(1, 3, 4, 0, 2).reshape(B, H, -1, 2 * N)  # (2B,N,inner_dim) -> (2,B,N,H,Dim_head) -> (B,H,Dim_head,2,N) -> (B,H,Dim_head,2N)

        # ===== Sampling position =====
        pos = torch.arange(N, device=device).view(1, 1, 1, N).expand(B, H, P, -1)  # (N,) -> (1,1,1,N) -> (B,H,P,N)
        sample_pos = pos + offsets  # (B, H, P, N)
        sample_pos = sample_pos.clamp(0, 2 * N - 1)

        # ===== Gather function =====
        def gather_feat(feat, pos):
            """
            feat: (B, H, D, L)
            pos:  (B, H, P, Nq)
            Returns:
                out: (B, H, D, Nq, P)
            """
            B, H, D, L = feat.shape
            _, _, P, Nq = pos.shape

            pos = pos.permute(0, 1, 3, 2).contiguous().long()  # (B, H, Nq, P)
            feat = feat.permute(0, 1, 3, 2).contiguous()       # (B, H, L, D)

            # flatten B and H to simplify gather
            feat = feat.view(B * H, L, D)                      # (B*H, L, D)
            pos = pos.view(B * H, Nq, P)                       # (B*H, Nq, P)

            index = pos.unsqueeze(-1).expand(-1, -1, -1, D)     # (B*H, Nq, P, D)
            feat = feat.unsqueeze(1).expand(-1, Nq, -1, -1)     # (B*H, Nq, L, D)

            gathered = torch.gather(feat, 2, index)            # (B*H, Nq, P, D)
            gathered = gathered.permute(0, 3, 1, 2).contiguous().view(B, H, D, Nq, P)

            return gathered  # (B, H, D, Nq, P)

        k_sampled = gather_feat(k, sample_pos)  # (B, H, D, N, P) || pos에 따라 key/value를 샘플링
        v_sampled = gather_feat(v, sample_pos)  # (B, H, D, N, P)

        # ===== Attention =====
        q = q.view(B, H, -1, N).transpose(2, 3) * self.scale  # (B, H, N, Dh)

        # Deformable DETR에서는 attn_weight만 softmax로 쓰기도 함
        if use_learned_weights:
            attn = attn_weights.permute(0, 1, 3, 2).unsqueeze(-1)  # (B, H, N, P, 1)
        else:
            sim = (q.unsqueeze(-1) * k_sampled.permute(0, 1, 3, 2, 4)).sum(-2)
            attn = sim.softmax(dim=-1).unsqueeze(-1)  # (B, H, N, P, 1)
        # v = v_sampled.permute(0, 1, 3, 4, 2)                  # (B, H, N, P, D)
        # out = (attn * v).sum(dim=3)
        
        v = v_sampled.permute(0, 1, 3, 4, 2)  # (B, H, N, P, D)
        out = (attn * v).sum(dim=3)                           # (B, H, N, D)

        out = out.permute(0, 1, 3, 2).reshape(B, -1, N)       # (B, inner_dim, N)
        out = self.to_out(out).transpose(1, 2)                # (B, N, D)
        return out, offsets
    
class TSABlock(nn.Module):
    def __init__(self, deformable_block : DeformableAttention1D, feedforward_block : FeedforwardBlock, dropout : float, normalized_shape ):
        super().__init__()
        self.deformabel_block = deformable_block
        self.feedforward_block = feedforward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout,normalized_shape) for _ in range(2)])

    def forward(self, bev_query, prev_bev):
        def deformable_fn(bev_query):
            out, offsets = self.deformabel_block(bev_query, prev_bev)
            return out, offsets

        out_and_offsets = self.residual_connection[0](bev_query, deformable_fn, return_with_aux=True)
        bev_query, offsets = out_and_offsets

        bev_query = self.residual_connection[1](bev_query, self.feedforward_block)
        return bev_query, offsets
    
class TSA_Loop(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, prev_x):
        offsets_list = []
        for layer in self.layers:
            x, offsets = layer(x, prev_x)
            offsets_list.append(offsets)
        return x, offsets_list
    
class BevSegHead(nn.Module):
    def __init__(self, target, input_dim, dynamic_output_class=None, static_output_class=None):
        super(BevSegHead, self).__init__()
        self.target = target
        if self.target == 'dynamic':
            assert dynamic_output_class is not None
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_output_class,
                                          kernel_size=3,
                                          padding=1)

        elif self.target == 'static':
            assert static_output_class is not None
            self.static_head = nn.Conv2d(input_dim,
                                         static_output_class,
                                         kernel_size=3,
                                         padding=1)

        else:  # both
            assert dynamic_output_class is not None and static_output_class is not None
            self.dynamic_head = nn.Sequential(nn.Conv2d(input_dim,
                                          dynamic_output_class,
                                          kernel_size=3,
                                          padding=1),
                                        #   nn.GroupNorm(1,dynamic_output_class,eps=1e-3),
                                          nn.ReLU(inplace=True))  # 또는 nn.SiLU() 등)     ## Conv2d 출력의 mean,var을 일정하게 만들어 안정적 스케일을 맞춰줌, GroupNorm(1[Conv2d output channel을 1개의 그룹으로 본다], 2)
                                          
            self.static_head = nn.Sequential(
                nn.Conv2d(input_dim,
                        static_output_class,
                        kernel_size=3,
                        padding=1),
                        # nn.GroupNorm(1,static_output_class,eps=1e-3),
                        nn.ReLU(inplace=True)  # 또는 nn.SiLU() 등
            )
            self.dynamic_head = self.dynamic_head.float()
            self.static_head = self.static_head.float()

    def forward(self,  x, b, l):
        if self.target == 'dynamic':
            dynamic_map = self.dynamic_head(x)
            dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                    b=b, l=l)
            static_map = torch.zeros_like(dynamic_map,
                                          device=dynamic_map.device)

        elif self.target == 'static':
            static_map = self.static_head(x)
            static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                                   b=b, l=l)
            dynamic_map = torch.zeros_like(static_map,
                                           device=static_map.device)

        else:
            with torch.cuda.amp.autocast(enabled=False):        ## float32로 강제
                x = x.float()
                x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
                dynamic_map = self.dynamic_head(x)
                dynamic_map = torch.nan_to_num(dynamic_map, nan=0.0, posinf=1e3, neginf=-1e3)
                assert torch.isfinite(dynamic_map).all(), f"NaN in dynamic_map!"
                dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                        b=b, l=l)
    
                static_map = self.static_head(x)
                static_map = torch.nan_to_num(static_map, nan=0.0, posinf=1e3, neginf=-1e3)
                assert torch.isfinite(static_map).all(), f"NaN in static_map!"
                static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                                    b=b, l=l)

        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        return output_dict

class Mini_cooper(nn.Module):
    def __init__(self,
                 data_aug_conf : dict,
                 grid_conf : dict,
                 pseudo_patching : PatchEmbed,
                 tsa_loop : TSA_Loop,
                 positional_encoding : PositionalEncodeing,
                 pseudo_decoding : PatchDecoder,
                 seg_head = BevSegHead,):
        super().__init__()

        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.tsa_loop = tsa_loop
        self.pseudo_patching = pseudo_patching
        self.positional_encoding = positional_encoding
        self.pseudo_decoding = pseudo_decoding
        self.seg_header = seg_head

    # def add_pe_in_combined_bev(self,combined_bev):
    #     return self.positional_encoding(combined_bev)
    ############################################################################################
    ############ ENCODER 최종 불러오는 곳
    ############################################################################################
    def encoding(self, images, intrins, extrins, positions, is_train=True):
        encoded_bev, vis_encoded_bev = Encoding(images, intrins, extrins, self.data_aug_conf, self.grid_conf, is_train)
        # print("ENCODED BEV SHAPE OF MULTI_AGENTS", encoded_bev.shape)     ## (max_padded_cavs(5), 1, 64, 200, 200)
        ##          64 channel Case (For Training)
        mapped_bev= get_large_bev(self.data_aug_conf, encoded_bev,vis_encoded_bev, positions, save_vis = False)

        assert torch.isfinite(mapped_bev).all(), "[NaN] mapped_bev after encoding!"
        # print("LARGE BEV:::: ", mapped_bev, mapped_bev.shape)     ##  torch.Size([1, 64, 400, 400]) [B, Channel, H, W]
        return mapped_bev    ## true_poas (x,y,yaw) 3차원으로 나옴
    
    ############################################################################################
    def loop_output(self, bev_query, prev_bev, bool_prev_pos_encoded):
        """
        bev_query shape : torch.Size([1, 64, 400, 400])
        """
        # bev_query = bev_query.unsqueeze(0)
        bev_query = self.pseudo_patching(bev_query)
        # prev_bev = prev_bev.unsqueeze(0)
        prev_bev = self.pseudo_patching(prev_bev)

        bev_query = self.positional_encoding(bev_query) 
        if bool_prev_pos_encoded:
            pass
            # print("prev bev is already positional encoded")
        else:
            prev_bev = self.positional_encoding(prev_bev)
            # print("prev bev is positional encoded")

        ## bev_query shape : torch.Size([1, 2500, 64])
        result, offsets_list = self.tsa_loop(bev_query,prev_bev)
        assert torch.isfinite(result).all(), "[NaN] tsa_loop output!"
        return result, offsets_list
    
    def postprocessing_after_model(self, loop_result):
        postprocessed_output = self.pseudo_decoding(loop_result)
        return postprocessed_output
    
    def seg_head(self, x):
        return self.seg_header(x=x,b=1,l=1)

    def add_item_to_dict(self, dictionary, key, value):
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]
        return dictionary

    def forward(self, current_bev, prev_bev, bool_prev_pos_encoded):

        loop_output, offsets_list = self.loop_output(current_bev, prev_bev, bool_prev_pos_encoded)
        # print("loop_output mean/std:", loop_output.mean().item(), loop_output.std().item())
        assert torch.isfinite(loop_output).all(), "[NaN] loop_output after tsa_loop!"
        # print(f'Loop Output shape : {loop_output.shape}')
        model_output = self.postprocessing_after_model(loop_output)
        # print("pseudo_decoding output mean/std:", model_output.mean().item(), model_output.std().item())

        model_output = torch.nan_to_num(model_output, nan=0.0, posinf=1e3, neginf=-1e3)
        model_output = torch.clamp(model_output,-10,10) ## NaN 방지
        assert torch.isfinite(model_output).all(), "[NaN] before seg_head!"

        # print(f'After PostProcessing from loop output : {model_output.shape}')  ## shape : torch.Size([1, 64, 400, 400])
        seg_loss_dict = self.seg_head(model_output)
        # with torch.no_grad():
        #     print("dynamic_seg:", seg_loss_dict["dynamic_seg"].mean().item(), seg_loss_dict["dynamic_seg"].std().item())
        #     print("static_seg :", seg_loss_dict["static_seg"].mean().item(), seg_loss_dict["static_seg"].std().item())

        # print(f'Head Output shape : {head_output["dynamic_seg"].shape}')
        
        dummy_pos_loss = torch.tensor(0.0).to('cuda')
        final_dict = self.add_item_to_dict(seg_loss_dict,'offsets', offsets_list)
        final_dict = self.add_item_to_dict(final_dict,'pos_loss', dummy_pos_loss)
        
        return final_dict
        
def ysmodel(
        config : dict
    ):

    #######################################################
    ########## 여기서 파라미터 다 부르고 위에 내 encoder 붙이고 
    ########## position 학습 모델 붙이고 
    ########## large bev 에 embedding 하는 것까지 완성 
    #######################################################
    ####### ENCODER PARAMS
    #######################################################
    #B = config['encoder']['batch_size']
    H = config['encoder']['H']  
    W = config['encoder']['W']  
    resize_lim = config['encoder']['resize_lim']
    final_dim = config['encoder']['final_dim']
    bot_pct_lim = config['encoder']['bot_pct_lim']
    rot_lim = config['encoder']['rot_lim']
    rand_flip = config['encoder']['rand_flip']
    ncams = config['encoder']['ncams']
    max_grad_norm = config['encoder']['max_grad_norm']
    pos_weight = config['encoder']['pos_weight']
    final_bev = config['encoder']['final_bev']
    large_bev = config['encoder']['large_bev']
    cams = config['encoder']['cams']
    device = config['deconv']['device']
    bev_dim = config['encoder']['bev_dim']
    
    xbound = config['encoder']['xbound']
    ybound = config['encoder']['ybound']
    zbound = config['encoder']['zbound']
    dbound = config['encoder']['dbound']
    depth = config['encoder']['depth']

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
        'depth': depth
    }

    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'bot_pct_lim': bot_pct_lim,
        'rand_flip': rand_flip,
        'Ncams': ncams,
        'H': H,
        'W': W,
        'cams' : cams,
        'final_bev' : final_bev,
        'device' : device,
        'large_bev' : large_bev,
        'bev_dim' : bev_dim,
    }

    #######################################################
    ####### DECODER PARAMS
    #######################################################
    large_bev_size = config['deconv']['large_bev_size']  
    patch_size = config['deconv']['patch_size']  ## 16 x 16 x c patch로 나누기 위한 사이즈
    bev_emb_dim = config['encoder']['bev_dim']
    N = config['deconv']['N']
    h = config['deconv']['h']
    dropout = config['deconv']['dropout']
    d_feed_forward = config['deconv']['d_feed_forward']
    device = config['deconv']['device']
    seg_type = config['target']
    dynamic_output_class = config['dynamic_output_class']
    static_output_class = config['static_output_class']
    
    positional_embed = PositionalEncodeing(bev_emb_dim,(large_bev_size//patch_size)**2,dropout,device)
    pseudo_patching = PatchEmbed(img_size=large_bev_size, patch_size=patch_size, in_channels=bev_emb_dim, embed_dim=bev_emb_dim).to(device=device)
    # pseudo_decoding = PatchDecoder(embed_dim=bev_emb_dim, out_channels=bev_emb_dim, patch_size=patch_size, bev_size=large_bev_size).to(device=device)
    pseudo_decoding = PatchDecoder(emb_dim=bev_emb_dim).to(device=device)

    model = []
    for _ in range(N):
        multi_attention_block = DeformableAttention1D(dim = bev_emb_dim,heads=h,device=device)
        feed_forward_block = FeedforwardBlock(bev_emb_dim,d_feed_forward,dropout,device)
        model_block = TSABlock(multi_attention_block,feed_forward_block,dropout,bev_dim)
        model.append(model_block)
    
    tsa_loop = TSA_Loop(nn.ModuleList(model))

    seg_head = BevSegHead(seg_type, bev_emb_dim, dynamic_output_class, static_output_class).to(device=device)    

    mini_cooper = Mini_cooper(data_aug_conf, grid_conf,pseudo_patching,tsa_loop,positional_embed,pseudo_decoding,seg_head)

    return mini_cooper
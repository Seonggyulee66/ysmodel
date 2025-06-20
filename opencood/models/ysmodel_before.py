import sys
import os
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

class Scale(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale
    
class Encoding_input(nn.Module):
    def __init__(self,seq_len : int, output_dim : int, device):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.layer = nn.Linear(self.seq_len,self.output_dim,device=device)

    def forward(self,x):
        return self.layer(x)

class Decoding_output(nn.Module):
    def __init__(self,output_dim : int, seq_len : int,device):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len

        self.layer = nn.Linear(self.output_dim,self.seq_len,device=device)

    def forward(self,x):
        return self.layer(x)
    
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
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) ## 논문에서는 normalize하기 전에 sublayer를 썻지만 Effective를 위해 norm 한뒤 sublayer를 통과

class DeformableAttention1D(nn.Module):
    def __init__(
      self,
      dim,
      dim_head = 64,
      heads = 8,
      dropout = 0.,
      downsample_factor = 4,
      offset_scale = None,
      offset_groups = None,
      offset_kernel_size = 6,
      group_queries = True,
      group_key_value = True,
      device = None
    ):
        super().__init__()
        offset_scale = default(offset_scale,downsample_factor)  ## 앞에 param이 존재하면 앞에꺼 없으면 뒤에꺼
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)   ## offset group은 여러개의 offest을 묶은 집합, 각 head가 다른 위치를 참조할 수 있도록 하나의 head에 할당된 offset들의 모음
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups  

        offset_dims = inner_dim // offset_groups
        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv1d(offset_dims,offset_dims,offset_kernel_size,groups=offset_dims,stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2,device=device),
            nn.GELU(),
            nn.Conv1d(offset_dims,1,1,bias=False,device=device),
            Rearrange('b 1 n -> b n'),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv1d(2*dim, inner_dim, 1, groups=offset_groups if group_queries else 1, bias=False,device=device)
        self.to_k = nn.Conv1d(dim, inner_dim, 1, groups=offset_groups if group_key_value else 1, bias = False,device=device)
        self.to_v = nn.Conv1d(dim, inner_dim, 1, groups=offset_groups if group_key_value else 1, bias = False,device=device)

        self.to_out = nn.Conv1d(inner_dim,dim,1,device=device)

    def forward(self,x,prev_x):
        """
        b - batch
        h - heads
        n - sequence dimension
        d - dimension
        g - offset groups

        x (input) : (batch, sequence, dim)
        prev_x (input) : (batch, sequence, dim)

        Processing shape : (batch, dim, sequence)
        """
        x = rearrange(x, 'b n d -> b d n')
        prev_x = rearrange(prev_x, 'b n d -> b d n')

        heads, b, n, downsample_factor, device = self.heads, x.shape[0], x.shape[-1], self.downsample_factor, x.device

        concatenated_x = torch.cat([prev_x,x],dim=1)
        q = self.to_q(concatenated_x)       ## (b, inner_dim, sequence)
        
        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g=self.offset_groups)

        grouped_queries = group(q)  ## (b*heads, offset_dims , sequence)
        offsets = self.to_offsets(grouped_queries)

        grid = torch.arange(offsets.shape[-1], device=device)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = grid_sample_1d(
            group(prev_x),
            vgrid_scaled,
            mode='bilinear', padding_mode = 'zeros', align_corners = False
        )
        
        kv_feats = rearrange(kv_feats,'(b g) d n -> b (g d) n', b = b)      ## ( batch * heads, dim / heads, offset-modifired sequnce ) ==> ( batch, dim, offset-modifired sequnce )

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)     ## (batch, inner_dim, offset-modified sequnce)

        q = q * self.scale

        q, k, v = map(lambda t : rearrange(t,'b (h d) n -> b h n d', h = heads), (q, k, v))     
        # q : (batch, heads, sequnece, dim)
        # k : (batch, heads, offset-modified sequnce, dim)
        # v : (batch, heads, offset-modified sequnce, dim)

        similarity = einsum('b h i d, b h j d -> b h i j', q ,k)    ## q (batch, heads, sequnece, dim) @ k (batch, heads, offset-modified sequnce, dim)  ==> ( batch, heads, sequnce, offset-modified sequnce )

        atten = similarity.softmax(dim=-1)  
        atten = self.dropout(atten)

        out = einsum('b h i j, b h j d -> b h i d', atten, v)       ## ( batch, heads, sequnce, offset-modified sequnce ) @ v (batch, heads, offset-modified sequnce, dim) ==> (batch, heads, sequnce, dim)
        out = rearrange(out, 'b h n d -> b (h d) n')                ## (batch, heads, sequnce, dim) ==> (batch, dim, sequnce)
        out = self.to_out(out)
        out = rearrange(out,'b d n -> b n d')

        # print('Deformable Attetion finished')

        return out

class TSABlock(nn.Module):
    def __init__(self, deformable_block : DeformableAttention1D, feedforward_block : FeedforwardBlock, dropout : float):
        super().__init__()
        self.deformabel_block = deformable_block
        self.feedforward_block = feedforward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, bev_query, prev_bev):
        bev_query = self.residual_connection[0](bev_query,lambda bev_query: self.deformabel_block(bev_query,prev_bev))
        # print('Residual after Deformable finished')
        bev_query = self.residual_connection[1](bev_query,self.feedforward_block)
        # print('Residual after FFN finished')

        return bev_query
    
class TSA_Loop(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, bev_query, prev_bev):
        for layer in self.layers:
            bev_query = layer(bev_query,prev_bev)
        return self.norm(bev_query)
    
class BevSegHead(nn.Module):
    def __init__(self, target, input_dim, output_class):
        super(BevSegHead, self).__init__()
        self.target = target
        if self.target == 'dynamic':
            self.dynamic_head = nn.Conv2d(input_dim,
                                          output_class,
                                          kernel_size=3,
                                          padding=1)
        if self.target == 'static':
            # segmentation head
            self.static_head = nn.Conv2d(input_dim,
                                         output_class,
                                         kernel_size=3,
                                         padding=1)
        else:
            self.dynamic_head = nn.Conv2d(input_dim,
                                          output_class,
                                          kernel_size=3,
                                          padding=1)
            self.static_head = nn.Conv2d(input_dim,
                                         output_class,
                                         kernel_size=3,
                                         padding=1)

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
            dynamic_map = self.dynamic_head(x)
            dynamic_map = rearrange(dynamic_map, '(b l) c h w -> b l c h w',
                                    b=b, l=l)
            static_map = self.static_head(x)
            static_map = rearrange(static_map, '(b l) c h w -> b l c h w',
                                   b=b, l=l)

        output_dict = {'static_seg': static_map,
                       'dynamic_seg': dynamic_map}

        return output_dict

class Mini_cooper(nn.Module):
    def __init__(self,
                 tsa_loop : TSA_Loop,
                 encoding_input : Encoding_input,
                 positional_encoding : PositionalEncodeing,
                 decoding_output : Decoding_output,
                 seg_head = BevSegHead):
        super().__init__()

        self.tsa_loop = tsa_loop
        self.encoding_input = encoding_input
        self.positional_encoding = positional_encoding
        self.decoding_output = decoding_output
        self.seg_header = seg_head

    # def add_pe_in_combined_bev(self,combined_bev):
    #     return self.positional_encoding(combined_bev)
    
    def loop_output(self, bev_query, prev_bev):
        bev_query = bev_query.permute(0,2,1)
        bev_query = self.encoding_input(bev_query).permute(0,2,1)
        prev_bev = prev_bev.permute(0,2,1)
        prev_bev = self.encoding_input(prev_bev).permute(0,2,1)

        bev_query = self.positional_encoding(bev_query)
        return self.tsa_loop(bev_query, prev_bev)
    
    def postprocessing_after_model(self, loop_result):
        loop_result = loop_result.permute(0,2,1)
        loop_result = self.decoding_output(loop_result).permute(0,2,1)
        return loop_result
    
    def seg_head(self, x):
        return self.seg_header(x=x,b=1,l=1)

def ysmodelbefore(
        config : dict
    ):

    raw_seq_len = config['deconv']['raw_seq_len']  ## 400 x 400 x c bev 에서 masking을 제외한 sequence의 길이
    seq_len = config['deconv']['seq_len']      ## Mini Cooper에 넣기 위한 줄어든 sequence 길이 
    large_bev_size = config['large_bev_size']  
    bev_emb_dim = config['deconv']['bev_emb_dim']
    N = config['deconv']['N']
    h = config['deconv']['h']
    dropout = config['deconv']['dropout']
    d_feed_forward = config['deconv']['d_feed_forward']
    device = config['deconv']['device']

    positional_embed = PositionalEncodeing(bev_emb_dim,seq_len,dropout,device)
    encoding_input = Encoding_input(raw_seq_len,seq_len,device)
    decoding_output = Decoding_output(seq_len,large_bev_size * large_bev_size,device)
    
    model = []
    for _ in range(N):
        multi_attention_block = DeformableAttention1D(dim = bev_emb_dim,heads=h,device=device)
        feed_forward_block = FeedforwardBlock(bev_emb_dim,d_feed_forward,dropout,device)
        model_block = TSABlock(multi_attention_block,feed_forward_block,dropout)
        model.append(model_block)
    
    tsa_loop = TSA_Loop(nn.ModuleList(model))

    seg_head = BevSegHead('dynamic',bev_emb_dim,2).to(device=device)    ## 768은 일단은 고정된 값으로 사용한다.

    mini_cooper = Mini_cooper(tsa_loop,encoding_input,positional_embed,decoding_output,seg_head)

    return mini_cooper

if  __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ## x (input) : (batch, sequence, dim)
    ## prev_x (input) : (batch, sequence, dim)

    ## 아직 B_t-1 (prev_bev)은 align 하지 않은 상태
    B, C ,H, W = 1, 256, 400, 400
    tsa_input_seq_len = 50 * 50

    bev_query = torch.randn(B,C,H,W) # Ego bev
    prev_bev_prime = torch.randn(B,C,H,W)  # vehicle 01 bev

    flatten_bev_query = rearrange(bev_query, 'b c h w -> b (h w) c')
    flatten_prev_bev_prime = rearrange(prev_bev_prime, 'b c h w -> b (h w) c')

    after_sub_mask_dim = 100 * 100 ## masking을 제거한 상태의 sequence 길이
    example_mask = torch.zeros_like(bev_query)
    example_mask[:,:,:5,:5] = 1

    flatten_bev_query = flatten_bev_query[:,:after_sub_mask_dim,:].to(device)
    flatten_prev_bev_prime = flatten_prev_bev_prime[:,:after_sub_mask_dim,:].to(device)

    print(f'Input Shape : {flatten_bev_query.shape}')

    d_feed_forawrd = 1024
    dropout = 0.1
    model = ysmodelbefore(after_sub_mask_dim,tsa_input_seq_len,C)

    model_output = model.loop_output(flatten_bev_query,flatten_prev_bev_prime)

    print(f'Model Loop Output : {model_output.shape}')

    model_output = model.postprocessing_after_model(model_output)

    print(f'After Postprocessing from model output : {model_output.shape}')

    bev_feature_bf_rpn = torch.zeros_like(bev_query)
    
    # flatten한 model output을 mask에 맞는 크기로 잘라서 사용
    ## model_output의 개수 : after_sub_mask_dim * after_sub_mask_dim * C
    flatten_model_output = model_output.detach().cpu().flatten()[:(example_mask == 1).sum()]       ## 여기서는 임의의 숫자를 사용해서 [:(example_mask == 1).sum()] 사용됫지만 실제로는 dim이 같아야함

    # 마스크가 1인 부분에 값을 할당
    bev_feature_bf_rpn[example_mask == 1] = flatten_model_output

    print(f'Before RPNetwork BEV Feature Shape : {bev_feature_bf_rpn.shape}')

    pred_cls_score, pred_anchor_locs = model.detetion_head_result(bev_feature_bf_rpn)
    
    print(f'Class prediction Result Shape : {pred_cls_score.shape}')
    print(f'Location Prediction Result Shape : {pred_anchor_locs.shape}')
   
    print('---------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------')
    print('                                Finished                                               ')
    print('---------------------------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------')

    

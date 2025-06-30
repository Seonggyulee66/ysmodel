"""
Seg head for bev understanding
"""

import torch
import torch.nn as nn
from einops import rearrange


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
            self.dynamic_head = nn.Conv2d(input_dim,
                                          dynamic_output_class,
                                          kernel_size=3,
                                          padding=1)
            self.static_head = nn.Conv2d(input_dim,
                                         static_output_class,
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



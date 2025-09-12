# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type

# # 在 IMedSAM 中 common 的代码
# # 输入特征：x_freq 是经过FFT变换和Patch Embedding后的频率特征。
# # Adapter的作用是对已经提取的频率特征进行进一步处理和增强
# class Adapter(nn.Module):
#     def __init__(self, D_features, hidden_features=4, act_layer=nn.GELU, skip_connect=True):
#         super().__init__()
#         self.skip_connect = skip_connect
#         D_hidden_features = int(hidden_features)
#         self.act = act_layer()
#         # 特征变换：通过两层全连接层（D_fc1 和 D_fc2）对输入特征进行非线性变换。
#         self.D_fc1 = nn.Linear(D_features, D_hidden_features)
#         self.D_fc2 = nn.Linear(D_hidden_features, D_features)
#
#     def forward(self, x):
#         # x is (BT, HW+1, D)
#         xs = self.D_fc1(x)
#         xs = self.act(xs)
#         xs = self.D_fc2(xs)
#         # 特征融合：通过残差连接（skip_connect）将变换后的特征与原始特征融合。
#         if self.skip_connect:
#             x = x + xs
#         else:
#             x = xs
#         return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

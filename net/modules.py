import torch.nn as nn
import sys

print(sys.prefix)

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from bisect import bisect
import torch.nn.functional as F
import numpy as np
from quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=IntGELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = QuantLinear(
            in_features,
            hidden_features
        )
        self.qact_gelu = QuantAct()

        self.act = act_layer()
        self.qact1 = QuantAct()
        self.fc2 = QuantLinear(
            hidden_features,
            out_features
        )
        self.qact2 = QuantAct(16)

        self.drop = nn.Dropout(drop)

    def forward(self, x, act_scaling_factor):
        x, act_scaling_factor = self.fc1(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_gelu(x, act_scaling_factor)
        x, act_scaling_factor = self.act(x, act_scaling_factor)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x = self.drop(x)
        x, act_scaling_factor = self.fc2(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        x = self.drop(x)
        return x, act_scaling_factor

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = QuantLinear(dim, dim * 3, bias=qkv_bias)

        self.qact1 = QuantAct()
        self.qact_attn1 = QuantAct()
        self.qact_table = QuantAct()
        self.qact2 = QuantAct()
        self.log_int_softmax = IntSoftmax()
        self.qact3 = QuantAct()
        self.qact4 = QuantAct(16)
        self.matmul_1 = QuantMatMul()
        self.matmul_2 = QuantMatMul()


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = QuantLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, act_scaling_factor, add_token=True, token_num=0, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        x, act_scaling_factor = self.qkv(x, act_scaling_factor)
        x, act_scaling_factor_1= self.qact1(x, act_scaling_factor)
        qkv = x.reshape(B_, N, 3, self.num_heads, C //
                        self.num_heads).permute(2, 0, 3, 1, 4)#3 B_ self.num_heads N C //self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn, act_scaling_factor = self.matmul_1(q, act_scaling_factor_1,
                                                 k.transpose(-2, -1), act_scaling_factor_1)
        attn = attn * self.scale
        act_scaling_factor = act_scaling_factor * self.scale

        attn, act_scaling_factor = self.qact_attn1(attn, act_scaling_factor)


        relative_position_bias_table_q, act_scaling_factor_tabel = self.qact_table(
            self.relative_position_bias_table)
        relative_position_bias = relative_position_bias_table_q[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn, act_scaling_factor = self.qact2(attn, act_scaling_factor, relative_position_bias.unsqueeze(0), act_scaling_factor_tabel)



        if mask is not None:
            if add_token:
                # padding mask matrix
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn, act_scaling_factor = self.log_int_softmax(attn, act_scaling_factor)
        else:
            attn, act_scaling_factor = self.log_int_softmax(attn, act_scaling_factor)

        attn = self.attn_drop(attn)
        x, act_scaling_factor = self.matmul_2(attn, act_scaling_factor,
                                              v, act_scaling_factor_1)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)

        x, act_scaling_factor = self.proj(x, act_scaling_factor)
        x, act_scaling_factor = self.qact4(x, act_scaling_factor)
        x = self.proj_drop(x)
        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.qact1 = QuantAct()
        self.reduction = QuantLinear(4 * dim, out_dim, bias=False)
        self.qact2 = QuantAct()
        self.norm = norm_layer(4 * dim)
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, act_scaling_factor):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(self.input_resolution)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H*W//4, 4 * C)  # B H/2*W/2 4*C


        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)

        x, act_scaling_factor = self.norm(x, act_scaling_factor)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x, act_scaling_factor = self.reduction(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops




class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, input_resolution, dim, out_dim, norm_layer=IntLayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.increment = QuantLinear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)

        self.qact1 = QuantAct()
        self.qact2 = QuantAct()
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x, act_scaling_factor):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x, act_scaling_factor = self.norm(x, act_scaling_factor)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x, act_scaling_factor = self.increment(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # print(x.shape)
        x = x.flatten(2).permute(0, 2, 1)
        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = QuantConv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.qact_before_norm = QuantAct()
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.qact = QuantAct(16)
    def forward(self, x, act_scaling_factor):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x, act_scaling_factor = self.proj(x, act_scaling_factor)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x, act_scaling_factor = self.qact_before_norm(x, act_scaling_factor)
            x, act_scaling_factor = self.norm(x, act_scaling_factor)
        x, act_scaling_factor = self.qact(x, act_scaling_factor)

        return x, act_scaling_factor

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

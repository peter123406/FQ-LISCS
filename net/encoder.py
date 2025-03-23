from net.modules import *
import torch
from quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul,Intsigmoid


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=IntGELU,
                 norm_layer=IntLayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.qact1 = QuantAct()
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.qact2 = QuantAct(16)

        self.norm2 = norm_layer(dim)
        self.qact3 = QuantAct()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.qact4 = QuantAct(16)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self,  x_1, act_scaling_factor_1):

        H, W = self.input_resolution
        B, L, C = x_1.shape
        assert L == H * W, "input feature has wrong size"

        x, act_scaling_factor = self.norm1(x_1, act_scaling_factor_1)
        x, act_scaling_factor = self.qact1(x, act_scaling_factor)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows, act_scaling_factor = self.attn(x_windows,act_scaling_factor,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x_2, act_scaling_factor_2 = self.qact2(x, act_scaling_factor, x_1, act_scaling_factor_1)
        x, act_scaling_factor = self.norm2(x_2, act_scaling_factor_2)
        x, act_scaling_factor = self.qact3(x, act_scaling_factor)
        x, act_scaling_factor = self.mlp(x, act_scaling_factor)
        x, act_scaling_factor = self.qact4(x, act_scaling_factor, x_2, act_scaling_factor_2)

        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=IntLayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, act_scaling_factor):
        if self.downsample is not None:
            x, act_scaling_factor = self.downsample(x, act_scaling_factor)
        for _, blk in enumerate(self.blocks):
            x, act_scaling_factor = blk(x, act_scaling_factor)
        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)

class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.QLinear1=QuantLinear(1, M)
        self.qact1=QuantAct()
        # self.GELU1=nn.GELU()

        self.QLinear2=QuantLinear(M, M)
        self.qact2=QuantAct()
        self.qact3=QuantAct()

        # self.GELU2=IntGELU()
        self.qact4=QuantAct()

        self.QLinear3=QuantLinear(M, M)

        self.Sigmoid=Intsigmoid()


    def forward(self, snr):
        x,act_scaling_factor=self.QLinear1(snr,torch.max(snr)/255)
        x=torch.clamp(x,min=0)
        x, act_scaling_factor=self.qact1(x, act_scaling_factor)
        x,act_scaling_factor=self.QLinear2(x, act_scaling_factor)
        x=torch.clamp(x,min=0)
        x, act_scaling_factor=self.qact2(x, act_scaling_factor)
        x,act_scaling_factor=self.QLinear3(x, act_scaling_factor)
        x, act_scaling_factor=self.qact3(x, act_scaling_factor)
        x,act_scaling_factor =self.Sigmoid(x,act_scaling_factor)
        x, act_scaling_factor=self.qact4(x,act_scaling_factor)

        return x,act_scaling_factor

class FQ_LISCS_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=IntLayerNorm, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0])

        self.qact_input = QuantAct()
        self.qact1 = QuantAct(16)

        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.qact_list1=nn.ModuleList()
        self.qact_list2=nn.ModuleList()

        self.sm_list.append(QuantLinear(self.embed_dims[len(embed_dims)-1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims)-1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(QuantLinear(self.hidden_dim, outdim))
            self.qact_list1.append(QuantAct())
            self.qact_list2.append(QuantAct())
        self.intsigmoid = Intsigmoid()
        self.qact2 = QuantAct()
        self.qact3 = QuantAct()
        self.qact4 = QuantAct()
        self.qact5 = QuantAct()
        self.qact6 = QuantAct()
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.head_list = QuantLinear(embed_dims[-1], C)

    def forward(self, x, snr):
        B, C, H, W = x.size()
        device = x.get_device()
        x, act_scaling_factor = self.qact_input(x)
        x, act_scaling_factor = self.patch_embed(x, act_scaling_factor)
        for i_layer, layer in enumerate(self.layers):
            x, act_scaling_factor = layer(x, act_scaling_factor)
        x, act_scaling_factor = self.norm(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp,act_scaling_factor1 = self.sm_list[i](x.detach(),act_scaling_factor.detach())
            else:
                temp,act_scaling_factor1= self.sm_list[i](temp,act_scaling_factor1)
            temp,act_scaling_factor1=self.qact_list1[i](temp,act_scaling_factor1)

            bm,act_scaling_factor2 = self.bm_list[i](snr_batch)
            bm=bm.unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
            temp = temp * bm
            act_scaling_factor1=act_scaling_factor2*act_scaling_factor1
            temp,act_scaling_factor1=self.qact_list2[i](temp,act_scaling_factor1)
        aa,act_scaling_factor3=self.sm_list[-1](temp,act_scaling_factor1)
        aa,act_scaling_factor3=self.qact3(aa,act_scaling_factor3)
        mod_val,act_scaling_factor3=self.intsigmoid(aa,act_scaling_factor3)
        mod_val,act_scaling_factor3=self.qact4(mod_val,act_scaling_factor3)

        x = x * mod_val
        act_scaling_factor=act_scaling_factor*act_scaling_factor3
        x,act_scaling_factor=self.qact5(x,act_scaling_factor)

        x, act_scaling_factor = self.head_list(x, act_scaling_factor)
        x, act_scaling_factor = self.qact6(x, act_scaling_factor)

        return x, act_scaling_factor

    def _init_weights(self, m):
        if isinstance(m, QuantLinear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, QuantLinear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, IntLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))


def create_encoder(**kwargs):
    model = FQ_LISCS_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))

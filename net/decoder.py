from net.modules import *
import torch
from net.encoder import SwinTransformerBlock, AdaptiveModulator
from quantization_utils import QuantLinear, QuantAct, QuantConv2d, IntLayerNorm, IntSoftmax, IntGELU, QuantMatMul,Intsigmoid


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, act_scaling_factor):
        for _, blk in enumerate(self.blocks):
            x, act_scaling_factor = blk(x, act_scaling_factor)

        if self.upsample is not None:
            x, act_scaling_factor = self.upsample(x, act_scaling_factor)
        return x, act_scaling_factor

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class FQ_LISCS_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()
        self.qact_input = QuantAct()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        self.head_list = QuantLinear(C, embed_dims[0])
        #self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.qact_list1=nn.ModuleList()
        self.qact_list2=nn.ModuleList()
        self.sm_list.append(QuantLinear(self.embed_dims[0], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[0]
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
    def forward(self, x,  snr):
        B, L, C = x.size()
        x, act_scaling_factor = self.qact_input(x)
        device = x.get_device()
        x, act_scaling_factor = self.head_list(x, act_scaling_factor)
        x, act_scaling_factor = self.qact2(x, act_scaling_factor)

        # token modulation according to input snr value
        snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
        snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
        for i in range(self.layer_num):
            if i == 0:
                temp,act_scaling_factor1 = self.sm_list[i](x.detach(),act_scaling_factor.detach())
            else:
                temp,act_scaling_factor1= self.sm_list[i](temp,act_scaling_factor1)
            temp,act_scaling_factor1=self.qact_list1[i](temp,act_scaling_factor1)

            bm,act_scaling_factor2 = self.bm_list[i](snr_batch)
            bm=bm.unsqueeze(1).expand(-1, L, -1)

            temp = temp * bm
            act_scaling_factor1=act_scaling_factor1*act_scaling_factor2
            temp,act_scaling_factor1=self.qact_list2[i](temp,act_scaling_factor1)
        aa,act_scaling_factor3=self.sm_list[-1](temp,act_scaling_factor1)
        aa,act_scaling_factor3=self.qact3(aa,act_scaling_factor3)
        aa=torch.clamp(aa,-8,8)
        mod_val,act_scaling_factor3=self.intsigmoid(aa,act_scaling_factor3)
        mod_val,act_scaling_factor3=self.qact4(mod_val,act_scaling_factor3)

        x = x * mod_val
        act_scaling_factor=act_scaling_factor*act_scaling_factor3
        x,act_scaling_factor=self.qact5(x,act_scaling_factor)

        for i_layer, layer in enumerate(self.layers):
            x, act_scaling_factor = layer(x, act_scaling_factor)
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


def create_decoder(**kwargs):
    model = FQ_LISCS_Decoder(**kwargs)
    return model



def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))


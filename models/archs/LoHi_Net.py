import torch
import torch.nn as nn
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.utils.checkpoint as checkpoint
import math
from torch import Tensor
from functools import partial
# from torchstat import stat
import torch.nn.functional as F
import numbers
import numpy as np
from matplotlib import pyplot as plt

from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange, einsum

# from Dwconv.dwconv_layer import DepthwiseFunction
try:
    from .waveelt_utils import _as_wavelet, get_filter_tensors, DWT, IDWT
    from .SSD2D_arch import SS2D, SS2D6, DSSM  # import SS2D and SS2D6
except:
    from waveelt_utils import _as_wavelet, get_filter_tensors, DWT, IDWT
    from SSD2D_arch import SS2D, SS2D6, DSSM  # import SS2D and SS2D6


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=1),
        )

    def forward(self, x):
        x = self.proj(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        x = self.waverec([ya, (yh, yv, yd)], None)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, patch_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)

        self.proj = nn.Sequential(
            nn.Conv2d(input_dim * patch_size ** 2, input_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        ya, (yh, yv, yd) = self.wavedec(x)
        x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=True):
        super().__init__()
        self.channels_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3,
                      stride=1, padding=1, groups=in_features),
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer(),
            nn.Conv2d(hidden_features, out_features, 1),
            nn.Conv2d(out_features, out_features, kernel_size=3,
                      stride=1, padding=1, groups=out_features),
        )

    def forward(self, x):
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2)  # [B,H,W,C] -> [B,C,H,W]

        x = self.mlp(x)

        if not self.channels_first:
            x = x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
        return x


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    """Layer Scale Module"""
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1) * init_value)

    def forward(self, x):
        return x * self.weight


class OurMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        mlp_ratio: float = 2.0,
        stage: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3,
                              padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = DSSM(
            d_model=hidden_dim,
            d_state=d_state,
            dropout=attn_drop_rate,
            dt_init=dt_init,
            stage=stage,
            **kwargs
        )

        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3,
                              padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)

        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            channels_first=False
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        x = x + self.cpe1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        x_cpe2 = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = x + self.cpe2(x_cpe2).permute(0, 2, 3, 1)

        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x


class OurMambaLayer(nn.Module):
    """
    Args:
        dim: input channels
        depth: number of blocks
        attn_drop: attention dropout
        drop_path: stochastic depth
        norm_layer: normalization
        use_checkpoint: gradient checkpointing
    """

    def __init__(
        self,
        dim,
        depth=1,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        d_state=1,
        dt_init="random",
        mlp_ratio=2.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            OurMambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)
        ])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

    def forward(self, x):
        # x: [B, C, H, W]
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class OurStage(nn.Module):
    def __init__(
        self,
        depth=int,
        in_channels=int,
        use_wavelet_scan=False,
        d_state=16,
        mlp_ratio=2.0,
        stage=0,
    ) -> None:
        super(OurStage, self).__init__()
        self.use_wavelet_scan = use_wavelet_scan

        if use_wavelet_scan:
            # input is concat([ya, yh, yv, yd])
            component_dim = in_channels // 4

            self.blocks_ya = nn.ModuleList([
                OurMambaLayer(
                    dim=component_dim,
                    depth=depth * 2,
                    use_checkpoint=False,
                    d_state=d_state,
                    dt_init="random",
                    mlp_ratio=mlp_ratio,
                )
            ])

            self.blocks_high = nn.ModuleList()
            for layer in range(depth):
                self.blocks_high.append(
                    SS2D6_Block(
                        hidden_dim=component_dim,
                        drop_path=0,
                        d_state=d_state,
                        mlp_ratio=mlp_ratio,
                    )
                )

            self.conv_fusechannel = nn.Conv2d(
                component_dim * 2, component_dim, 1, stride=1, bias=False
            )

            self.horizontal_conv, self.vertical_conv, self.diagonal_conv = self.create_wave_conv(dim=component_dim)

        else:
            # Original implementation
            self.blocks = nn.Sequential(*[
                OurMambaLayer(
                    dim=in_channels,
                    depth=depth,
                    use_checkpoint=False,
                    d_state=16,
                    dt_init="random",
                    mlp_ratio=2.0,
                )
            ])

    def create_conv_layer(self, kernel, dim):
        conv = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        conv.weight.data = kernel.repeat(dim, dim, 1, 1)
        return conv

    def create_wave_conv(self, dim):
        horizontal_kernel = torch.tensor([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        vertical_kernel = torch.tensor([[1, 1, 1],
                                        [0, 0, 0],
                                        [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        diagonal_kernel = torch.tensor([[0, 1, 0],
                                        [1, -4, 1],
                                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        horizontal_conv = self.create_conv_layer(horizontal_kernel, dim)
        vertical_conv = self.create_conv_layer(vertical_kernel, dim)
        diagonal_conv = self.create_conv_layer(diagonal_kernel, dim)
        return horizontal_conv, vertical_conv, diagonal_conv

    def forward(self, input=torch.Tensor):
        if self.use_wavelet_scan:
            # input: [B, C, H, W] where C = 4 * component_dim
            ya, yh, yv, yd = torch.chunk(input, 4, dim=1)

            for i in range(len(self.blocks_ya)):
                ya = self.blocks_ya[i](ya)

            res_yh = self.horizontal_conv(ya)
            res_yv = self.vertical_conv(ya)
            res_yd = self.diagonal_conv(ya)

            yh = torch.cat((yh, res_yh), dim=1)
            yv = torch.cat((yv, res_yv), dim=1)
            yd = torch.cat((yd, res_yd), dim=1)

            y_high = torch.cat((yh, yv, yd), dim=0)
            y_high = self.conv_fusechannel(y_high)

            for blk in self.blocks_high:
                y_high = blk(y_high)

            b = ya.shape[0]
            yh, yv, yd = torch.chunk(y_high, 3, dim=0)

            x = torch.cat([ya, yh, yv, yd], dim=1)
            return x
        else:
            output = self.blocks(input)
            return output


class SS2D6_Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        mlp_ratio: float = 2.0,
        scan_type: str = 'hl',
        **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = SS2D6(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            scan_type=scan_type,
            **kwargs
        )

        self.drop_path = DropPath(drop_path)

        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)

        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            channels_first=False
        )

    def forward(self, input: torch.Tensor):
        # input: [B, C, H, W]
        x = input + self.cpe1(input)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        x_cpe2 = x.permute(0, 3, 1, 2)
        x = x + self.cpe2(x_cpe2).permute(0, 2, 3, 1)

        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        )

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


class SCM(nn.Module):
    def __init__(self, dim, factor):
        super(SCM, self).__init__()
        self.sp = nn.Sequential(
            nn.Conv2d(3 * factor ** 2, dim // 2, 1),
            nn.Conv2d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.sp(x)


class DCM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DCM, self).__init__()
        self.sp = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        )

    def forward(self, x):
        return self.sp(x)


class Backbone(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2]):
        super(Backbone, self).__init__()

        self.patch_size = patch_size

        self.patch_embed = SCM(embed_dim[0], 1)
        self.layer1 = OurStage(depth=depth[0], in_channels=embed_dim[0], use_wavelet_scan=False, stage=0)

        self.skip1 = nn.Conv2d(embed_dim[0] + embed_dim[4], embed_dim[4], 1)

        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1])

        self.layer2 = OurStage(depth=depth[1], in_channels=embed_dim[1], use_wavelet_scan=True, stage=1)

        self.skip2 = nn.Conv2d(embed_dim[1] + embed_dim[3], embed_dim[3], 1)

        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2])

        self.layer3 = OurStage(depth=depth[2], in_channels=embed_dim[2], use_wavelet_scan=True, stage=2)

        self.upsample3 = PatchUnEmbed_for_upsample(
            patch_size=2,
            embed_dim=embed_dim[2],
            out_dim=embed_dim[3]
        )

        self.layer4 = OurStage(depth=depth[3], in_channels=embed_dim[3], use_wavelet_scan=True, stage=3)

        self.upsample4 = PatchUnEmbed_for_upsample(
            patch_size=2,
            embed_dim=embed_dim[3],
            out_dim=embed_dim[4]
        )

        self.layer5 = OurStage(depth=depth[4], in_channels=embed_dim[4], use_wavelet_scan=False, stage=4)

        self.patch_unembed = DCM(embed_dim[4], out_chans)

        self.FAM1 = FAM(embed_dim[1])
        self.SCM1 = SCM(embed_dim[1], 2)
        self.FAM2 = FAM(embed_dim[2])
        self.SCM2 = SCM(embed_dim[2], 4)

        self.decoder3 = DCM(embed_dim[2], out_chans * 16)
        self.decoder4 = DCM(embed_dim[3], out_chans * 4)

        wt_type = 'db3'
        self.wavelet = _as_wavelet(wt_type)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wt_type, flip=True
        )

        self.dec_lo = nn.Parameter(dec_lo, requires_grad=False)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=False)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=False)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=False)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wt_type, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wt_type, level=1)

    def forward(self, x):
        # Wavelet decomposition
        ya, (yh, yv, yd) = self.wavedec(x)
        x_1 = torch.cat([ya, yh, yv, yd], dim=1)

        ya, (yh, yv, yd) = self.wavedec(x_1)
        x_2 = torch.cat([ya, yh, yv, yd], dim=1)

        copy0 = x

        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)

        z_1 = self.SCM1(x_1)
        x = self.FAM1(x, z_1)

        x = self.layer2(x)

        copy2 = x
        x = self.downsample2(x)

        z_2 = self.SCM2(x_2)
        x = self.FAM2(x, z_2)

        x = self.layer3(x)
        out_3 = self.decoder3(x) + x_2

        x = self.upsample3(x)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer4(x)
        out_4 = self.decoder4(x) + x_1

        x = self.upsample4(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer5(x)
        out_5 = self.patch_unembed(x) + copy0

        return out_3, out_4, out_5


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    device = 'cuda:3'
    x = torch.randn((1, 3, 224, 224)).to(device)

    net = Backbone().to(device)
    y1, y2, y3 = net(x)
    print(y2.shape)

    input_shape = (3, 224, 224)
    macs, params = get_model_complexity_info(
        net, input_shape,
        as_strings=True,
        print_per_layer_stat=False
    )

    print("MACs:", macs)
    print("Parameters:", params)
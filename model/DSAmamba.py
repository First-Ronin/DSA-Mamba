import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .cross_attention import CrossAttention

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L

    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
    

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)      # (2,7,7,1536)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x = self.norm(x)     # (2,14,14,384)

        return x
    

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride, padding=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # padding on the both ends of time series
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size-1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size-1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))        #
        x = x.permute(0, 2, 1).view(B, H, W, C)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1, padding=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return x, moving_mean

class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            kernel_size=15,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            mambaout=False,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.mambaout = mambaout
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / (16 * 2)) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner*2, bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.hid_proj = nn.Linear(self.d_inner, self.d_inner)

        self.ssm_single = SSM2D(d_state=self.d_state, 
                                dt_rank=self.dt_rank, 
                                d_inner=self.d_inner//2, 
                                device=device, dtype=dtype)
        self.ssm_double = SSM2D(d_state=self.d_state, dt_rank=self.dt_rank, d_inner=self.d_inner//2, device=device, dtype=dtype)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))     # (b, d, h, w)
        x = x.permute(0, 2, 3, 1).contiguous()      # (b, h, w, d)

        wy = self.hid_proj(x)
        w, y = wy.chunk(2, dim=-1)  # (b, h, w, d)

        # y = self.ssm(y)             # x(64,96,7,7)
        # w = self.ssm(w)             # x(64,96,7,7)
        y = self.ssm_single(y)             # x(64,96,7,7)
        w = self.ssm_double(w)             # x(64,96,7,7)
        assert y.dtype == torch.float32
        y = y.contiguous().view(B, H, W, -1)
        w = w.contiguous().view(B, H, W, -1)
        out = torch.cat([w, y], dim=-1)

        out = self.out_norm(out)

        out = out * F.silu(z)

        out = self.out_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SSM2D(nn.Module):
    def __init__(self,
                d_state, 
                dt_rank, 
                d_inner, 
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                device=None,
                dtype=None,
                **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale,
                                         dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, H, W, C = x.shape
        L = H * W
        K = 1

        # x(64,48,7,7) xs(64,4,48,49) self.x_proj_weight(4,35,48) x_dbl(64,4,35,49)
        x_dbl = self.x_proj(x)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)    x_dbl(64,2,38,49)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dts(64,4,48,49)
        dts = F.softplus(self.dt_projs(dts))
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = x.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.contiguous().float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.contiguous().float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)

        out_y = self.selective_scan(    # # xs(64,384,49) dts(64,384,49) As(384) Bs(64,4,16,49)=Cs Ds(384)
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, -1, L)
        assert out_y.dtype == torch.float
        # x(64,48,7,7) out_y(64,2,96,49)

        return out_y


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)

    x = torch.transpose(x, 3, 4).contiguous()

    # flatten
    x = x.view(batch_size, height, width, -1)

    return x

class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        kernel_size: int = 35,
        decoder=False,
        **kwargs,
    ):
        super().__init__()
        self.ln = norm_layer(hidden_dim)
        self.rmder_mamba = Mamba(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, mambaout=False, **kwargs)
        self.trend_mamba = Mamba(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, mambaout=False, **kwargs)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.drop_path = DropPath(drop_path)

        if decoder:
            self.cattn = CrossAttention(hidden_dim//2, key_dim=hidden_dim//2, value_dim=hidden_dim//2)

        self.decompsition = series_decomp(kernel_size)

    def forward(self, input: torch.Tensor, last_input=None):
        if last_input is None:
            x = input
        else:
            x = self.cattn(input, last_input)       # x(768)
        
        x = input
        remainder_x, trend_x = self.decompsition(x)

        remainder_x = remainder_x.permute(0,3,1,2).contiguous()
        remainder_x = self.conv(remainder_x)
        remainder_x = remainder_x.permute(0,2,3,1).contiguous()

        tx = self.drop_path(self.trend_mamba(self.ln(trend_x)))
        rx = self.drop_path(self.rmder_mamba(self.ln(remainder_x)))
        
        output = rx + tx
        output = channel_shuffle(output, groups=2)      # 
        return output+input


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        decoder=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SS_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                decoder=decoder,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, last_x=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, last_x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x
    

class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, in_depths=[2, 2, 4], out_depths=[2, 2],
                 in_dims=[96,192,384], out_dims=[768, 384], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_inlayers = len(in_depths)
        self.num_outlayers = len(out_depths)

        if isinstance(in_dims, int):
            in_dims = [int(in_dims * 2 ** i_layer) for i_layer in range(self.num_inlayers)]
        if isinstance(out_dims, int):
            out_dims = [int(out_dims * 2 ** i_layer) for i_layer in range(self.num_outlayers)]

        self.embed_dim = in_dims[0]
        self.num_features = out_dims[-1]

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        int_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(in_depths))]  # stochastic depth decay rule
        out_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(out_depths))]  # stochastic depth decay rule

        self.Encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_inlayers):
            layer = VSSLayer(
                dim=in_dims[i_layer],
                depth=in_depths[i_layer],
                d_state=math.ceil(in_dims[0] / 6) if d_state is None else d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=int_dpr[sum(in_depths[:i_layer]):sum(in_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_inlayers) else None,
                decoder=False,
                use_checkpoint=use_checkpoint,
            )
            self.Encoder_layers.append(layer)

        self.Decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_outlayers):
            layer = VSSLayer(
                dim=out_dims[i_layer],
                depth=out_depths[i_layer],
                d_state=math.ceil(out_dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=out_dpr[sum(out_depths[:i_layer]):sum(out_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchExpand2D if (i_layer < self.num_outlayers - 1) else None,
                decoder=True,
                use_checkpoint=use_checkpoint,
            )
            self.Decoder_layers.append(layer)

        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in SS_Conv_SSM, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, SS_Conv_SSM initialization is useless
        
        Conv2D is not intialized !!!
        """
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

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

        xl = []
        for layer in self.Encoder_layers:
            x = layer(x)  # (96) (192) (384) (768)
            xl.append(x)

        for layer, i in zip(self.Decoder_layers, range(self.num_outlayers)):
            x = layer(x, xl[1-i])  # (96) (192) (384) (768)
        return x

    def forward(self, x):
        x = self.forward_backbone(x)
        x = x.permute(0,3,1,2)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x


# model = VSSM(num_classes=6).to("cuda")

# data = torch.randn(2,3,224,224).to("cuda")

# print(model(data).shape)

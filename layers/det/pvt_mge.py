import math
from typing import List, Optional, Tuple
from warnings import formatwarning
from functools import partial
import megengine
import megengine.functional as F
import megengine.module as M

from .utils import DropPath, to_2tuple, trunc_normal_

# from timm.models.layers.weight_init import trunc_normal_


class DWConv(M.Module):
    def __init__(self, dim: int = 768) -> None:
        super(DWConv, self).__init__()
        self.dwconv = M.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: megengine.Tensor, H: int, W: int) -> megengine.Tensor:
        B, N, C = x.shape
        size = list(range(len(x.shape)))
        size[1], size[2] = size[2], size[1]
        x = x.transpose(*size).reshape(B, C, H, W)
        x = self.dwconv(x)

        x = F.flatten(x, 2)
        size = list(range(len(x.shape)))
        size[1], size[2] = size[2], size[1]
        x = x.transpose(*size)

        return x


class Mlp(M.Module):
    def __init__(self, in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: M.Module = M.GELU,
                 drop: float = 0.,
                 linear: bool = False) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = M.Linear(hidden_features, out_features)
        self.drop = M.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = M.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m: M.Module) -> None:
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.fill_(m.bias, 0)
        elif isinstance(m, M.LayerNorm):
            M.init.fill_(m.bias, 0)
            M.init.fill_(m.weight, 1.0)
        elif isinstance(m, M.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, mean=0, std=math.sqrt(2. / fan_out))
            if m.bias is not None:
                M.init.zeros_(m.bias)

    def forward(self, x: megengine.Tensor, H: int, W: int) -> megengine.Tensor:
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(M.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        sr_ratio: int = 1,
        linear: bool = False
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = M.Linear(dim, dim, bias=qkv_bias)
        self.kv = M.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = M.Dropout(attn_drop)
        self.proj = M.Linear(dim, dim)
        self.proj_drop = M.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = M.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = M.LayerNorm(dim)
        else:
            self.pool = M.AdaptiveAvgPool2d(7)
            self.sr = M.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = M.LayerNorm(dim)
            self.act = M.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m: megengine.Tensor):
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.fill_(m.bias, 0)
        elif isinstance(m, M.LayerNorm):
            M.init.fill_(m.bias, 0)
            M.init.fill_(m.weight, 1.0)
        elif isinstance(m, M.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, mean=0, std=math.sqrt(2. / fan_out))
            if m.bias is not None:
                M.init.zeros_(m.bias)

    def forward(self, x: megengine.Tensor, H: int, W: int) -> megengine.Tensor:
        B, N, C = x.shape
        q: megengine.Tensor = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).transpose(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            x_ = x.transpose(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        size = list(range(len(k.shape)))
        size[-2], size[-1] = size[-1], size[-2]

        attn = F.matmul(q, k.transpose(*size)) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = F.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(M.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: M.Module = M.GELU,
                 norm_layer: M.Module = M.LayerNorm,
                 sr_ratio: int = 1,
                 linear: bool = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else M.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m: megengine.Tensor) -> None:
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.fill_(m.bias, 0)
        elif isinstance(m, M.LayerNorm):
            M.init.fill_(m.bias, 0)
            M.init.fill_(m.weight, 1.0)
        elif isinstance(m, M.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, mean=0, std=math.sqrt(2. / fan_out))
            if m.bias is not None:
                M.init.zeros_(m.bias)

    def forward(self, x: megengine.Tensor, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(M.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 7,
                 stride: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = M.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = M.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: megengine.Tensor) -> None:
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.fill_(m.bias, 0)
        elif isinstance(m, M.LayerNorm):
            M.init.fill_(m.bias, 0)
            M.init.fill_(m.weight, 1.0)
        elif isinstance(m, M.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, mean=0, std=math.sqrt(2. / fan_out))
            if m.bias is not None:
                M.init.zeros_(m.bias)

    def forward(self, x: megengine.Tensor) -> Tuple[megengine.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = F.flatten(x, 2)
        x = x.transpose(0, 2, 1)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(M.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dims: Tuple[int] = [64, 128, 256, 512],
                 num_heads: Tuple[int] = [1, 2, 4, 8],
                 mlp_ratios: Tuple[int] = [4, 4, 4, 4],
                 qkv_bias: bool = False,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer: M.Module = M.LayerNorm,
                 depths: Tuple[int] = [3, 4, 6, 3],
                 sr_ratios: Tuple[int] = [8, 4, 2, 1],
                 num_stages: int = 4,
                 linear: bool = False,
                 pretrained: bool = None) -> None:

        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in F.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            block = [
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    linear=linear
                )
                for j in range(depths[i])]
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = M.Linear(embed_dims[3], num_classes) if num_classes > 0 else M.Identity()

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m: megengine.Tensor) -> None:
        if isinstance(m, M.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, M.Linear) and m.bias is not None:
                M.init.fill_(m.bias, 0)
        elif isinstance(m, M.LayerNorm):
            M.init.fill_(m.bias, 0)
            M.init.fill_(m.weight, 1.0)
        elif isinstance(m, M.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            M.init.normal_(m.weight, mean=0, std=math.sqrt(2. / fan_out))
            if m.bias is not None:
                M.init.zeros_(m.bias)

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        # for model in glob.glob('./*.pth'):
        #     mge_weight = {k: v.numpy() for k,v in torch.load(model, map_location='cpu').items()}
        #     megengine.save(mge_weight, model.replace('.pth','.pkl'))
        
        if isinstance(pretrained, str):
            weight = megengine.load(pretrained)
            for k, v in self.named_parameters():
                weight[k] = weight[k].reshape(tuple(v.shape))
            self.load_state_dict(weight, strict=False)

    def freeze_patch_emb(self) -> None:
        self.patch_embed1.requires_grad = False

    def no_weight_decay(self) -> None:
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self) -> None:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = '') -> None:
        self.num_classes = num_classes
        self.head = M.Linear(self.embed_dim, num_classes) if num_classes > 0 else M.Identity()

    def extract_features(self, x: megengine.Tensor) -> List[megengine.Tensor]:
        B = x.shape[0]
        outs = {}

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
            outs[i] = x

        return outs
        # for i in range(self.num_stages):
        #     patch_embed = getattr(self, f"patch_embed{i + 1}")
        #     block = getattr(self, f"block{i + 1}")
        #     norm = getattr(self, f"norm{i + 1}")
        #     x, H, W = patch_embed(x)
        #     for blk in block:
        #         x = blk(x, H, W)
        #     x = norm(x)
        #     if i != self.num_stages - 1:
        #         x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)

        # return x.mean(axis=1)

    def forward(self, x: megengine.Tensor):
        x = self.extract_features(x)
        x = self.head(x)

        return x

pvt_config = {
    'pvt_v2_b0': [32, 64, 160, 256],
    'pvt_v2_b1': [64, 128, 320, 512],
    'pvt_v2_b2': [64, 128, 320, 512],
    'pvt_v2_b2_li': [64, 128, 320, 512],
    'pvt_v2_b3': [64, 128, 320, 512],
    'pvt_v2_b4': [64, 128, 320, 512],
    'pvt_v2_b5': [64, 128, 320, 512],
}


class pvt_v2_b0(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b1(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b2(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b2_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b3(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b4(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))


class pvt_v2_b5(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs.get('pretrained', None))

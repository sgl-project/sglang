import collections
import math
from enum import Enum
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from sglang.srt.configs import DeepseekVL2Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


def _no_grad_tunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        dtype = tensor.dtype
        tensor_fp32 = tensor.float()
        tensor_fp32 = _no_grad_tunc_normal_(tensor_fp32, mean, std, a, b)
        tensor_dtpye = tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtpye)


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)


def init_weights_vit_timm(
    module: nn.Module,
) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


# Imported multimodal helpers from timm
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


# https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
def resample_patch_embed(
    patch_embed,
    new_size: List[int],
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np

    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias
        )[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.0
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(
        np.linalg.pinv(resize_mat.T), device=patch_embed.device
    )

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv2d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(
                    resample_patch_embed(self.proj.weight, new_patch_size, verbose=True)
                )
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(
                img_size
            )

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert (
                    H == self.img_size[0]
                ), f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert (
                    W == self.img_size[1]
                ), f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            elif not self.dynamic_img_pad:
                assert (
                    H % self.patch_size[0] == 0
                ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."

                assert (
                    W % self.patch_size[1] == 0
                ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


# Copied from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks

    NOTE: When use_conv=True, expects 2D NCHW tensors, otherwise N*C expected.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        if use_conv:
            self.fc1 = nn.Conv2d(
                in_features, hidden_features, kernel_size=1, bias=bias[0]
            )
        else:
            self.fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        # self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        if use_conv:
            self.fc2 = nn.Conv2d(
                hidden_features, out_features, kernel_size=1, bias=bias[1]
            )
        else:
            self.fc2 = RowParallelLinear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        deterministic: bool = False,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=dim,
            # qkv_bias=qkv_bias,
            # qk_norm=qk_norm,
            # attn_drop=attn_drop,
            # proj_drop=proj_drop,
            # norm_layer=norm_layer,
            # deterministic=deterministic,
            layer_id=layer_id,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


LayerType = Union[str, Callable, Type[torch.nn.Module]]


class DeepseekVL2VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "map",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
        ignore_head: bool = False,
        deterministic: bool = False,
        num_recomputing_layers: int = 0,
    ):

        super.__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = partial(nn.GELU, approximate="tanh")

        self.num_classes = num_classes  # always 0
        self.global_pool = global_pool  # always map
        self.num_features = self.embed_dim = embed_dim  # diff
        self.num_prefix_tokens = 1
        self.num_prefix_tokens += reg_tokens  # always 0
        self.has_class_token = class_token  # always true
        self.no_embed_class = no_embed_class  # always false
        self.dynamic_img_size = dynamic_img_size  # always false
        self.grad_checkpointing = False  # always false
        self.ignore_head = ignore_head  # always false

        embed_args = {}
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )

        num_patches = self.patch_embed.num_patches
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        )
        self.reg_token = (
            nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        )
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.patch_drop = nn.Identity()
        self.norm_pre = (
            norm_layer(embed_dim) if pre_norm else nn.Identity()
        )  # always false

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    deterministic=deterministic,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # 初始化权重
        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode="") -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:

        to_cat = []
        pos_embed = self.pos_embed
        # need check
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "is_first_stage", True):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        x = self.blocks(x)
        if getattr(self, "is_last_stage", True):
            x = self.norm(x)
        return x


class DeepseekVL2MlpProjector(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super.__init__()
        self.config = config
        self.quant_config = quant_config
        if config.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.depth
            mlp_ratio = config.mlp_ratio
            modules = [
                nn.Linear(
                    config.input_dim
                    * config.downsample_ratio
                    * config.downsample_ratio,
                    config.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(config.n_embed * mlp_ratio, config.n_embed * mlp_ratio)
                )
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.n_embed * mlp_ratio, config.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise NotImplementedError(
                f"Unsupported projector type: {config.projector_type}"
            )

        self.layers = modules

    def forward(self, x):
        bs, hw, input_dim = x.shape
        h = w = int((hw) ** 0.5)

        """compute padding"""
        if h % self.cfg.downsample_ratio:
            pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
        else:
            pad = 0
        x = x.reshape(bs, h, w, input_dim)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

        """4 to 1 concat"""
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = F.unfold(
            x,
            kernel_size=self.cfg.downsample_ratio,
            stride=self.cfg.downsample_ratio,
            padding=0,
        )  # B, C*4, HW // 4
        x = x.permute(0, 2, 1)

        return self.layers(x)


# todo
class DeepseekVL2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: DeepseekVL2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = DeepseekVL2VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers,
            quant_config=quant_config,
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = DeepseekVL2MlpProjector(projector_config, quant_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.language.forward(
            input_ids=input_ids, positions=positions, forward_batch=forward_batch
        )

        return outputs

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **ignore_kwargs,
    ):
        # todo: find the get_input_embeddings api
        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(images[idx, : batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]

                tile_index += num_tiles_in_image + 1

                # ----------------- global view add newline -----------------
                # [hw, D] -> [h, w, D]
                global_features = global_features.view(h, w, n_dim)
                # [D]     -> [h, 1, D]
                new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                global_features = torch.cat(
                    [global_features, new_lines_in_global], dim=1
                )
                # [h, w + 1, D] -> [h * (w + 1), D]
                global_features = global_features.view(-1, n_dim)

                # ----------------- local view add newline -----------------
                # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                local_features = rearrange(
                    local_features,
                    "(th tw) (h w) d -> (th h) (tw w) d",
                    th=num_height_tiles,
                    tw=num_width_tiles,
                    h=h,
                    w=w,
                )

                # [D] -> [num_height_tiles * h, 1, D]
                new_lines_in_local = repeat(
                    self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h
                )

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                local_features = local_features.view(-1, n_dim)

                # ----------------- merge global and local tiles -----------------
                if self.global_view_pos == "head":
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :], local_features],
                        dim=0,
                    )
                else:
                    global_local_features = torch.cat(
                        [local_features, self.view_seperator[None, :], global_features],
                        dim=0,
                    )

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(
                    images_seq_mask[idx].unsqueeze(-1), images_in_this_batch
                )

        return input_embeds


EntryClass = DeepseekVL2ForCausalLM

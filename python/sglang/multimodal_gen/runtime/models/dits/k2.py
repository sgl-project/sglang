"""Krea-2 (K2) single-stream MMDiT.

Text and image tokens are concatenated into a single joint-attention stream. The
model uses GQA attention with a sigmoid output gate, 6-way shared adaLN
modulation, a text-fusion transformer that fuses the selected text-encoder
hidden-state layers into one, and interleaved 3-axis RoPE. Module and parameter
names follow the released K2 checkpoint, so weights load without remapping.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from sglang.multimodal_gen.configs.models.dits.k2 import K2DitConfig
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.attention.layer import build_varlen_mask_meta
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# Functional helpers
# --------------------------------------------------------------------------- #
def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(period)
        * torch.arange(half, dtype=torch.float32, device=device)
        / half
    )
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


def norm_scale_shift(
    x: Tensor, weight: Tensor, scale: Tensor, shift: Tensor, eps: float
) -> Tensor:
    """Fused RMSNorm + modulation: ``rms_norm(x) * weight * (1 + scale) + shift``.

    ``weight`` is the effective RMSNorm weight (K2 stores ``scale``, so callers
    pass ``scale + 1``), kept off the checkpoint so the identity load is unaffected.
    """
    if x.is_cuda and x.shape[-1] % 256 == 0:
        from sglang.jit_kernel.diffusion.cutedsl.scale_residual_norm_scale_shift import (
            fused_norm_scale_shift,
        )

        return fused_norm_scale_shift(
            x.contiguous(),
            weight.contiguous(),
            None,
            scale.contiguous(),
            shift.contiguous(),
            "rms",
            eps,
        )
    normed = F.rms_norm(x.float(), (x.shape[-1],), weight=weight.float(), eps=eps)
    return (normed.to(x.dtype) * (1 + scale) + shift).to(x.dtype)


# --------------------------------------------------------------------------- #
# Submodules
# --------------------------------------------------------------------------- #
class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(
            6, dim=-1
        )
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [
                rope(pos[..., i], d, self.theta, self.ntk)
                for i, d in enumerate(self.axdims)
            ],
            dim=-3,
        )


class RMSNorm(nn.Module):
    """RMSNorm with effective weight ``scale + 1`` (``scale`` initialized to 0),
    computed in fp32."""

    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0)
        )
        return t.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(nn.Module):
    def __init__(
        self, features: int, multiplier: int, bias: bool = False, multiple: int = 128
    ):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        # Tensor-parallel: gate/up shard the hidden dim by column, down all-reduces.
        self.gate = ColumnParallelLinear(
            features, mlpdim, bias=bias, gather_output=False
        )
        self.up = ColumnParallelLinear(features, mlpdim, bias=bias, gather_output=False)
        self.down = RowParallelLinear(
            mlpdim, features, bias=bias, input_is_parallel=True
        )

    def forward(self, x: Tensor) -> Tensor:
        gate, _ = self.gate(x)
        up, _ = self.up(x)
        out, _ = self.down(F.silu(gate) * up)
        return out


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        # Tensor-parallel: q/k/v/gate shard heads by column, wo all-reduces.
        # Separate (non-fused) parallel linears keep the reference param names, so
        # the identity checkpoint mapping holds (each shards via its weight_loader).
        tp = get_tp_world_size()
        assert (
            self.heads % tp == 0 and self.kvheads % tp == 0
        ), f"heads={self.heads}, kvheads={self.kvheads} must be divisible by tp={tp}"
        self.local_heads = self.heads // tp
        self.local_kvheads = self.kvheads // tp

        self.wq = ColumnParallelLinear(
            dim, self.headdim * self.heads, bias=bias, gather_output=False
        )
        self.wk = ColumnParallelLinear(
            dim, self.headdim * self.kvheads, bias=bias, gather_output=False
        )
        self.wv = ColumnParallelLinear(
            dim, self.headdim * self.kvheads, bias=bias, gather_output=False
        )
        self.gate = ColumnParallelLinear(dim, dim, bias=bias, gather_output=False)
        self.qknorm = QKNorm(self.headdim)
        self.wo = RowParallelLinear(dim, dim, bias=bias, input_is_parallel=True)
        # Native GQA flash via the platform backend; parameterless.
        self.attn = USPAttention(
            num_heads=self.local_heads,
            head_size=self.headdim,
            num_kv_heads=self.local_kvheads,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
        )

    def forward(
        self,
        qkv: Tensor,
        freqs: Tensor | None = None,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
    ) -> Tensor:
        q, _ = self.wq(qkv)
        k, _ = self.wk(qkv)
        v, _ = self.wv(qkv)
        gate, _ = self.gate(qkv)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.local_heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.local_kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.local_kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)

        # USPAttention expects [B, S, H, D]; a [B, S] key mask + varlen metadata
        # routes a ragged batch through the FA varlen fast path, else maskless.
        out = self.attn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            attn_mask=key_mask,
            attn_mask_meta=mask_meta,
        ).flatten(2)
        out, _ = self.wo(out * F.sigmoid(gate))
        return out


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = norm_scale_shift(x, self.norm.scale + 1, scale, shift, self.norm.eps)
        x = self.linear(x)
        return x


class TextFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
    ) -> Tensor:
        x = x + self.attn(self.prenorm(x), key_mask=key_mask, mask_meta=mask_meta)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    """Fuses `num_txt_layers` selected encoder hidden-state layers into one.

    Depth is fixed at 2 layerwise + 2 refiner blocks; `num_txt_layers` is the
    projector input width (the layer axis), NOT the transformer depth.
    """

    def __init__(
        self,
        num_txt_layers: int,
        txt_dim: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads)
                for _ in range(2)
            ]
        )

    def forward(
        self,
        x: Tensor,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
    ) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous())
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, key_mask=key_mask, mask_meta=mask_meta)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        features: int,
        heads: int,
        multiplier: int,
        bias: bool = False,
        kvheads: int = None,
    ):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        freqs: Tensor,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn(
            norm_scale_shift(
                x, self.prenorm.scale + 1, prescale, preshift, self.prenorm.eps
            ),
            freqs,
            key_mask,
            mask_meta,
        )
        x = x + postgate * self.mlp(
            norm_scale_shift(
                x, self.postnorm.scale + 1, postscale, postshift, self.postnorm.eps
            )
        )
        return x


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class K2Transformer2DModel(CachableDiT):
    """K2 single-stream MMDiT for the SGLang diffusion runtime.

    Attribute names follow the released K2 checkpoint, so weights load with an
    identity ``param_names_mapping``.
    """

    _fsdp_shard_conditions = []
    _compile_conditions = []
    param_names_mapping = K2DitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: K2DitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[Any] = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        ac = config.arch_config
        self.arch_config = ac

        self.hidden_size = ac.features
        self.num_attention_heads = ac.heads
        self.num_channels_latents = ac.channels
        self.patch = ac.patch
        self.channels = ac.channels
        self.tdim = ac.tdim

        head_dim = ac.features // ac.heads
        axes = list(ac.axes_dims)
        assert sum(axes) == head_dim, f"sum(axes)={sum(axes)}, head_dim={head_dim}"
        assert all(a % 2 == 0 for a in axes), f"axes={axes}"

        self.posemb = PositionalEncoding(ac.features, axes, theta=ac.theta, ntk=1.0)
        self.first = nn.Linear(ac.channels * ac.patch**2, ac.features, bias=True)
        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    ac.features, ac.heads, ac.multiplier, ac.bias, ac.kvheads
                )
                for _ in range(ac.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(ac.tdim, ac.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac.features, ac.features),
        )
        self.txtfusion = TextFusionTransformer(
            ac.txtlayers,
            ac.txtdim,
            ac.txtheads,
            ac.multiplier,
            ac.bias,
            ac.txtkvheads,
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(ac.txtdim),
            nn.Linear(ac.txtdim, ac.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac.features, ac.features),
        )
        self.last = LastLayer(ac.features, ac.patch, ac.channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"), nn.Linear(ac.features, ac.features * 6)
        )
        self.seq_multiple_of = ac.seq_multiple_of

    def _forward_impl(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        # A single or same-prompt batch has no padding, so attention runs maskless
        # (native-GQA flash). A ragged batch builds varlen metadata from the
        # key mask and takes the FA varlen path instead.
        txt_key = txt_meta = joint_key = joint_meta = None
        if mask is not None and not bool(mask.all()):
            txt_key = mask[:, : context.shape[1]]
            txt_meta = build_varlen_mask_meta(txt_key)
            joint_key = mask
            joint_meta = build_varlen_mask_meta(mask)

        context = self.txtfusion(context, key_mask=txt_key, mask_meta=txt_meta)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        freqs = self.posemb(pos)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, joint_key, joint_meta)

        final = self.last(combined, t)
        output = final[:, txtlen : txtlen + imglen, :]
        return output

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states_image=None,
        guidance=None,
        pos: Tensor = None,
        mask: Tensor = None,
        **kwargs,
    ) -> Tensor:
        return self._forward_impl(
            img=hidden_states,
            context=encoder_hidden_states,
            t=timestep,
            pos=pos,
            mask=mask,
        )


EntryClass = [K2Transformer2DModel]

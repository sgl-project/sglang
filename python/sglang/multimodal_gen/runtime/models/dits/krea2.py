"""Krea-2 (K2) single-stream MMDiT.

Text and image tokens are concatenated into a single joint-attention stream. The
model uses GQA attention with a sigmoid output gate, 6-way shared adaLN
modulation, a text-fusion transformer that fuses the selected text-encoder
hidden-state layers into one, and interleaved 3-axis RoPE. Module and parameter
names follow the released K2 checkpoint, so weights load without remapping.
"""

import math
import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from sglang.multimodal_gen.configs.models.dits.krea2 import Krea2DitConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.attention.layer import build_varlen_mask_meta
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
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


def _fused_qknorm_rope_enabled() -> bool:
    return os.getenv("SGLANG_ENABLE_FUSED_QKNORM_ROPE", "1").lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _can_use_fused_qknorm_rope(head_dim: int, dtype: torch.dtype) -> bool:
    from sglang.kernels.ops.diffusion.qknorm_rope import (
        can_use_fused_inplace_qknorm_rope,
    )

    return can_use_fused_inplace_qknorm_rope(head_dim, head_dim, False, dtype)


def _qknorm_rope_cos_sin_cache(freqs: Tensor) -> Tensor:
    """``[num_tokens, head_dim]`` cos|sin cache for the fused QKNorm+RoPE kernel.

    K2's ``rope`` packs each token's rotation as ``[[cos, -sin], [sin, cos]]`` in a
    ``[B, N, head_dim//2, 2, 2]`` tensor; the kernel wants the per-token cosines then
    sines concatenated. Positions come from the image grid (batch-invariant), so the
    first batch row is representative.
    """
    return torch.cat([freqs[0, :, :, 0, 0], freqs[0, :, :, 1, 0]], dim=-1).float()


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
        from sglang.kernels.ops.diffusion.cutedsl.scale_residual_norm_scale_shift import (
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
class TimeEmbed(nn.Module):
    """Timestep embedding MLP: linear_1 -> gelu(tanh) -> linear_2."""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.gelu(self.linear_1(x), approximate="tanh"))


class TxtIn(nn.Module):
    """Text-context projection: rms-norm -> linear_1 -> gelu(tanh) -> linear_2."""

    def __init__(self, txt_dim: int, dim: int):
        super().__init__()
        self.norm = RMSNorm(txt_dim)
        self.linear_1 = nn.Linear(txt_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.gelu(self.linear_1(self.norm(x)), approximate="tanh"))


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
    """RMSNorm with effective scale ``weight + 1`` (``weight`` initialized to 0),
    computed in fp32. The parameter is named ``weight`` to match the released
    checkpoint; the ``+ 1`` is applied in the forward."""

    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.weight = nn.Parameter(
            torch.zeros(features, device=device, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(
            t, (self.features,), eps=self.eps, weight=(self.weight.float() + 1.0)
        )
        return t.to(dtype)


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

        # Tensor-parallel: q/k/v/gate shard heads by column, to_out all-reduces.
        # Parameter names match the released checkpoint (to_q/to_k/to_v/to_gate,
        # norm_q/norm_k, to_out.0) so the checkpoint loads with an identity mapping.
        tp = get_tp_world_size()
        assert (
            self.heads % tp == 0 and self.kvheads % tp == 0
        ), f"heads={self.heads}, kvheads={self.kvheads} must be divisible by tp={tp}"
        self.local_heads = self.heads // tp
        self.local_kvheads = self.kvheads // tp

        self.to_q = ColumnParallelLinear(
            dim, self.headdim * self.heads, bias=bias, gather_output=False
        )
        self.to_k = ColumnParallelLinear(
            dim, self.headdim * self.kvheads, bias=bias, gather_output=False
        )
        self.to_v = ColumnParallelLinear(
            dim, self.headdim * self.kvheads, bias=bias, gather_output=False
        )
        self.to_gate = ColumnParallelLinear(dim, dim, bias=bias, gather_output=False)
        self.norm_q = RMSNorm(self.headdim)
        self.norm_k = RMSNorm(self.headdim)
        # to_out is a ModuleList ([linear]) so the param is to_out.0.weight, matching
        # the diffusers Attention layout in the released checkpoint.
        self.to_out = nn.ModuleList(
            [RowParallelLinear(dim, dim, bias=bias, input_is_parallel=True)]
        )
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
        num_replicated_prefix: int = 0,
        skip_sequence_parallel: bool = False,
    ) -> Tensor:
        q, _ = self.to_q(qkv)
        k, _ = self.to_k(qkv)
        v, _ = self.to_v(qkv)
        gate, _ = self.to_gate(qkv)

        hd = self.headdim
        # Fast path: fuse RMSNorm(q), RMSNorm(k) and RoPE into one in-place kernel on
        # the [B, S, H, D] layout USPAttention consumes (also skips the [B, H, L, D]
        # transpose round-trip the eager path needs). Eager fallback below preserves
        # parity off CUDA / for unsupported dtypes.
        if (
            freqs is not None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and _fused_qknorm_rope_enabled()
            and _can_use_fused_qknorm_rope(hd, q.dtype)
        ):
            from sglang.kernels.ops.diffusion.qknorm_rope import (
                fused_inplace_qknorm_rope,
            )

            b, s = qkv.shape[0], qkv.shape[1]
            q = q.view(b, s, self.local_heads, hd)
            k = k.view(b, s, self.local_kvheads, hd)
            v = v.view(b, s, self.local_kvheads, hd)
            positions = torch.arange(s, device=q.device, dtype=torch.long)
            if b > 1:
                positions = positions.repeat(b)
            fused_inplace_qknorm_rope(
                q.reshape(-1, self.local_heads, hd),
                k.reshape(-1, self.local_kvheads, hd),
                (self.norm_q.weight.float() + 1.0).to(q.dtype),
                (self.norm_k.weight.float() + 1.0).to(k.dtype),
                _qknorm_rope_cos_sin_cache(freqs),
                positions,
                is_neox=False,
                eps=self.norm_q.eps,
                head_dim=hd,
                rope_dim=hd,
            )
            out = self.attn(
                q,
                k,
                v,
                attn_mask=key_mask,
                attn_mask_meta=mask_meta,
                num_replicated_prefix=num_replicated_prefix,
                skip_sequence_parallel_override=skip_sequence_parallel,
            ).flatten(2)
        else:
            q, k, v = (
                rearrange(q, "B L (H D) -> B H L D", H=self.local_heads),
                rearrange(k, "B L (H D) -> B H L D", H=self.local_kvheads),
                rearrange(v, "B L (H D) -> B H L D", H=self.local_kvheads),
            )
            q, k = self.norm_q(q), self.norm_k(k)
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
                num_replicated_prefix=num_replicated_prefix,
                skip_sequence_parallel_override=skip_sequence_parallel,
            ).flatten(2)
        out, _ = self.to_out[0](out * F.sigmoid(gate))
        return out


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, features))

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        mod = tvec + rearrange(self.scale_shift_table, "two d -> 1 two d")
        scale, shift = mod.chunk(2, dim=1)
        x = norm_scale_shift(x, self.norm.weight + 1, scale, shift, self.norm.eps)
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
        self.norm1 = RMSNorm(features)
        self.norm2 = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.ff = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        x: Tensor,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
    ) -> Tensor:
        # Text-fusion runs on the full replicated text, so skip the SP all-to-all.
        x = x + self.attn(
            self.norm1(x),
            key_mask=key_mask,
            mask_meta=mask_meta,
            skip_sequence_parallel=True,
        )
        x = x + self.ff(self.norm2(x))
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
        # (6, features) modulation table added to the timestep projection (AdaLN-single),
        # stored directly on the block to match the released checkpoint.
        self.scale_shift_table = nn.Parameter(torch.zeros(6, features))
        self.norm1 = RMSNorm(features)
        self.norm2 = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.ff = SwiGLU(features, multiplier, bias)

    def forward(
        self,
        hidden_states: Tensor,
        vec: Tensor,
        freqs: Tensor,
        key_mask: Tensor | None = None,
        mask_meta: dict | None = None,
        num_replicated_prefix: int = 0,
    ) -> Tensor:
        mod = vec + self.scale_shift_table.reshape(-1)
        prescale, preshift, pregate, postscale, postshift, postgate = mod.chunk(
            6, dim=-1
        )
        hidden_states = hidden_states + pregate * self.attn(
            norm_scale_shift(
                hidden_states,
                self.norm1.weight + 1,
                prescale,
                preshift,
                self.norm1.eps,
            ),
            freqs,
            key_mask,
            mask_meta,
            num_replicated_prefix=num_replicated_prefix,
        )
        hidden_states = hidden_states + postgate * self.ff(
            norm_scale_shift(
                hidden_states,
                self.norm2.weight + 1,
                postscale,
                postshift,
                self.norm2.eps,
            )
        )
        return hidden_states


# --------------------------------------------------------------------------- #
# Top-level model
# --------------------------------------------------------------------------- #
class Krea2Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """K2 single-stream MMDiT for the SGLang diffusion runtime.

    Attribute names follow the released K2 checkpoint, so weights load with an
    identity ``param_names_mapping``.
    """

    _fsdp_shard_conditions = []
    _compile_conditions = []
    param_names_mapping = Krea2DitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: Krea2DitConfig,
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
        self.img_in = nn.Linear(ac.channels * ac.patch**2, ac.features, bias=True)
        self.transformer_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    ac.features, ac.heads, ac.multiplier, ac.bias, ac.kvheads
                )
                for _ in range(ac.layers)
            ]
        )
        self.time_embed = TimeEmbed(ac.tdim, ac.features)
        self.text_fusion = TextFusionTransformer(
            ac.txtlayers,
            ac.txtdim,
            ac.txtheads,
            ac.multiplier,
            ac.bias,
            ac.txtkvheads,
        )
        self.txt_in = TxtIn(ac.txtdim, ac.features)
        self.final_layer = LastLayer(ac.features, ac.patch, ac.channels)
        # GELU(tanh) is applied in the forward; the linear matches time_mod_proj.weight.
        self.time_mod_proj = nn.Linear(ac.features, ac.features * 6)
        self.seq_multiple_of = ac.seq_multiple_of
        # The 28 single-stream blocks (the ~24GB bulk) are streamed layer-by-layer
        # under --dit-layerwise-offload, keeping only a small working set resident.
        self.layer_names = ["transformer_blocks"]

    def _forward_impl(
        self,
        img: Tensor,
        context: Tensor,
        t: Tensor,
        pos: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        img = self.img_in(img)
        t = self.time_embed(temb(t, self.tdim, device=img.device, dtype=img.dtype))
        tvec = self.time_mod_proj(F.gelu(t, approximate="tanh"))

        # A single or same-prompt batch has no padding, so attention runs maskless
        # (native-GQA flash). A ragged batch builds varlen metadata from the
        # key mask and takes the FA varlen path instead.
        txt_key = txt_meta = joint_key = joint_meta = None
        if mask is not None and not bool(mask.all()):
            txt_key = mask[:, : context.shape[1]]
            txt_meta = build_varlen_mask_meta(txt_key)
            joint_key = mask
            joint_meta = build_varlen_mask_meta(mask)

        context = self.text_fusion(context, key_mask=txt_key, mask_meta=txt_meta)
        context = self.txt_in(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = torch.cat((context, img), dim=1)
        freqs = self.posemb(pos)

        # Under SP the image tokens are sharded across ranks while the text prefix
        # stays replicated; keep the leading txtlen tokens out of the all-to-all.
        num_replicated_prefix = txtlen if get_sp_world_size() > 1 else 0
        for block in self.transformer_blocks:
            combined = block(
                combined,
                tvec,
                freqs,
                joint_key,
                joint_meta,
                num_replicated_prefix=num_replicated_prefix,
            )

        final = self.final_layer(combined, t)
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


EntryClass = [Krea2Transformer2DModel]

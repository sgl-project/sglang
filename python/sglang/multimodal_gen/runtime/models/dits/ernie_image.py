# Copyright 2026 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

from sglang.kernels.ops.activation.activation import (
    gelu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion.bitexact_gate import (
    BitExactFusionGate,
    flashinfer_rmsnorm_diagnostic_hint,
    tensors_equal,
)
from sglang.kernels.ops.diffusion.residual_gate_add import residual_gate_add
from sglang.kernels.ops.diffusion.triton.rmsnorm_scale_shift_bitexact import (
    can_use_fused_rmsnorm_scale_shift,
    can_use_fused_scale_residual_rmsnorm_scale_shift,
    fused_rmsnorm_scale_shift_bitexact,
    fused_scale_residual_rmsnorm_scale_shift_bitexact,
)
from sglang.kernels.ops.diffusion.triton.rope_rotate_half_bitexact import (
    can_use_fused_rope_rotate_half,
    fused_rope_rotate_half_bitexact,
)
from sglang.multimodal_gen.configs.models.dits.ernie_image import (
    ErnieImageDitConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_layer
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import (
    USPAttention,
    build_varlen_mask_meta,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    apply_qk_norm,
    apply_qk_norm_rope,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


_ERNIE_NORM = BitExactFusionGate("ERNIE fused-norm")
_ERNIE_GATED_NORM = BitExactFusionGate("ERNIE fused gated-norm")
_ERNIE_ROPE = BitExactFusionGate("ERNIE fused RoPE")
_ERNIE_QKNORM_ROPE = BitExactFusionGate("ERNIE fused QKNorm+RoPE")
_ERNIE_GEGLU = BitExactFusionGate("ERNIE fused GELU-mul")


def _eager_norm_scale_shift(
    norm: RMSNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return norm(x) * (1 + scale) + shift


def _ernie_norm_scale_shift(
    norm: RMSNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """Single-kernel ``norm(x) * (1 + scale) + shift``, bit-exact vs eager.

    The Triton kernel replicates the flashinfer CuTe rmsnorm reduction order
    and every aten bf16 rounding boundary.  Because bit-exactness depends on
    which rmsnorm implementation ``RMSNorm.forward_cuda`` dispatches to, the
    first call verifies ``torch.equal`` against the eager chain and disables
    the fast path permanently on any mismatch.
    """
    verified = _ERNIE_NORM.verified
    if (
        not _ERNIE_NORM.disabled
        and norm.variance_size_override is None
        and can_use_fused_rmsnorm_scale_shift(x, norm.weight, scale, shift)
        and (verified or _ERNIE_NORM.can_attempt_once())
    ):
        try:
            out = fused_rmsnorm_scale_shift_bitexact(
                x, norm.weight, scale, shift, norm.variance_epsilon
            )
        except Exception as exc:
            _ERNIE_NORM.on_exception(exc, logger=logger)
        else:
            if verified:
                return out
            return _ERNIE_NORM.accept_or_fallback(
                out,
                _eager_norm_scale_shift(norm, x, scale, shift),
                logger=logger,
                mismatch_msg=(
                    "ERNIE fused-norm fast path is not bit-exact against this "
                    "platform's rmsnorm dispatch; falling back to eager"
                ),
                diagnostic_hint=flashinfer_rmsnorm_diagnostic_hint,
            )

    return _eager_norm_scale_shift(norm, x, scale, shift)


def _ernie_gated_norm_scale_shift(
    norm: RMSNorm,
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``res = residual + gate * update`` then the fused norm/scale/shift.

    Returns ``(modulated, res)``.  Single kernel, bit-exact vs the eager pair
    (and the ``residual_gate_add_cuda`` fast path) + norm chain; first call
    self-verifies like :func:`_ernie_norm_scale_shift`.
    """
    verified = _ERNIE_GATED_NORM.verified
    if (
        not _ERNIE_GATED_NORM.disabled
        and norm.variance_size_override is None
        and can_use_fused_scale_residual_rmsnorm_scale_shift(
            residual, update, gate, norm.weight, scale, shift
        )
        and (verified or _ERNIE_GATED_NORM.can_attempt_once())
    ):
        try:
            out, res = fused_scale_residual_rmsnorm_scale_shift_bitexact(
                residual,
                update,
                gate,
                norm.weight,
                scale,
                shift,
                norm.variance_epsilon,
            )
        except Exception as exc:
            _ERNIE_GATED_NORM.on_exception(exc, logger=logger)
        else:
            if verified:
                return out, res
            res_ref = residual + gate * update
            ref = _eager_norm_scale_shift(norm, res_ref, scale, shift)
            return _ERNIE_GATED_NORM.accept_or_fallback(
                (out, res),
                (ref, res_ref),
                equal=tensors_equal,
                logger=logger,
                mismatch_msg=(
                    "ERNIE fused gated-norm fast path is not bit-exact against "
                    "this platform's rmsnorm dispatch; falling back to eager"
                ),
                diagnostic_hint=flashinfer_rmsnorm_diagnostic_hint,
            )

    res = residual_gate_add(residual, update, gate)
    return _eager_norm_scale_shift(norm, res, scale, shift), res


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # codespell:ignore nd
    return out.float()


class EmbedND3(nn.Module):
    """3D rotary positional embedding for (temporal/batch_idx, height, width)."""

    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)],
            dim=-1,
        )
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)


class ErnieImageSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V projections and QK LayerNorm.

    Module name hierarchy matches diffusers Attention naming convention:
      self_attention.to_q, self_attention.to_k, self_attention.to_v,
      self_attention.to_out.0, self_attention.norm_q, self_attention.norm_k.

    Supports tensor parallelism: Q/K/V projections use ColumnParallelLinear
    (output dim sharded by heads), output projection uses RowParallelLinear
    (input dim sharded, all-reduce after matmul).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        tp_size = get_tp_world_size()
        self.num_local_heads = num_heads // tp_size
        assert (
            num_heads % tp_size == 0
        ), f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"

        self.to_q = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    hidden_size,
                    hidden_size,
                    bias=False,
                    input_is_parallel=True,
                    prefix=f"{prefix}.to_out.0",
                ),
            ]
        )

        self.qk_layernorm = qk_layernorm
        if qk_layernorm:
            self.norm_q = RMSNorm(head_dim, eps=eps)
            self.norm_k = RMSNorm(head_dim, eps=eps)

        # The joint [image, text] stream is fully replicated, so the ulysses
        # all-to-all would wrongly treat it as sharded and duplicate it. Skip
        # SP until the stream is sharded (sp_shard + num_replicated_suffix).
        self.attn = USPAttention(
            num_heads=self.num_local_heads,
            head_size=head_dim,
            prefix=f"{prefix}.attn",
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cache: torch.Tensor,
        rope_positions: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
    ) -> torch.Tensor:
        B, S, H = x.shape

        q, _ = self.to_q(x)
        k, _ = self.to_k(x)
        v, _ = self.to_v(x)

        q = q.view(B, S, self.num_local_heads, self.head_dim)
        k = k.view(B, S, self.num_local_heads, self.head_dim)
        v = v.view(B, S, self.num_local_heads, self.head_dim)

        if self.qk_layernorm:
            q, k = _ernie_qknorm_rope(
                q,
                k,
                self.norm_q,
                self.norm_k,
                self.head_dim,
                rope_cos,
                rope_sin,
                rope_cache,
                rope_positions,
            )
        else:
            q = _ernie_rope(q, rope_cos, rope_sin)
            k = _ernie_rope(k, rope_cos, rope_sin)

        attn_out = self.attn(
            q, k, v, attn_mask=attn_mask, attn_mask_meta=attn_mask_meta
        )
        attn_out = attn_out.reshape(B, S, self.num_local_heads * self.head_dim)
        out, _ = self.to_out[0](attn_out)
        return out


class ErnieImageMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [ffn_hidden_size, ffn_hidden_size],
            bias=False,
            gather_output=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.linear_fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.linear_fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = _ernie_geglu(gate_up)
        x, _ = self.linear_fc2(x)
        return x


class ErnieImageSharedAdaLNBlock(nn.Module):
    """Single-stream transformer block with externally-computed Shared AdaLN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageSelfAttention(
            hidden_size,
            num_heads,
            head_dim,
            eps,
            qk_layernorm,
            prefix=f"{prefix}.self_attention",
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageMLP(hidden_size, ffn_hidden_size, prefix=f"{prefix}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cache: torch.Tensor,
        rope_positions: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
    ) -> torch.Tensor:
        residual = x
        x = _ernie_norm_scale_shift(self.adaLN_sa_ln, x, scale_msa, shift_msa)
        attn_out = self.self_attention(
            x,
            rope_cos,
            rope_sin,
            rope_cache,
            rope_positions,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
        )
        x, residual = _ernie_gated_norm_scale_shift(
            self.adaLN_mlp_ln, residual, attn_out, gate_msa, scale_mlp, shift_mlp
        )
        x = residual_gate_add(residual, self.mlp(x), gate_mlp)

        return x


def _precompute_rope_cos_sin(
    freqs: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of the rotary embedding, computed once per forward.

    ``freqs`` is the ``(S, B, 1, rot_dim)`` output of :class:`EmbedND3`; the
    eager chain recomputed ``torch.cos(freqs).to(dtype)`` per layer per
    projection.  Returns bit-identical ``(B * S, rot_dim)`` rows.
    """
    freqs = freqs.permute(1, 0, 2, 3)
    cos_ = torch.cos(freqs).to(dtype)
    sin_ = torch.sin(freqs).to(dtype)
    rot_dim = freqs.shape[-1]
    return cos_.reshape(-1, rot_dim), sin_.reshape(-1, rot_dim)


def _apply_rotary_bshd_eager(
    x: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor
) -> torch.Tensor:
    """Reference rotate-half chain on precomputed cos/sin (bit-exact vs the
    original per-layer version, which materialized the same cos/sin)."""
    batch, seq_len = x.shape[0], x.shape[1]
    rot_dim = cos_.shape[-1]
    cos_b = cos_.view(batch, seq_len, 1, rot_dim)
    sin_b = sin_.view(batch, seq_len, 1, rot_dim)
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]

    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)

    x_rot = x_rot * cos_b + x_rotated * sin_b
    return torch.cat((x_rot, x_pass), dim=-1)


def _ernie_rope(
    x: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor
) -> torch.Tensor:
    """Single-kernel rotate-half RoPE, bit-exact vs the eager chain.

    Pure elementwise math, so the Triton kernel reproduces every aten bf16
    rounding boundary exactly; the first call still verifies ``torch.equal``
    against the eager chain and disables the fast path on any mismatch.
    """
    verified = _ERNIE_ROPE.verified
    if (
        not _ERNIE_ROPE.disabled
        and can_use_fused_rope_rotate_half(x, cos_, sin_)
        and (verified or _ERNIE_ROPE.can_attempt_once())
    ):
        try:
            out = fused_rope_rotate_half_bitexact(x, cos_, sin_)
        except Exception as exc:
            _ERNIE_ROPE.on_exception(exc, logger=logger)
        else:
            if verified:
                return out
            return _ERNIE_ROPE.accept_or_fallback(
                out,
                _apply_rotary_bshd_eager(x, cos_, sin_),
                logger=logger,
                mismatch_msg=(
                    "ERNIE fused RoPE fast path is not bit-exact on this "
                    "platform; falling back to eager"
                ),
            )
    return _apply_rotary_bshd_eager(x, cos_, sin_)


def _ernie_qknorm_rope_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k = apply_qk_norm(q, k, q_norm, k_norm, head_dim)
    return _ernie_rope(q, rope_cos, rope_sin), _ernie_rope(k, rope_cos, rope_sin)


def _ernie_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cache: torch.Tensor,
    rope_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ERNIE's QK RMSNorm and rotate-half RoPE without changing bits."""
    verified = _ERNIE_QKNORM_ROPE.verified
    if not _ERNIE_QKNORM_ROPE.disabled and (
        verified or _ERNIE_QKNORM_ROPE.can_attempt_once()
    ):
        q_input = q.clone() if not verified else q
        k_input = k.clone() if not verified else k
        try:
            out = apply_qk_norm_rope(
                q=q,
                k=k,
                q_norm=q_norm,
                k_norm=k_norm,
                head_dim=head_dim,
                cos_sin_cache=rope_cache,
                is_neox=True,
                positions=rope_positions,
                round_norm_before_rope=True,
                cache_has_full_width=True,
            )
        except Exception as exc:
            _ERNIE_QKNORM_ROPE.on_exception(exc, logger=logger)
            return _ernie_qknorm_rope_reference(
                q_input,
                k_input,
                q_norm,
                k_norm,
                head_dim,
                rope_cos,
                rope_sin,
            )
        else:
            if verified:
                return out
            ref = _ernie_qknorm_rope_reference(
                q_input,
                k_input,
                q_norm,
                k_norm,
                head_dim,
                rope_cos,
                rope_sin,
            )
            return _ERNIE_QKNORM_ROPE.accept_or_fallback(
                out,
                ref,
                equal=tensors_equal,
                logger=logger,
                mismatch_msg=(
                    "ERNIE fused QKNorm+RoPE fast path is not bit-exact on "
                    "this platform; falling back to split kernels"
                ),
            )

    return _ernie_qknorm_rope_reference(
        q, k, q_norm, k_norm, head_dim, rope_cos, rope_sin
    )


def _eager_geglu(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return up * F.gelu(gate)


def _ernie_geglu(gate_up: torch.Tensor) -> torch.Tensor:
    """``up * gelu(gate)`` in one kernel, bit-exact vs the eager pair.

    Uses the activation kernel's rounding variant, which rounds the erf-GELU
    to bf16 before the multiply exactly like the eager two-step; first call
    self-verifies like :func:`_ernie_rope`.
    """
    verified = _ERNIE_GEGLU.verified
    if (
        not _ERNIE_GEGLU.disabled
        and gate_up.dtype in (torch.bfloat16, torch.float16)
        and gate_up.is_cuda
        and gate_up.is_contiguous()
        and gate_up.shape[-1] % 2 == 0
        and (verified or _ERNIE_GEGLU.can_attempt_once())
    ):
        try:
            out = gelu_and_mul_with_activation_rounding(gate_up)
        except Exception as exc:
            _ERNIE_GEGLU.on_exception(exc, logger=logger)
        else:
            if verified:
                return out
            return _ERNIE_GEGLU.accept_or_fallback(
                out,
                _eager_geglu(gate_up),
                logger=logger,
                mismatch_msg=(
                    "ERNIE fused GELU-mul fast path is not bit-exact on this "
                    "platform; falling back to eager"
                ),
            )
    return _eager_geglu(gate_up)


class ErnieImageTransformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """ErnieImage DiT: Single-stream transformer with Shared AdaLN."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ErnieImageSharedAdaLNBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    _fsdp_shard_conditions = [is_layer]
    _compile_conditions = []
    param_names_mapping = ErnieImageDitConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: ErnieImageDitConfig,
        hf_config: dict[str, Any],
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(config=config, hf_config=hf_config)

        arch = self.config
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.out_channels
        self.head_dim = arch.attention_head_dim
        self.num_layers = arch.num_layers
        self.patch_size = arch.patch_size
        self.out_channels = arch.out_channels
        self.inner_dim = self.hidden_size

        self.x_embedder = nn.ModuleDict(
            {
                "proj": nn.Conv2d(
                    arch.in_channels,
                    self.inner_dim,
                    kernel_size=arch.patch_size,
                    stride=arch.patch_size,
                    bias=True,
                ),
            }
        )

        if arch.text_in_dim != self.inner_dim:
            self.text_proj = nn.Linear(arch.text_in_dim, self.inner_dim, bias=False)
        else:
            self.text_proj = None

        self.time_proj = Timesteps(
            self.inner_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=self.inner_dim,
            time_embed_dim=self.inner_dim,
        )

        self.pos_embed = EmbedND3(
            dim=self.head_dim,
            theta=arch.rope_theta,
            axes_dim=arch.rope_axes_dim,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, 6 * self.inner_dim),
        )

        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(
                    hidden_size=self.inner_dim,
                    num_heads=self.num_attention_heads,
                    head_dim=self.head_dim,
                    ffn_hidden_size=arch.ffn_hidden_size,
                    eps=arch.eps,
                    qk_layernorm=arch.qk_layernorm,
                    prefix=f"layers.{i}",
                )
                for i in range(self.num_layers)
            ]
        )

        self.final_norm = nn.ModuleDict(
            {
                "norm": nn.LayerNorm(
                    self.inner_dim, elementwise_affine=False, eps=arch.eps
                ),
                "linear": nn.Linear(self.inner_dim, self.inner_dim * 2),
            }
        )

        self.final_linear = ColumnParallelLinear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
            gather_output=True,
            prefix="final_linear",
        )

        self.layer_names = ["layers"]

        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, C, H, W] latent images (patchified, 128 channels)
            encoder_hidden_states: [B, T, text_dim] or list of text embeddings
            timestep: [B] timestep values
        Returns:
            output: [B, C, H, W] predicted noise / denoised output
        """
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p
        N_img = Hp * Wp

        img_tokens = self.x_embedder["proj"](hidden_states)  # [B, D, Hp, Wp]
        img_tokens = img_tokens.reshape(B, self.inner_dim, N_img).transpose(
            1, 2
        )  # [B, N_img, D]

        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[0]
        text_tokens = encoder_hidden_states  # [B, T, text_dim]
        if self.text_proj is not None and text_tokens.numel() > 0:
            text_tokens = self.text_proj(text_tokens)
        Tmax = text_tokens.shape[1]

        x = torch.cat([img_tokens, text_tokens], dim=1)  # [B, S, D]

        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)

        image_ids = torch.cat(
            [
                torch.full((B, N_img, 1), Tmax, device=device, dtype=torch.float32),
                grid_yx.view(1, N_img, 2).expand(B, -1, -1),
            ],
            dim=-1,
        )

        if Tmax > 0:
            text_ids = torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32)
                    .view(1, Tmax, 1)
                    .expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
        else:
            text_ids = torch.zeros((B, 0, 3), device=device)

        all_ids = torch.cat([image_ids, text_ids], dim=1)
        rotary_pos_emb = self.pos_embed(all_ids)
        rope_cos, rope_sin = _precompute_rope_cos_sin(rotary_pos_emb, dtype)
        rope_cache = torch.cat((rope_cos, rope_sin), dim=-1).contiguous()
        rope_positions = torch.arange(
            rope_cache.shape[0], device=device, dtype=torch.long
        )

        attn_mask = attn_mask_meta = None
        if encoder_hidden_states_mask is not None:
            image_mask = torch.ones((B, N_img), dtype=torch.bool, device=device)
            attn_mask = torch.cat(
                [
                    image_mask,
                    encoder_hidden_states_mask.to(device=device, dtype=torch.bool),
                ],
                dim=1,
            )
            attn_mask_meta = build_varlen_mask_meta(attn_mask)

        t_emb = self.time_proj(timestep.to(dtype))
        c = self.time_embedding(t_emb.to(dtype=dtype))

        mod_params = self.adaLN_modulation(c)
        # .contiguous() is a bit-exact copy of the tiny (B, 1, D) modulation
        # params; the fused residual-gate kernel requires dense inputs.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            t.unsqueeze(1).contiguous() for t in mod_params.chunk(6, dim=-1)
        )

        for layer in self.layers:
            x = layer(
                x,
                rope_cos,
                rope_sin,
                rope_cache,
                rope_positions,
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                attn_mask=attn_mask,
                attn_mask_meta=attn_mask_meta,
            )

        scale, shift = self.final_norm["linear"](c).chunk(2, dim=-1)
        x = self.final_norm["norm"](x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        patches, _ = self.final_linear(x[:, :N_img, :])

        output = patches.view(B, Hp, Wp, p, p, self.out_channels)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()
        output = output.view(B, self.out_channels, H, W)

        return output


EntryClass = ErnieImageTransformer2DModel

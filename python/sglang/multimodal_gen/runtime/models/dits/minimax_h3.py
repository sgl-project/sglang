# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed-token DiT.

Native SGLang implementation of the MiniMax H3 audio-video DiT. The forward
contract accepts packed inference keyword arguments and returns packed logits.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding_,
)
from sglang.kernels.ops.diffusion.qknorm_rope import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
)
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

_ARCH_DEFAULTS = MiniMaxH3DiTArchConfig()
_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32

_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(_MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _required_kwarg(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


# The exhaustive keyword contract of MiniMaxH3DiTModel.forward. Anything not
# listed here is rejected with a TypeError before any tensor work starts.
_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)


def _ulysses_ctx() -> tuple[int, int]:
    """(world_size, rank) of the Ulysses sequence-parallel group.

    Returns (1, 0) when model parallelism is not initialized (unit tests /
    single-process debug paths init tp=1 sp=1 which also yields ws=1).
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_rank,
        get_ulysses_parallel_world_size,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return 1, 0
    return get_ulysses_parallel_world_size(), get_ulysses_parallel_rank()


def _ring_world_size() -> int:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ring_parallel_world_size,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return 1
    return get_ring_parallel_world_size()


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            "qkv weight has incompatible output dim for grouped checkpoint layout: "
            f"got {tuple(weight.shape)}, expected first dim {expected_out}."
        )

    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        [heads_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ],
        dim=0,
    )


def _copy_grouped_qkv_tp_shard(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    num_query_groups: int,
    head_dim: int,
    tp_rank: int,
    tp_size: int,
) -> bool:
    """Copy a dense MHA checkpoint directly into its TP-local Q/K/V rows."""
    if (
        tp_size <= 0
        or not 0 <= tp_rank < tp_size
        or num_query_groups % tp_size
        or getattr(param, "output_dim", None) != 0
        or getattr(param, "is_sharded_weight", False)
        or getattr(param, "packed_dim", None) is not None
        or param.dtype != _BF16_DTYPE
        or loaded_weight.dtype != _BF16_DTYPE
        or not param.is_contiguous()
        or not loaded_weight.is_contiguous()
    ):
        return False

    expected_rows = num_query_groups * 3 * head_dim
    local_groups = num_query_groups // tp_size
    rest_shape = loaded_weight.shape[1:]
    if loaded_weight.shape[0] != expected_rows or tuple(param.shape) != (
        3 * local_groups * head_dim,
        *rest_shape,
    ):
        return False

    grouped = loaded_weight.view(num_query_groups, 3, head_dim, *rest_shape)
    grouped = grouped.narrow(0, tp_rank * local_groups, local_groups)
    target = param.data.view(3, local_groups, head_dim, *rest_shape)
    for index in range(3):
        target[index].copy_(grouped[:, index])
    return True


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16_DTYPE) -> nn.RMSNorm:
    # RMSNorm uses fp32 accumulation with bf16 inputs and outputs.
    # torch.nn.RMSNorm upcasts reduced-precision inputs for the variance
    # reduction, matching that accumulation semantic.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply indexed affine modulation, reusing disposable CUDA BF16 input."""
    # Apply per-index affine modulation: x * (1 + scale[idx]) + shift[idx].
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and shift.dtype == _BF16_DTYPE
        and scale.dtype == _BF16_DTYPE
        and x.is_contiguous()
    ):
        return indexed_scale_shift_bf16_(x, shift, scale, indices)
    return (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply indexed gated residual, reusing disposable CUDA BF16 input."""
    # Apply the per-index gated residual: x + gate[idx] * other.
    if (
        x.is_cuda
        and x.dtype == _BF16_DTYPE
        and dtype == _BF16_DTYPE
        and gate.dtype == _BF16_DTYPE
        and other.dtype == _BF16_DTYPE
        and x.is_contiguous()
        and other.is_contiguous()
    ):
        return indexed_gate_bf16_(x, gate, other, indices)
    return (x + gate.index_select(0, indices) * other).to(dtype)


def _silu_mul(hidden: torch.Tensor, *, reuse_input: bool) -> torch.Tensor:
    if (
        reuse_input
        and hidden.is_cuda
        and hidden.dtype == _BF16_DTYPE
        and hidden.is_contiguous()
        and hidden.shape[-1] % 16 == 0
    ):
        return silu_and_mul_with_activation_rounding_(hidden)
    gate, up = hidden.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


def _apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        q.is_cuda
        and q.dtype == _BF16_DTYPE
        and q.dtype == k.dtype == q_norm.weight.dtype == k_norm.weight.dtype
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == head_dim
        and q_norm.eps == k_norm.eps
        and not torch.compiler.is_compiling()
    ):
        fused_inplace_qknorm(
            q,
            k,
            q_norm.weight,
            k_norm.weight,
            eps=q_norm.eps,
            head_dim=head_dim,
        )
        return q, k
    return q_norm(q), k_norm(k)


class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


def _rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Build the activation-dtype cos|sin cache for fused Q/K RoPE."""
    half = freqs.shape[-1] // 2
    return (
        torch.cat(
            (torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])),
            dim=-1,
        )
        .to(dtype=dtype, copy=False)
        .contiguous()
    )


def _apply_rope_cos_sin(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate the first cached RoPE dims; pass the remaining head dims through."""
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((x_rot, x_pass), dim=-1)


def _apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q.is_cuda:
        half = cos_sin_cache.shape[-1] // 2
        cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
        return (
            _apply_rope_cos_sin(q, cos, sin),
            _apply_rope_cos_sin(k, cos, sin),
        )

    from sgl_kernel import rotary_embedding as apply_sgl_kernel_rotary_embedding

    apply_sgl_kernel_rotary_embedding(
        positions,
        q.view(q.shape[0], -1),
        k.view(k.shape[0], -1),
        q.shape[-1],
        cos_sin_cache,
        True,
    )
    return q, k


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_in",
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.proj_out",
        )
        self.register_buffer("_frequency_cache", None, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [M] -> [M, time_embed_dim] fp32.

        The sinusoidal embedding stays fp32 throughout and concatenates cosine
        values before sine values.
        """
        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            # Construct this on the execution device once so the values keep
            # the established CUDA numerics without repeating arange/exp on
            # every denoise step.
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=_FP32_DTYPE, device=t.device)
                / half
            )
            self._frequency_cache = freqs
        args = t.to(_FP32_DTYPE)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        hidden, _ = self.proj_in(t_freq)
        hidden = nn.functional.silu(hidden)
        out, _ = self.proj_out(hidden)
        return out


def _minimax_h3_attention_core_impl(
    attention: MiniMaxH3Attention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    ulysses_active: bool,
) -> torch.Tensor:
    """Dynamic varlen attention and Ulysses collectives.

    This is the narrow BCG break point: projections, normalization, RoPE,
    residuals, and MLPs remain captured while the dynamic packed attention
    kernel and sequence-parallel collectives execute eagerly.
    """

    if ulysses_active:
        from sglang.multimodal_gen.runtime.layers.usp import (
            _usp_input_all_to_all_packed_qkv,
            _usp_output_all_to_all,
        )

        q, k, v = _usp_input_all_to_all_packed_qkv(q, k, v)

    if attention._attention_impl is None:
        attention._set_attention_backend(
            get_attn_backend(
                attention.head_dim,
                q.dtype,
                attention_requirements=AttentionRequirements(packed_varlen=True),
            )
        )
    out = attention._attention_impl.forward_varlen(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        cu_seqlens_host=cu_seqlens_host,
    )
    if ulysses_active:
        out = _usp_output_all_to_all(out[None], head_dim=2)[0]
    return out


_minimax_h3_attention_core_bcg = eager_on_graph(True)(_minimax_h3_attention_core_impl)


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        bcg_breakpoint: bool = True,
    ) -> None:
        super().__init__()
        self.bcg_breakpoint = bcg_breakpoint
        self.tp_size = get_tp_world_size()
        if arch.num_attention_heads % self.tp_size:
            raise ValueError(
                "MiniMax H3 attention heads must be divisible by TP size: "
                f"{arch.num_attention_heads} % {self.tp_size} != 0"
            )
        self.total_num_heads = arch.num_attention_heads
        self.num_heads = self.total_num_heads // self.tp_size
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.total_num_heads * self.head_dim
        self.local_inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self._attention_impl = None
        # The checkpoint stores one fused qkv tensor. Each logical Q/K/V
        # matrix must be sharded independently; a plain ColumnParallelLinear
        # would instead slice across the concatenated tensor and is incorrect
        # for TP > 1.
        self.qkv_proj = MergedColumnParallelLinear(
            arch.hidden_size,
            [self.inner_dim] * 3,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self._install_qkv_weight_loader(arch)
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        # cache width covers cos/sin for temporal, height, and width frequencies
        rope_dim = 6 * arch.rope_inv_freq_len
        self._use_fused_qknorm_rope = (
            current_platform.is_cuda()
            and can_use_fused_inplace_qknorm_rope(
                arch.attention_head_dim,
                rope_dim,
                True,
                _BF16_DTYPE,
                cache_dtype=_BF16_DTYPE,
                round_norm_before_rope=True,
            )
        )
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def _set_attention_backend(self, backend) -> None:
        impl_cls = backend.get_impl_cls()
        self._attention_impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_heads,
        )

    def _install_qkv_weight_loader(self, arch: MiniMaxH3DiTArchConfig) -> None:
        weight = self.qkv_proj.weight
        base_loader = weight.weight_loader

        def _weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # The grouped checkpoint layout is
            # [num_query_groups, q_per_group + k + v] before splitting.
            # MiniMax H3 uses MHA, so checkpoint rows are per-head [q, k, v],
            # while SGLang stores [q_all, k_all, v_all].
            if _copy_grouped_qkv_tp_shard(
                param,
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                head_dim=arch.attention_head_dim,
                tp_rank=self.qkv_proj.tp_rank,
                tp_size=self.tp_size,
            ):
                return
            reordered = _reorder_grouped_qkv_to_qkv(
                loaded_weight,
                num_query_groups=arch.num_attention_heads,
                heads_per_group=1,
                head_dim=arch.attention_head_dim,
            )
            base_loader(param, reordered)

        if hasattr(weight, "_weight_loader"):
            weight._weight_loader = _weight_loader
        else:
            weight.weight_loader = _weight_loader

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        ulysses_active: bool = False,
    ) -> torch.Tensor:
        """x: [T, hidden] packed thd rows -> [T, hidden].

        Operation order: fused qkv projection -> per-head q/k RMSNorm -> RoPE
        on q/k -> variable-length non-causal flash attention -> output projection.

        With Ulysses sequence parallelism, x holds this rank's row shard;
        qkv/norm/RoPE run locally, an all-to-all trades sequence for heads.
        Each rank attends the full sequence with heads/world_size local heads,
        so cu_seqlens retains global packed-document semantics. The inverse
        all-to-all restores the row shard before the output projection.
        """
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(self.local_inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        if rope_cache is None:
            q, k = _apply_qk_norm(
                q,
                k,
                self.q_norm,
                self.k_norm,
                self.head_dim,
            )
        else:
            cos_sin_cache, positions = rope_cache
            if self._use_fused_qknorm_rope and not torch.compiler.is_compiling():
                fused_inplace_qknorm_rope(
                    q,
                    k,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    cos_sin_cache,
                    positions,
                    is_neox=True,
                    eps=self.q_norm.eps,
                    head_dim=self.head_dim,
                    rope_dim=cos_sin_cache.shape[-1],
                    round_norm_before_rope=True,
                )
            else:
                q, k = _apply_qk_norm(
                    q,
                    k,
                    self.q_norm,
                    self.k_norm,
                    self.head_dim,
                )
                q, k = _apply_rope_qk(q, k, cos_sin_cache, positions)

        attention_core = (
            _minimax_h3_attention_core_bcg
            if self.bcg_breakpoint
            else _minimax_h3_attention_core_impl
        )
        out = attention_core(
            self,
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ulysses_active=ulysses_active,
        )
        out = out.reshape(total, self.num_heads * self.head_dim)
        out, _ = self.out_proj(out)
        return out


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        # As with qkv, gate and up are two independently sharded logical
        # matrices even though the checkpoint stores them fused.
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size] * 2,
            bias=False,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        # Chunk the fused fc1 output as [gate, up], then compute
        # silu(gate) * up.
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        self.reuse_fc1_activation = quant_config is None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.fc1(x)
        hidden = _silu_mul(hidden, reuse_input=self.reuse_fc1_activation)
        out, _ = self.fc2(hidden)
        return out


class MiniMaxH3AdalnProj(nn.Module):
    """SiLU + zero-init linear over unique condition embeddings.

    Per block, three modalities each produce six H-wide vectors:
    [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    The final layer uses one modality and produces two H-wide vectors:
    [M, t_dim] -> [M, 2H] -> chunk(2).
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                "adaln out_features mismatch: "
                f"{out_features} != {expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=False,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
        )

    def project_local(self, adaln_input: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear(adaln_input)
        return x

    def split_output(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """adaln_input: SiLU(t_emb) BF16 -> expand_ratio tensors of [M*modality_num, H]."""
        x = self.project_local(adaln_input)
        if get_tp_world_size() > 1:
            x = tensor_model_parallel_all_gather(x)
        return self.split_output(x)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Standard pre-norm transformer block without AdaLN or RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        # The whole dynamic text-refiner/scatter phase is one BCG break point;
        # do not nest per-attention graph breaks inside it.
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
            bcg_breakpoint=False,
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    arch,
                    quant_config,
                    prefix=f"{prefix}.blocks.{index}",
                )
                for index in range(arch.token_refiner_num_layers)
            ]
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(
            arch,
            quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiniMaxH3MLP(arch, quant_config, prefix=f"{prefix}.mlp")
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            quant_config,
            prefix=f"{prefix}.adaln_proj",
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None = None,
        max_seqlen: int,
        ulysses_active: bool = False,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """x: [T, H]; adaln_input: [M, t_dim]; combined_indices: [T]
        (= inverse_indices * modality_num + token_tags.clamp(min=0)).

        Each block computes AdaLN parameters once, then applies
        norm1 -> scale/shift -> attention -> gated residual, followed by
        norm2 -> scale/shift -> MLP -> gated residual.
        """
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = self.norm1(x)
        h = _modulate_scale_shift(
            h, shift_msa, scale_msa, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.attn(
            h,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ulysses_active=ulysses_active,
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16_DTYPE)

        residual = x
        h = self.norm2(x)
        h = _modulate_scale_shift(
            h, shift_mlp, scale_mlp, combined_indices, dtype=_BF16_DTYPE
        )
        h = self.mlp(h)
        return _modulate_gate(
            residual, gate_mlp, h, combined_indices, dtype=_BF16_DTYPE
        )


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        video_patch_dim = (
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2]
        )
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            quant_config,
            prefix=f"{prefix}.adaln_proj",
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.video_out",
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=False,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.audio_out",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project all rows into TP-local video/audio output shards.

        Apply single-modality shift/scale AdaLN to the final normalized
        activations, cast to fp32, then apply both output heads to all rows.
        The model gathers output columns only after selecting live media rows,
        preserving the GEMM shape while reducing collective payload.
        """
        shift, scale = self.adaln_proj(adaln_input)
        h = self.norm(x)
        h = _modulate_scale_shift(h, shift, scale, inverse_indices, dtype=_BF16_DTYPE)
        # Preserve full precision through both final output projections.
        h = h.to(_FP32_DTYPE)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio


class MiniMaxH3DiTModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = _ARCH_DEFAULTS._fsdp_shard_conditions
    # parameters mix fp32 (patch projections, timestep embedder, and output
    # heads) with bf16 blocks; FSDP must gather in each parameter's own dtype
    _fsdp_mixed_dtype_params = True
    _compile_conditions = _ARCH_DEFAULTS._compile_conditions
    param_names_mapping = _ARCH_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_DEFAULTS.lora_param_names_mapping

    def _can_batch_block_adaln(self) -> bool:
        return (
            get_tp_world_size() > 1
            and not torch.compiler.is_compiling()
            and not envs.SGLANG_CACHE_DIT_ENABLED
            and not hasattr(self, "_sglang_cache_dit_adapter")
            and not is_layerwise_offloaded_module(self)
            and all(type(block) is MiniMaxH3DiTBlock for block in self.blocks)
        )

    def _validate_tp_config(
        self, *, arch: MiniMaxH3DiTArchConfig, tp_size: int
    ) -> None:
        if tp_size <= 0:
            raise ValueError("TP size must be positive.")
        if arch.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if arch.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if arch.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if arch.ffn_hidden_size <= 0:
            raise ValueError("ffn_hidden_size must be positive.")
        for name, value in (
            ("num_attention_heads", arch.num_attention_heads),
            ("hidden_size", arch.hidden_size),
            ("ffn_hidden_size", arch.ffn_hidden_size),
            ("time_embed_hidden_size", arch.time_embed_hidden_size),
            ("adaln_out_features", arch.adaln_out_features),
            ("final_adaln_out_features", arch.final_adaln_out_features),
            ("video_patch_output_dim", arch.latents_dim * math.prod(arch.patch_size)),
            ("audio_patch_output_dim", arch.audio_latents_dim),
        ):
            if value % tp_size:
                raise ValueError(
                    f"MiniMax H3 {name}={value} must be divisible by "
                    f"TP size {tp_size}."
                )

    @staticmethod
    def _validate_sequence_parallel_config(
        *,
        arch: MiniMaxH3DiTArchConfig,
        tp_size: int,
        ulysses_size: int,
        ring_size: int,
    ) -> None:
        if ulysses_size <= 0:
            raise ValueError("MiniMax H3 Ulysses size must be positive.")
        if ring_size != 1:
            raise NotImplementedError(
                "MiniMax H3 packed multi-segment attention does not support "
                "Ring or mixed USP. Set --ring-degree 1 and use Ulysses "
                "sequence parallelism."
            )
        local_heads = arch.num_attention_heads // tp_size
        if local_heads % ulysses_size:
            raise ValueError(
                f"MiniMax H3 TP-local heads {local_heads} must be divisible by "
                f"Ulysses size {ulysses_size} (total heads="
                f"{arch.num_attention_heads}, TP={tp_size})."
            )
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % ulysses_size:
            raise ValueError(
                "MiniMax H3 packed sequence alignment "
                f"{MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT} must be divisible by "
                f"Ulysses size {ulysses_size}. Choose a Ulysses size that "
                "divides both the TP-local attention heads and the packed "
                "sequence alignment."
            )

    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        self.arch = arch
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.latents_dim
        tp_size = get_tp_world_size()
        ulysses_size, _ = _ulysses_ctx()
        self._validate_tp_config(arch=arch, tp_size=tp_size)
        self._validate_sequence_parallel_config(
            arch=arch,
            tp_size=tp_size,
            ulysses_size=ulysses_size,
            ring_size=_ring_world_size(),
        )

        self.video_patch_proj = ColumnParallelLinear(
            arch.latents_dim
            * arch.patch_size[0]
            * arch.patch_size[1]
            * arch.patch_size[2],
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="video_patch_proj",
        )
        self.audio_patch_proj = ColumnParallelLinear(
            arch.audio_latents_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix="audio_patch_proj",
        )
        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix="condition_proj",
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(
            arch,
            prefix="time_embedder",
        )
        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(
            arch,
            quant_config,
            prefix="token_refiner",
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    arch,
                    quant_config,
                    prefix=f"blocks.{index}",
                )
                for index in range(arch.num_layers)
            ]
        )
        self.layer_names = ["blocks"]
        self.final_layer = MiniMaxH3FinalLayer(
            arch,
            quant_config,
            prefix="final_layer",
        )
        self._resolved_attention_backend: AttentionBackendEnum | None = None
        self._mark_missing_params_required()

    def _resolve_attention_backend_once(self) -> None:
        if self._resolved_attention_backend is not None:
            return
        backend = get_attn_backend(
            self.arch.attention_head_dim,
            _BF16_DTYPE,
            attention_requirements=AttentionRequirements(packed_varlen=True),
        )
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module._set_attention_backend(backend)
        self._resolved_attention_backend = backend.get_enum()

    def _mark_missing_params_required(self) -> None:
        for _, param in self.named_parameters():
            param.missing_param_init = "error"

    def post_load_weights(self) -> None:
        for name in _MINIMAX_H3_FP32_PARAM_NAMES_IN_MODEL_ORDER:
            param = self.get_parameter(name)
            if param.dtype != _FP32_DTYPE:
                raise ValueError(
                    f"{name} must stay fp32 after load, got {param.dtype}."
                )
        # assign=True loading may re-register this persistent buffer as a parameter
        rope_inv_freq = self.rope.inv_freq
        if rope_inv_freq.dtype != _FP32_DTYPE:
            raise ValueError(
                f"rope.inv_freq must stay fp32 after load, got {rope_inv_freq.dtype}."
            )

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if isinstance(pos_info, dict):
            ids = pos_info.get("position_ids")
        else:
            ids = getattr(pos_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _psp_field(psp: Any, key: str, field: str) -> Any:
        if isinstance(psp, dict):
            value = psp.get(field)
        else:
            value = getattr(psp, field, None)
        if value is None:
            raise ValueError(f"{key}.{field} is required")
        return value

    @staticmethod
    def _psp_optional_field(psp: Any, field: str) -> Any:
        if isinstance(psp, dict):
            return psp.get(field)
        return getattr(psp, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Project and refine request-static text conditioning once."""
        text_len = int(refiner_cu_seqlens[1].item())
        if text_len <= 0 or text_len > int(prompt_embeds.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(prompt_embeds.shape[0])}], got {text_len}"
            )
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=_BF16_DTYPE)
        true_refiner_cu = torch.stack(
            (
                refiner_cu_seqlens[0],
                refiner_cu_seqlens[1],
                refiner_cu_seqlens[1],
            )
        )
        text_embed, _ = self.condition_proj(text_rows)
        return self.token_refiner(
            text_embed,
            cu_seqlens=true_refiner_cu,
            cu_seqlens_host=(0, text_len, text_len),
            max_seqlen=text_len,
        )

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build request-static RoPE inputs for this Ulysses rank."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        seq_len = int(img_position_ids.shape[1])
        sp_ws, sp_rank = _ulysses_ctx()
        if seq_len % sp_ws:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by ulysses world size {sp_ws}"
            )
        local_seq_len = seq_len // sp_ws
        row_start = sp_rank * local_seq_len
        rope_freqs = self.rope(
            img_position_ids[:, row_start : row_start + local_seq_len]
        ).to(device)
        return (
            _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
            torch.arange(
                local_seq_len,
                device=device,
                dtype=torch.long,
            ),
        )

    @eager_on_graph(True)
    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refiner_max_seqlen: int,
        row_start: int,
        row_stop: int,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None = None,
        local_embedding_layout: dict[str, torch.Tensor | int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build embeddings for one contiguous block-stack row shard.

        Returns (decoder_input [S_local, H] bf16, t_emb [M, t_dim] fp32).
        """
        # BCG pads the prompt tensor only to stabilize its input signature.
        # Raw-input callers recover the live length from refiner metadata;
        # request-static refined inputs carry it as a host integer and avoid a
        # per-step device scalar read. Running the refiner at the bucketed M
        # dimension changes GEMM selection and is not bitwise equivalent.
        if refined_prompt_embeds_length is None:
            text_len = int(refiner_cu_seqlens[1].item())
        elif torch.is_tensor(refined_prompt_embeds_length):
            # BCG turns this request-varying host constant into a scalar input
            # so different live lengths can replay one padded-text signature.
            # _embed is an eager graph break, so this value is read outside
            # captured CUDA graphs.
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError(
                "refiner cu_seqlens live text length must be in "
                f"[1, {int(text_embeddings_selected.shape[0])}], got {text_len}"
            )
        text_pos = text_pos[:text_len]
        if refined_prompt_embeds_length is not None:
            text_embed = text_embeddings_selected[:text_len].to(
                device=device, dtype=_BF16_DTYPE
            )
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    "refined prompt embeddings must have hidden width "
                    f"{self.hidden_size}, got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                text_embeddings_selected,
                refiner_cu_seqlens,
                device=device,
            )

        local_seq_len = row_stop - row_start
        trusted_layout = local_embedding_layout is not None
        if trusted_layout:
            used_len = text_len + int(img_pos.numel()) + int(audio_pos.numel())
            local_live_rows = min(max(used_len - row_start, 0), local_seq_len)
            embeddings = torch.empty(
                (local_seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE
            )
            if local_live_rows < local_seq_len:
                embeddings[local_live_rows:].zero_()
        else:
            # Direct model callers do not provide the serving-time partition
            # contract. Preserve their historical zero-fill/add semantics for
            # sparse or overlapping row maps.
            embeddings = torch.zeros(
                (local_seq_len, self.hidden_size), device=device, dtype=_BF16_DTYPE
            )

        if local_embedding_layout is None:
            text_source_ids = torch.nonzero(
                (text_pos >= row_start) & (text_pos < row_stop),
                as_tuple=False,
            ).view(-1)
            text_row_ids = text_pos.index_select(0, text_source_ids) - row_start
            img_global_ids = img_pos.index_select(
                0,
                torch.nonzero(
                    (img_pos >= row_start) & (img_pos < row_stop),
                    as_tuple=False,
                ).view(-1),
            )
            img_row_ids = img_global_ids - row_start
            audio_global_ids = audio_pos.index_select(
                0,
                torch.nonzero(
                    (audio_pos >= row_start) & (audio_pos < row_stop),
                    as_tuple=False,
                ).view(-1),
            )
            audio_row_ids = audio_global_ids - row_start
        else:
            text_source_start = int(local_embedding_layout["text_source_start"])
            text_source_stop = int(local_embedding_layout["text_source_stop"])
            img_global_ids = local_embedding_layout["img_global_ids"]
            img_row_ids = local_embedding_layout["img_row_ids"]
            audio_global_ids = local_embedding_layout["audio_global_ids"]
            audio_row_ids = local_embedding_layout["audio_row_ids"]

        write_rows = embeddings.index_copy_ if trusted_layout else embeddings.index_add_
        if trusted_layout:
            text_rows = text_source_stop - text_source_start
            if text_rows:
                embeddings[:text_rows].copy_(
                    text_embed[text_source_start:text_source_stop]
                )
        elif text_row_ids.numel():
            write_rows(
                0,
                text_row_ids,
                text_embed.index_select(0, text_source_ids).to(_BF16_DTYPE),
            )

        # latent embedders stay fp32; only rows owned by this SP rank are
        # projected, then cast during scattering into the bf16 sequence
        if img_row_ids.numel():
            x_rows = (
                x.view(-1, x.shape[-1]).index_select(0, img_global_ids).to(_FP32_DTYPE)
            )
            video_embed, _ = self.video_patch_proj(x_rows)
            write_rows(
                0,
                img_row_ids,
                video_embed.to(_BF16_DTYPE),
            )

        if audio_row_ids.numel():
            audio_rows = (
                audio_x.view(-1, audio_x.shape[-1])
                .index_select(0, audio_global_ids)
                .to(_FP32_DTYPE)
            )
            audio_embed, _ = self.audio_patch_proj(audio_rows)
            write_rows(
                0,
                audio_row_ids,
                audio_embed.to(_BF16_DTYPE),
            )

        t_emb = self.time_embedder(unique_timesteps)
        return embeddings, t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed inference forward.

        Keyword names follow the checkpoint's serving contract.
        Returns `(video_logits, audio_logits)` from rows selected by
        `img_pos_for_infer_output_info` and `audio_pos_info`, with condition
        rows zeroed by update masks.
        """
        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        img_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        update_mask = _required_kwarg(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        skip_mask_out_condition = bool(kwargs.get("skip_mask_out_condition", False))

        text_selected = _required_kwarg(kwargs, "prompt_embeds")

        img_pos = self._pos_ids(_required_kwarg(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._pos_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._pos_ids(
            _required_kwarg(kwargs, "text_pos_info"),
            "text_pos_info",
        )
        infer_out_pos = self._pos_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        psp = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._psp_field(psp, "packed_seq_params", "cu_seqlens_q").to(
            torch.int32
        )
        raw_cu_seqlens_host = self._psp_optional_field(psp, "cu_seqlens_q_host")
        cu_seqlens_host = tuple(
            int(value)
            for value in (
                cu_seqlens.tolist()
                if raw_cu_seqlens_host is None
                else raw_cu_seqlens_host
            )
        )
        max_seqlen = int(self._psp_field(psp, "packed_seq_params", "max_seqlen_q"))
        refiner_psp = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._psp_field(
            refiner_psp, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        refiner_max = int(
            self._psp_field(refiner_psp, "refiner_packed_seq_params", "max_seqlen_q")
        )

        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        seq_len = int(x.shape[1])
        if token_tags is not None and token_tags.shape[0] != seq_len:
            raise ValueError(
                "token_tags must cover the full packed sequence "
                f"({seq_len}), got {token_tags.shape[0]}."
            )
        if inverse_indices.shape[0] != seq_len:
            raise ValueError(
                f"inverse_indices must be [{seq_len}], got {list(inverse_indices.shape)}"
            )
        device = x.device
        self._resolve_attention_backend_once()
        if _ring_world_size() != 1:
            raise NotImplementedError(
                "MiniMax H3 packed multi-segment attention requires "
                "--ring-degree 1; Ring and mixed USP are unsupported."
            )

        sp_ws, sp_rank = _ulysses_ctx()
        local_seq_len = seq_len
        if sp_ws > 1:
            if seq_len % sp_ws:
                raise ValueError(
                    f"packed seq_len {seq_len} not divisible by ulysses "
                    f"world size {sp_ws}"
                )
            local_heads = self.num_attention_heads // get_tp_world_size()
            if local_heads % sp_ws:
                raise ValueError(
                    f"TP-local heads {local_heads} not divisible by Ulysses "
                    f"world size {sp_ws} (total heads={self.num_attention_heads}, "
                    f"TP={get_tp_world_size()})"
                )
            local_seq_len = seq_len // sp_ws
        row_start = sp_rank * local_seq_len
        row_stop = row_start + local_seq_len

        # RoPE and latent projections are row-local before Ulysses exchanges
        # sequence for heads inside attention. Serving normally prepares the
        # request-static cache once; direct model callers use this fallback.
        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_freqs = self.rope(img_position_ids[:, row_start:row_stop]).to(device)
            rope_cache = (
                _rope_cos_sin_cache(rope_freqs, dtype=_BF16_DTYPE),
                torch.arange(
                    local_seq_len,
                    device=device,
                    dtype=torch.long,
                ),
            )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)

        decoder_input, t_emb = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=text_selected,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            refiner_max_seqlen=refiner_max,
            row_start=row_start,
            row_stop=row_stop,
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            local_embedding_layout=kwargs.get("local_embedding_layout"),
        )
        # request-step AdaLN input shared by all blocks
        adaln_input = nn.functional.silu(t_emb).to(_BF16_DTYPE)
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        if block_token_tags is None:
            assert token_tags is not None
            token_tags = token_tags.to(device)
            block_token_tags = token_tags[row_start:row_stop].clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if block_token_tags.shape[0] != local_seq_len:
                raise ValueError(
                    "block_token_tags must cover the rank-local packed sequence "
                    f"({local_seq_len}), got {block_token_tags.shape[0]}."
                )
        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_combined = torch.add(
                block_token_tags,
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )

        hidden = decoder_input
        cu_seqlens = cu_seqlens.to(device)
        block_adaln_params = None
        if self._can_batch_block_adaln():
            local_adaln = torch.stack(
                [block.adaln_proj.project_local(adaln_input) for block in self.blocks]
            )
            gathered_adaln = tensor_model_parallel_all_gather(local_adaln)
            block_adaln_params = tuple(
                block.adaln_proj.split_output(output)
                for block, output in zip(self.blocks, gathered_adaln)
            )
        # With Ulysses sequence parallelism, shard rows across the group for
        # the block stack. Attention trades sequence for heads internally;
        # everything else, including the final layer, is row-local.
        for index, block in enumerate(self.blocks):
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=block_combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=sp_ws > 1,
                adaln_params=(
                    None if block_adaln_params is None else block_adaln_params[index]
                ),
            )
        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=block_inverse,
        )
        if sp_ws > 1:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_sp_group,
            )

            video_width = video_logits.shape[-1]
            logits = get_sp_group().all_gather(
                torch.cat((video_logits, audio_logits), dim=-1), dim=0
            )
            video_logits, audio_logits = logits.split(
                (video_width, logits.shape[-1] - video_width), dim=-1
            )

        # Preserve the full-row output GEMM (and therefore its numerical
        # contract), but defer TP column gathers until after dead text/padding
        # rows have been removed. For hybrid TP+Ulysses, the preceding SP row
        # gather also carries only the TP-local output width.
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        audio_logits = audio_logits.index_select(0, audio_pos.to(device))
        if get_tp_world_size() > 1:
            video_logits = tensor_model_parallel_all_gather(video_logits)
            audio_logits = tensor_model_parallel_all_gather(audio_logits)
        if not skip_mask_out_condition:
            update_mask = update_mask.view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length mismatch: "
                    f"{update_mask.shape[0]} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            # Audio has no condition rows in the supported tasks, so its
            # derived update mask is all ones. Honor an explicit mask when
            # provided.
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiTModel",
    "_reorder_grouped_qkv_to_qkv",
]

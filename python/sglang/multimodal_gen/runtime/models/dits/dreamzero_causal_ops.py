# SPDX-License-Identifier: Apache-2.0
"""Shared DreamZero causal DiT helper ops."""

from __future__ import annotations

import os
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    split_tensor_along_last_dim,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    backend_name_to_enum,
    get_attn_backend,
)
from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import STR_BACKEND_ENV_VAR

logger = init_logger(__name__)

SGLANG_DREAMZERO_TP_FP32_ROW_REDUCE = os.getenv(
    "SGLANG_DREAMZERO_TP_FP32_ROW_REDUCE", "0"
).lower() not in ("0", "false", "no")
SGLANG_DREAMZERO_TP_FP32_RESIDUAL = os.getenv(
    "SGLANG_DREAMZERO_TP_FP32_RESIDUAL", "0"
).lower() not in ("0", "false", "no")
_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS = {
    AttentionBackendEnum.FA2,
    AttentionBackendEnum.TORCH_SDPA,
}


def _residual_add(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    tensor_parallel: bool,
) -> torch.Tensor:
    if SGLANG_DREAMZERO_TP_FP32_RESIDUAL and tensor_parallel:
        src_dtype = x.dtype
        if scale is None:
            return (x.float() + y.float()).to(src_dtype)
        return (x.float() + y.float() * scale.float()).to(src_dtype)
    if scale is None:
        return x + y
    return x + (y * scale)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(half),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    return rope_params_polar(max_seq_len, dim, theta)


def rope_params_polar(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2).to(torch.float64).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if (
        SGLANG_DREAMZERO_TP_FP32_ROW_REDUCE
        and isinstance(layer, RowParallelLinear)
        and layer.tp_size > 1
    ):
        if layer.input_is_parallel:
            input_parallel = x
        else:
            input_parallel = split_tensor_along_last_dim(
                x, num_partitions=layer.tp_size
            )[layer.tp_rank].contiguous()

        # Preserve the normal RowParallelLinear GEMM and bias epilogue. Only
        # promote the partial output while summing it across TP ranks.
        bias = None if (layer.tp_rank > 0 or layer.skip_bias_add) else layer.bias
        output_parallel = layer.quant_method.apply(layer, input_parallel, bias=bias)
        if layer.reduce_results:
            output = layer.tp_group.all_reduce(output_parallel.float()).to(
                output_parallel.dtype
            )
        else:
            output = output_parallel
        if layer.skip_bias_add and layer.bias is not None:
            output = output + layer.bias
        return output

    out = layer(x)
    if not isinstance(out, tuple):
        return out
    output, output_bias = out
    if output_bias is not None:
        output = output + output_bias
    return output


def _tp_wan_rms_norm(x: torch.Tensor, norm: WanRMSNorm) -> torch.Tensor:
    tp_size = get_tp_world_size()
    tp_rank = get_tp_rank()
    src_dtype = x.dtype
    weight = norm.weight.tensor_split(tp_size)[tp_rank]
    x_fp32 = x.float()
    variance = x_fp32.pow(2).sum(dim=-1, keepdim=True)
    variance = get_tp_group().all_reduce(variance)
    variance = variance / (x.shape[-1] * tp_size)
    return (x_fp32 * torch.rsqrt(variance + norm.eps)).to(src_dtype) * weight


def _maybe_qk_norm(
    x: torch.Tensor, norm: nn.Module, *, tensor_parallel: bool
) -> torch.Tensor:
    if isinstance(norm, nn.Identity):
        return x
    if tensor_parallel:
        return _tp_wan_rms_norm(x, norm)
    return norm(x)


def align_modulation(
    parts: tuple[torch.Tensor, ...], target_len: int
) -> tuple[torch.Tensor, ...]:
    aligned = []
    for part in parts:
        part_len = part.shape[1]
        if part_len == target_len:
            aligned.append(part)
        elif part_len >= target_len:
            aligned.append(part[:, :target_len])
        else:
            repeat = (target_len + part_len - 1) // part_len
            aligned.append(part.repeat_interleave(repeat, dim=1)[:, :target_len])
    return tuple(aligned)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device
        ) * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_w = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_w) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        action_emb = self.W1(actions, cat_ids)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        x = torch.cat([action_emb, timestep_emb], dim=-1)
        x = self.W2(x, cat_ids)
        x = x * torch.sigmoid(x)
        return self.W3(x, cat_ids)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attention_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_dtype = q.dtype
    if attention_dtype is None:
        attention_dtype = q.dtype
    q = q.to(attention_dtype)
    k = k.to(attention_dtype)
    v = v.to(attention_dtype)
    backend = _dreamzero_attention_backend()
    if backend == AttentionBackendEnum.FA2:
        try:
            from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                flash_attn_varlen_func,
            )

            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                softmax_scale=q.shape[-1] ** -0.5,
                causal=causal,
                return_softmax_lse=False,
            )
        except ImportError:
            if _dreamzero_attention_backend_is_explicit():
                raise
            _warn_dreamzero_fa2_fallback_once()
        else:
            return out.contiguous().to(out_dtype)

    impl = _dreamzero_attention_impl(
        q.shape[-2],
        q.shape[-1],
        causal,
        attention_dtype,
        AttentionBackendEnum.TORCH_SDPA,
    )
    return impl.forward(q, k, v, attn_metadata=None).contiguous().to(out_dtype)


def _dreamzero_attention_backend_raw() -> str | None:
    return os.environ.get(STR_BACKEND_ENV_VAR)


def _dreamzero_attention_backend_is_explicit() -> bool:
    return _dreamzero_attention_backend_raw() is not None


@cache
def _warn_dreamzero_fa2_fallback_once() -> None:
    logger.warning(
        "DreamZero FA2 backend requested by default, but SGLang FlashAttention "
        "kernel is unavailable; falling back to TORCH_SDPA. Set %s=FA2 to "
        "require FA2 and fail fast.",
        STR_BACKEND_ENV_VAR,
    )


def _dreamzero_attention_backend() -> AttentionBackendEnum:
    raw = _dreamzero_attention_backend_raw()
    if raw is None:
        # Prefer SGLang FlashAttention by default; _attention falls back to SDPA
        # only when the default FA kernel is unavailable.
        return AttentionBackendEnum.FA2

    backend = backend_name_to_enum(raw.strip().upper().replace("-", "_"))
    if backend not in _DREAMZERO_SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            "DreamZero supports only FA2 and TORCH_SDPA attention backends, "
            f"got {raw!r}"
        )
    return backend


@cache
def _dreamzero_attention_impl(
    num_heads: int,
    head_size: int,
    causal: bool,
    dtype: torch.dtype,
    selected_backend: AttentionBackendEnum,
):
    backend_cls = get_attn_backend(
        head_size,
        dtype,
        supported_attention_backends=_DREAMZERO_SUPPORTED_ATTENTION_BACKENDS,
        selected_attention_backend=selected_backend,
    )
    backend = backend_cls.get_enum()
    logger.info_once(
        "DreamZero attention backend: "
        f"{backend.name} (requested={selected_backend.name}, causal={causal})"
    )
    return backend_cls.get_impl_cls()(
        num_heads=num_heads,
        head_size=head_size,
        num_kv_heads=num_heads,
        softmax_scale=head_size**-0.5,
        causal=causal,
    )


def rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    return rope_action_apply_polar(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length,
        num_action_per_block,
        num_state_per_block,
    )


def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    batch, seq_len, num_heads, _ = x.shape
    out_dtype = x.dtype
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(batch, seq_len, num_heads, -1, 2)
    )

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None
        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )
        freqs_action = freqs_action[: chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_state = freqs_state[: chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block, 1, -1
        )
        freqs = torch.cat([freqs, freqs_action, freqs_state], dim=0)

    freqs = freqs.unsqueeze(0)
    return torch.view_as_real(x * freqs).flatten(3).to(out_dtype)


def causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    return causal_rope_action_apply_polar(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length,
        num_action_per_block,
        num_state_per_block,
        action_state_index,
    )


def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    batch, seq_len, num_heads, _ = x.shape
    out_dtype = x.dtype
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(batch, seq_len, num_heads, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index
            * num_action_per_block : (action_state_index + 1)
            * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index
            * num_state_per_block : (action_state_index + 1)
            * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    freqs = freqs.unsqueeze(0)
    return torch.view_as_real(x * freqs).flatten(3).to(out_dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x.to(out_dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

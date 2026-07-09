# SPDX-License-Identifier: Apache-2.0
"""DreamZero causal Wan DiT pieces.

This module starts with the single-block parity surface. It intentionally keeps
the parameter names aligned with Groot's `CausalWanAttentionBlock` so Phase 2
tests can transfer weights with a strict state-dict load before TP sharding is
introduced.
"""

from __future__ import annotations

import math
import os
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_parallel_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.utils import (
    flatten_dim_sp_into_sequence,
    gather_full_sequence_parallel_tensor,
    remove_redundant_action_register,
    shard_sequence_parallel_sequence,
    shard_sequence_parallel_time_embedding,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import STR_BACKEND_ENV_VAR


logger = init_logger(__name__)

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
DREAMZERO_TP_FP32_ROW_REDUCE = (
    os.getenv("DREAMZERO_TP_FP32_ROW_REDUCE", "0").lower() not in ("0", "false", "no")
)
DREAMZERO_DEBUG_DIT_LAYERS = os.getenv("DREAMZERO_DEBUG_DIT_LAYERS", "0") == "1"
DREAMZERO_SP_FULL_FFN_DIAGNOSTIC = (
    os.getenv("DREAMZERO_SP_FULL_FFN_DIAGNOSTIC", "0").lower()
    not in ("0", "false", "no")
)
DREAMZERO_TP_FP32_RESIDUAL = (
    os.getenv("DREAMZERO_TP_FP32_RESIDUAL", "0").lower() not in ("0", "false", "no")
)
DREAMZERO_DEBUG_FFN_ROW_CHANNEL = int(
    os.getenv("DREAMZERO_DEBUG_FFN_ROW_CHANNEL", "4394")
)
_DREAMZERO_ATTENTION_BACKEND_ALIASES = {
    "FA2": AttentionBackendEnum.FA2,
    "FLASH_ATTN": AttentionBackendEnum.FA2,
    "FLASH_ATTENTION": AttentionBackendEnum.FA2,
    "FLA": AttentionBackendEnum.FA2,
    "SDPA": AttentionBackendEnum.TORCH_SDPA,
    "TORCH": AttentionBackendEnum.TORCH_SDPA,
    "TORCH_SDPA": AttentionBackendEnum.TORCH_SDPA,
}


def _debug_block_indices() -> set[int]:
    raw = os.getenv(
        "DREAMZERO_DEBUG_DIT_STEP_BLOCKS",
        os.getenv("DREAMZERO_DEBUG_DIT_STEP_BLOCK", "0"),
    )
    indices: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if item:
            indices.add(int(item))
    return indices


def _debug_layer_call_indices() -> set[int] | None:
    raw = os.getenv("DREAMZERO_DEBUG_DIT_LAYER_CALL_INDICES", "").strip()
    if not raw:
        return None
    return {int(item.strip()) for item in raw.split(",") if item.strip()}


def _debug_step_append(module: nn.Module, name: str, x: torch.Tensor) -> None:
    debug_steps = getattr(module, "_dreamzero_debug_steps", None)
    if debug_steps is not None:
        debug_x = x
        if getattr(module, "_dreamzero_debug_full_sequence", False):
            if get_sp_parallel_rank() != 0:
                return
        elif (
            getattr(module, "_dreamzero_debug_sp_gather", False)
            and debug_x.dim() >= 3
        ):
            gathered_debug_x = gather_full_sequence_parallel_tensor(debug_x)
            action_register_length = getattr(
                module,
                "_dreamzero_debug_sp_action_register_length",
                None,
            )
            if action_register_length is not None:
                debug_x = remove_redundant_action_register(
                    gathered_debug_x,
                    action_register_length,
                )
            else:
                debug_x = flatten_dim_sp_into_sequence(gathered_debug_x)
            if get_sp_parallel_rank() != 0:
                return
        debug_steps.append({"name": name, "x": debug_x.detach().cpu().clone()})


def _residual_add(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    tensor_parallel: bool,
) -> torch.Tensor:
    if DREAMZERO_TP_FP32_RESIDUAL and tensor_parallel:
        src_dtype = x.dtype
        if scale is None:
            return (x.float() + y.float()).to(src_dtype)
        return (x.float() + y.float() * scale.float()).to(src_dtype)
    if scale is None:
        return x + y
    return x + (y * scale)


def _residual_branch(
    y: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    tensor_parallel: bool,
) -> torch.Tensor:
    if scale is None:
        return y
    if DREAMZERO_TP_FP32_RESIDUAL and tensor_parallel:
        return (y.float() * scale.float()).to(y.dtype)
    return y * scale


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(
                half
            ),
        ),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    if ENABLE_TENSORRT:
        return rope_params_no_polar(max_seq_len, dim, theta)
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


def rope_params_no_polar(
    max_seq_len: int, dim: int, theta: int = 10000
) -> torch.Tensor:
    assert dim % 2 == 0
    inv_freq = 1.0 / torch.pow(
        theta, torch.arange(0, dim, 2).to(torch.float32) / dim
    )
    timesteps = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(timesteps, inv_freq)
    return torch.stack((freqs.cos(), freqs.sin()), dim=-1).flatten(-2)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _linear(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if (
        DREAMZERO_TP_FP32_ROW_REDUCE
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
        output_parallel = layer.quant_method.apply(
            layer, input_parallel, bias=bias
        )
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


def _tp_wan_rms_norm(x: torch.Tensor, norm: "WanRMSNorm") -> torch.Tensor:
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


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        _, _ = timesteps.shape
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
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        action_emb = self.W1(actions, cat_ids)
        timestep_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        x = torch.cat([action_emb, timestep_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)


def assemble_action_state_tokens(
    video_tokens: torch.Tensor,
    video_timestep: torch.Tensor,
    action_features: torch.Tensor | None,
    state_features: torch.Tensor | None,
    timestep_action: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int | None, int]:
    """Append DreamZero action/state registers to video tokens.

    Groot order is:
      tokens:    [video_tokens, action_features, state_features]
      timesteps: [video_timestep, timestep_action, timestep_state]
    where timestep_state is a stride sample from timestep_action.
    """
    if action_features is None:
        return video_tokens, video_timestep, None, 0

    assert state_features is not None
    assert timestep_action is not None
    action_register = torch.cat([action_features, state_features], dim=1)
    action_length = action_features.shape[1]
    action_register_length = action_register.shape[1]
    tokens = torch.cat([video_tokens, action_register], dim=1)

    stride = timestep_action.shape[1] // state_features.shape[1]
    timestep_state = timestep_action[:, ::stride]
    timesteps = torch.cat([video_timestep, timestep_action, timestep_state], dim=1)
    return tokens, timesteps, action_register_length, action_length


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
        attention_dtype = q.dtype if q.dtype == torch.float32 else torch.bfloat16
    q = q.to(attention_dtype)
    k = k.to(attention_dtype)
    v = v.to(attention_dtype)
    backend = _dreamzero_attention_backend()
    if backend == AttentionBackendEnum.FA2:
        try:
            out = _dreamzero_flash_attention_varlen(q, k, v, causal=causal)
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


def _dreamzero_flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
        flash_attn_varlen_func,
    )

    return flash_attn_varlen_func(
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


def _dreamzero_attention_backend_raw() -> str | None:
    raw = os.environ.get("DREAMZERO_ATTENTION_BACKEND")
    if raw is None:
        raw = os.environ.get(STR_BACKEND_ENV_VAR)
    return raw


def _dreamzero_attention_backend_is_explicit() -> bool:
    return _dreamzero_attention_backend_raw() is not None


@cache
def _warn_dreamzero_fa2_fallback_once() -> None:
    logger.warning(
        "DreamZero FA2 backend requested by default, but SGLang FlashAttention "
        "kernel is unavailable; falling back to TORCH_SDPA. Set "
        "DREAMZERO_ATTENTION_BACKEND=FA2 to require FA2 and fail fast."
    )


def _dreamzero_attention_backend() -> AttentionBackendEnum:
    raw = _dreamzero_attention_backend_raw()
    if raw is None:
        # Prefer SGLang FlashAttention by default; _attention falls back to SDPA
        # only when the default FA kernel is unavailable.
        return AttentionBackendEnum.FA2

    normalized = raw.strip().upper().replace("-", "_")
    try:
        return _DREAMZERO_ATTENTION_BACKEND_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(
            "DreamZero supports only FA2 and TORCH_SDPA attention backends, "
            f"got {raw!r}"
        ) from exc


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
        supported_attention_backends={
            AttentionBackendEnum.FA2,
            AttentionBackendEnum.TORCH_SDPA,
        },
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
    if ENABLE_TENSORRT:
        return rope_action_apply_no_polar(
            x,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block,
            num_state_per_block,
        )
    return rope_action_apply_polar(
        x,
        freqs,
        freqs_action,
        freqs_state,
        action_register_length,
        num_action_per_block,
        num_state_per_block,
    )


def rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    batch, seq_len, num_heads, dim = x.shape

    if action_register_length is not None:
        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )
        freqs = torch.cat(
            [
                freqs,
                freqs_action[: chunk_size * num_action_per_block],
                freqs_state[: chunk_size * num_state_per_block],
            ],
            dim=0,
        )

    freqs = freqs.unsqueeze(0).unsqueeze(2)
    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)
    return torch.cat(
        (x0 * freqs_cos - x1 * freqs_sin, x1 * freqs_cos + x0 * freqs_sin),
        dim=-1,
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
    x = torch.view_as_complex(x.to(torch.float64).reshape(batch, seq_len, num_heads, -1, 2))

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
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(
            x,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block,
            num_state_per_block,
            action_state_index,
        )
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


def causal_rope_action_apply_no_polar(
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
    x = x.reshape(batch, seq_len, num_heads, -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0]
    freqs_sin = freqs[..., 1]

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
            action_register_length, 1, -1, 2
        )
        freqs_cos = torch.cat([freqs_cos[0], freqs_1d[..., 0]], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_1d[..., 1]], dim=0).unsqueeze(0)

    x_rotated = torch.stack(
        (
            x_real * freqs_cos - x_imag * freqs_sin,
            x_real * freqs_sin + x_imag * freqs_cos,
        ),
        dim=-1,
    )
    return x_rotated.flatten(3).to(out_dtype)


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
    x = torch.view_as_complex(x.to(torch.float64).reshape(batch, seq_len, num_heads, -1, 2))

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
    def __init__(
        self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False
    ):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)


class DreamZeroT2VCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm=True,
        eps: float = 1e-6,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.head_dim = dim // num_heads
        self.use_tensor_parallel = use_tensor_parallel
        linear_out = (
            lambda: RowParallelLinear(
                dim,
                dim,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=True,
            )
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        linear_in = (
            lambda: ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.q = linear_in()
        self.k = linear_in()
        self.v = linear_in()
        self.o = linear_out()
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        if context_lens is not None:
            raise NotImplementedError("context_lens masking is not part of Phase 2.1")
        batch = x.shape[0]
        q = _maybe_qk_norm(
            _linear(self.q, x), self.norm_q, tensor_parallel=self.use_tensor_parallel
        ).view(batch, -1, self.local_num_heads, self.head_dim)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = _maybe_qk_norm(
                    _linear(self.k, context),
                    self.norm_k,
                    tensor_parallel=self.use_tensor_parallel,
                ).view(
                    batch, -1, self.local_num_heads, self.head_dim
                )
                v = _linear(self.v, context).view(
                    batch, -1, self.local_num_heads, self.head_dim
                )
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = _maybe_qk_norm(
                _linear(self.k, context),
                self.norm_k,
                tensor_parallel=self.use_tensor_parallel,
            ).view(batch, -1, self.local_num_heads, self.head_dim)
            v = _linear(self.v, context).view(
                batch, -1, self.local_num_heads, self.head_dim
            )
        return _linear(self.o, _attention(q, k, v).flatten(2))


class DreamZeroI2VCrossAttention(DreamZeroT2VCrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm=True,
        eps: float = 1e-6,
        use_tensor_parallel: bool = False,
    ):
        super().__init__(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.k_img = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.v_img = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        crossattn_cache: dict | None = None,
    ) -> torch.Tensor:
        context_img = context[:, :257]
        context = context[:, 257:]
        batch = x.shape[0]
        q = _maybe_qk_norm(
            _linear(self.q, x), self.norm_q, tensor_parallel=self.use_tensor_parallel
        ).view(batch, -1, self.local_num_heads, self.head_dim)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = _maybe_qk_norm(
                    _linear(self.k, context),
                    self.norm_k,
                    tensor_parallel=self.use_tensor_parallel,
                ).view(
                    batch, -1, self.local_num_heads, self.head_dim
                )
                v = _linear(self.v, context).view(
                    batch, -1, self.local_num_heads, self.head_dim
                )
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = _maybe_qk_norm(
                _linear(self.k, context),
                self.norm_k,
                tensor_parallel=self.use_tensor_parallel,
            ).view(batch, -1, self.local_num_heads, self.head_dim)
            v = _linear(self.v, context).view(
                batch, -1, self.local_num_heads, self.head_dim
            )
        _debug_step_append(self, "cross_attn.q", q)
        _debug_step_append(self, "cross_attn.text_k", k)
        _debug_step_append(self, "cross_attn.text_v", v)
        text_x = _attention(q, k, v).flatten(2)
        _debug_step_append(self, "cross_attn.text_attn_out", text_x)
        k_img = _maybe_qk_norm(
            _linear(self.k_img, context_img),
            self.norm_k_img,
            tensor_parallel=self.use_tensor_parallel,
        ).view(batch, -1, self.local_num_heads, self.head_dim)
        v_img = _linear(self.v_img, context_img).view(
            batch, -1, self.local_num_heads, self.head_dim
        )
        _debug_step_append(self, "cross_attn.img_k", k_img)
        _debug_step_append(self, "cross_attn.img_v", v_img)
        img_x = _attention(q, k_img, v_img).flatten(2)
        _debug_step_append(self, "cross_attn.img_attn_out", img_x)
        merged_x = text_x + img_x
        _debug_step_append(self, "cross_attn.text_img_sum", merged_x)
        out = _linear(self.o, merged_x)
        _debug_step_append(self, "cross_attn.o_proj_out", out)
        return out


WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": DreamZeroT2VCrossAttention,
    "i2v_cross_attn": DreamZeroI2VCrossAttention,
}


class DreamZeroCausalWanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm=True,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.use_tensor_parallel = use_tensor_parallel
        self.max_attention_size = (
            21 * frame_seqlen
            if local_attn_size == -1
            else local_attn_size * frame_seqlen
        )
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.q = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.k = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.v = (
            ColumnParallelLinear(dim, dim, gather_output=False)
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.o = (
            RowParallelLinear(
                dim,
                dim,
                input_is_parallel=True,
                skip_bias_add=False,
                reduce_results=True,
            )
            if use_tensor_parallel
            else nn.Linear(dim, dim)
        )
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _process_clean_image_only(
        self,
        clean_image_q: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        clean_frames: int,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        if num_blocks == 0:
            return _attention(
                clean_image_q[:, : self.frame_seqlen],
                clean_image_k[:, : self.frame_seqlen],
                clean_image_v[:, : self.frame_seqlen],
            )

        output = torch.empty_like(clean_image_q)
        output[:, : self.frame_seqlen] = _attention(
            clean_image_q[:, : self.frame_seqlen],
            clean_image_k[:, : self.frame_seqlen],
            clean_image_v[:, : self.frame_seqlen],
        )
        if self.local_attn_size == -1:
            output[:, self.frame_seqlen :] = _attention(
                clean_image_q[:, self.frame_seqlen :],
                clean_image_k,
                clean_image_v,
                causal=True,
            )
        else:
            for block_idx in range(num_blocks):
                block_start = self.frame_seqlen + block_idx * block_size
                block_end = min(block_start + block_size, clean_image_q.shape[1])
                kv_start = max(0, block_end - self.local_attn_size * self.frame_seqlen)
                output[:, block_start:block_end] = _attention(
                    clean_image_q[:, block_start:block_end],
                    clean_image_k[:, kv_start:block_end],
                    clean_image_v[:, kv_start:block_end],
                )
        return output

    def _process_state_blocks(
        self,
        noisy_state_q: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        state_horizon: int,
    ) -> torch.Tensor:
        output = torch.empty_like(noisy_state_q)
        num_blocks = state_horizon // self.num_state_per_block
        for block_idx in range(num_blocks):
            start = block_idx * self.num_state_per_block
            end = start + self.num_state_per_block
            output[:, start:end] = _attention(
                noisy_state_q[:, start:end],
                noisy_state_k[:, start:end],
                noisy_state_v[:, start:end],
            )
        return output

    def _process_noisy_image_blocks(
        self,
        noisy_image_q: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        noisy_frames: int,
        action_horizon: int,
        state_horizon: int,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (noisy_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_image_q)
        output[:, : self.frame_seqlen] = _attention(
            noisy_image_q[:, : self.frame_seqlen],
            noisy_image_k[:, : self.frame_seqlen],
            noisy_image_v[:, : self.frame_seqlen],
        )
        assert action_horizon == num_blocks * self.num_action_per_block
        assert state_horizon == num_blocks * self.num_state_per_block
        for block_idx in range(num_blocks):
            block_start = self.frame_seqlen + block_idx * block_size
            block_end = min(block_start + block_size, noisy_image_q.shape[1])
            clean_end = self.frame_seqlen + block_idx * block_size
            if self.local_attn_size != -1:
                clean_start = max(
                    0, clean_end - self.local_attn_size * self.frame_seqlen
                )
            else:
                clean_start = 0
            action_start = block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            state_start = block_idx * self.num_state_per_block
            state_end = state_start + self.num_state_per_block
            output[:, block_start:block_end] = _attention(
                noisy_image_q[:, block_start:block_end],
                torch.cat(
                    [
                        clean_image_k[:, clean_start:clean_end],
                        noisy_image_k[:, block_start:block_end],
                        noisy_action_k[:, action_start:action_end],
                        noisy_state_k[:, state_start:state_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        clean_image_v[:, clean_start:clean_end],
                        noisy_image_v[:, block_start:block_end],
                        noisy_action_v[:, action_start:action_end],
                        noisy_state_v[:, state_start:state_end],
                    ],
                    dim=1,
                ),
            )
        return output

    def _process_noisy_action_blocks(
        self,
        noisy_action_q: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        noisy_frames: int,
        action_horizon: int,
        state_horizon: int,
    ) -> torch.Tensor:
        num_blocks = (noisy_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_action_q)
        assert action_horizon == num_blocks * self.num_action_per_block
        assert state_horizon == num_blocks * self.num_state_per_block
        block_size = self.frame_seqlen * self.num_frame_per_block
        for block_idx in range(num_blocks):
            action_start = block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            clean_end = self.frame_seqlen + block_idx * block_size
            noisy_img_start = self.frame_seqlen + block_idx * block_size
            noisy_img_end = noisy_img_start + block_size
            state_start = block_idx * self.num_state_per_block
            state_end = state_start + self.num_state_per_block
            output[:, action_start:action_end] = _attention(
                noisy_action_q[:, action_start:action_end],
                torch.cat(
                    [
                        clean_image_k[:, :clean_end],
                        noisy_image_k[:, noisy_img_start:noisy_img_end],
                        noisy_action_k[:, action_start:action_end],
                        noisy_state_k[:, state_start:state_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        clean_image_v[:, :clean_end],
                        noisy_image_v[:, noisy_img_start:noisy_img_end],
                        noisy_action_v[:, action_start:action_end],
                        noisy_state_v[:, state_start:state_end],
                    ],
                    dim=1,
                ),
            )
        return output

    def _blockwise_causal_flash_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        action_horizon: int | None = None,
        state_horizon: int | None = None,
        num_action_per_block: int | None = None,
        num_state_per_block: int | None = None,
    ) -> torch.Tensor:
        total_len = q.shape[1]
        has_action_state = action_horizon is not None and state_horizon is not None
        if not has_action_state:
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block
            if num_blocks <= 0:
                return _attention(q, k, v)
            if self.local_attn_size == -1:
                return _attention(q, k, v, causal=True)
            output = torch.empty_like(q)
            output[:, :frame_seqlen] = _attention(
                q[:, :frame_seqlen], k[:, :frame_seqlen], v[:, :frame_seqlen]
            )
            for block_idx in range(num_blocks):
                block_start = frame_seqlen + block_idx * block_size
                block_end = min(block_start + block_size, total_len)
                kv_start = max(0, block_end - self.local_attn_size * frame_seqlen)
                output[:, block_start:block_end] = _attention(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end],
                )
            return output

        assert action_horizon is not None
        assert state_horizon is not None
        assert num_action_per_block is not None
        assert num_state_per_block is not None
        first_image_len = frame_seqlen
        image_blocks_len = total_len - first_image_len - action_horizon - state_horizon
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        assert num_image_blocks == num_action_blocks == num_state_blocks

        image_blocks_start = first_image_len
        action_start = image_blocks_start + image_blocks_len
        state_start = action_start + action_horizon
        output = torch.empty_like(q)
        output[:, :first_image_len] = _attention(
            q[:, :first_image_len], k[:, :first_image_len], v[:, :first_image_len]
        )
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            image_kv_start = (
                max(
                    image_blocks_start,
                    block_end - self.local_attn_size * frame_seqlen,
                )
                if self.local_attn_size != -1
                else image_blocks_start
            )
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_block_start + num_action_per_block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_block_start + num_state_per_block
            output[:, block_start:block_end] = _attention(
                q[:, block_start:block_end],
                torch.cat(
                    [
                        k[:, :first_image_len],
                        k[:, image_kv_start:block_end],
                        k[:, action_block_start:action_block_end],
                        k[:, state_block_start:state_block_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        v[:, :first_image_len],
                        v[:, image_kv_start:block_end],
                        v[:, action_block_start:action_block_end],
                        v[:, state_block_start:state_block_end],
                    ],
                    dim=1,
                ),
            )
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_block_start + num_action_per_block
            image_block_end = (
                image_blocks_start
                + (block_idx + 1) * num_frame_per_block * frame_seqlen
            )
            image_kv_start = (
                max(
                    image_blocks_start,
                    image_block_end - self.local_attn_size * frame_seqlen,
                )
                if self.local_attn_size != -1
                else image_blocks_start
            )
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_block_start + num_state_per_block
            output[:, action_block_start:action_block_end] = _attention(
                q[:, action_block_start:action_block_end],
                torch.cat(
                    [
                        k[:, :first_image_len],
                        k[:, image_kv_start:image_block_end],
                        k[:, action_block_start:action_block_end],
                        k[:, state_block_start:state_block_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        v[:, :first_image_len],
                        v[:, image_kv_start:image_block_end],
                        v[:, action_block_start:action_block_end],
                        v[:, state_block_start:state_block_end],
                    ],
                    dim=1,
                ),
            )
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_block_start + num_state_per_block
            output[:, state_block_start:state_block_end] = _attention(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end],
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = x.shape[:2]
        q = _maybe_qk_norm(
            _linear(self.q, x), self.norm_q, tensor_parallel=self.use_tensor_parallel
        ).view(batch, seq_len, self.local_num_heads, self.head_dim)
        k = _maybe_qk_norm(
            _linear(self.k, x), self.norm_k, tensor_parallel=self.use_tensor_parallel
        ).view(batch, seq_len, self.local_num_heads, self.head_dim)
        v = _linear(self.v, x).view(
            batch, seq_len, self.local_num_heads, self.head_dim
        )
        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            if is_tf:
                half_seq_len = (
                    seq_len - (action_register_length if action_register_length else 0)
                ) // 2
                q_context = q[:, :half_seq_len]
                k_context = k[:, :half_seq_len]
                q_noisy = q[:, half_seq_len:]
                k_noisy = k[:, half_seq_len:]
                roped_query = torch.cat(
                    [
                        rope_action_apply(
                            q_context,
                            freqs,
                            freqs_action,
                            freqs_state,
                            action_register_length=None,
                        ).type_as(v),
                        rope_action_apply(
                            q_noisy,
                            freqs,
                            freqs_action,
                            freqs_state,
                            action_register_length=action_register_length,
                            num_action_per_block=self.num_action_per_block,
                            num_state_per_block=self.num_state_per_block,
                        ).type_as(v),
                    ],
                    dim=1,
                )
                roped_key = torch.cat(
                    [
                        rope_action_apply(
                            k_context,
                            freqs,
                            freqs_action,
                            freqs_state,
                            action_register_length=None,
                        ).type_as(v),
                        rope_action_apply(
                            k_noisy,
                            freqs,
                            freqs_action,
                            freqs_state,
                            action_register_length=action_register_length,
                            num_action_per_block=self.num_action_per_block,
                            num_state_per_block=self.num_state_per_block,
                        ).type_as(v),
                    ],
                    dim=1,
                )

                if action_register_length is not None:
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (
                        noisy_frames - 1
                    ) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block
                    expected_len = (
                        half_seq_len
                        + noisy_image_seq_len
                        + action_horizon
                        + state_horizon
                    )
                    if roped_query.shape[1] != expected_len:
                        raise ValueError(
                            "Sequence length does not match DreamZero action/state block layout: "
                            f"got {roped_query.shape[1]}, expected {expected_len}."
                        )

                    clean_image_outputs = self._process_clean_image_only(
                        roped_query[:, :clean_image_seq_len],
                        roped_key[:, :clean_image_seq_len],
                        v[:, :clean_image_seq_len],
                        clean_frames,
                    )
                    noisy_image_start = half_seq_len
                    noisy_action_start = noisy_image_start + noisy_image_seq_len
                    noisy_state_start = noisy_action_start + action_horizon
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        roped_query[
                            :, noisy_image_start : noisy_image_start + noisy_image_seq_len
                        ],
                        roped_key[
                            :, noisy_image_start : noisy_image_start + noisy_image_seq_len
                        ],
                        v[:, noisy_image_start : noisy_image_start + noisy_image_seq_len],
                        roped_key[:, :clean_image_seq_len],
                        v[:, :clean_image_seq_len],
                        roped_key[
                            :,
                            noisy_action_start : noisy_action_start + action_horizon,
                        ],
                        v[:, noisy_action_start : noisy_action_start + action_horizon],
                        roped_key[:, noisy_state_start:],
                        v[:, noisy_state_start:],
                        noisy_frames,
                        action_horizon,
                        state_horizon,
                    )
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        roped_query[
                            :,
                            noisy_action_start : noisy_action_start + action_horizon,
                        ],
                        roped_key[
                            :,
                            noisy_action_start : noisy_action_start + action_horizon,
                        ],
                        v[:, noisy_action_start : noisy_action_start + action_horizon],
                        roped_key[:, :clean_image_seq_len],
                        v[:, :clean_image_seq_len],
                        roped_key[
                            :, noisy_image_start : noisy_image_start + noisy_image_seq_len
                        ],
                        v[:, noisy_image_start : noisy_image_start + noisy_image_seq_len],
                        roped_key[:, noisy_state_start:],
                        v[:, noisy_state_start:],
                        noisy_frames,
                        action_horizon,
                        state_horizon,
                    )
                    noisy_state_outputs = self._process_state_blocks(
                        roped_query[:, noisy_state_start:],
                        roped_key[:, noisy_state_start:],
                        v[:, noisy_state_start:],
                        state_horizon,
                    )
                    out = torch.cat(
                        [
                            clean_image_outputs,
                            noisy_image_outputs,
                            noisy_action_outputs,
                            noisy_state_outputs,
                        ],
                        dim=1,
                    )
                else:
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    clean_out = self._blockwise_causal_flash_attn(
                        clean_q,
                        clean_k,
                        clean_v,
                        self.frame_seqlen,
                        self.num_frame_per_block,
                    )
                    noisy_out = _attention(
                        noisy_q,
                        torch.cat([clean_k, noisy_k], dim=1),
                        torch.cat([clean_v, noisy_v], dim=1),
                    )
                    out = torch.cat([clean_out, noisy_out], dim=1)
            else:
                roped_query = rope_action_apply(
                    q,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    self.num_action_per_block,
                    self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    k,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    self.num_action_per_block,
                    self.num_state_per_block,
                ).type_as(v)
                if action_register_length is not None:
                    chunk_size = action_register_length // (
                        self.num_action_per_block + self.num_state_per_block
                    )
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None
                out = self._blockwise_causal_flash_attn(
                    roped_query,
                    roped_key,
                    v,
                    self.frame_seqlen,
                    self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=(
                        self.num_action_per_block if action_register_length else None
                    ),
                    num_state_per_block=(
                        self.num_state_per_block if action_register_length else None
                    ),
                )
        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block
            roped_query = causal_rope_action_apply(
                q,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length,
                self.num_action_per_block,
                self.num_state_per_block,
                action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                k,
                freqs,
                freqs_action,
                freqs_state,
                action_register_length,
                self.num_action_per_block,
                self.num_state_per_block,
                action_state_index,
            ).type_as(v)
            if enable_sequence_parallel:
                roped_key = gather_full_sequence_parallel_tensor(roped_key)
                v = gather_full_sequence_parallel_tensor(v)
                if action_register_length is not None:
                    roped_key = remove_redundant_action_register(
                        roped_key, action_register_length
                    )
                    v = remove_redundant_action_register(v, action_register_length)
                else:
                    roped_key = flatten_dim_sp_into_sequence(roped_key)
                    v = flatten_dim_sp_into_sequence(v)
            roped_action_query = roped_action_key = action_v = None
            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]

            updated_k = torch.cat([kv_cache[0], roped_key], dim=1)
            updated_v = torch.cat([kv_cache[1], v], dim=1)
            updated_k = updated_k[:, -self.max_attention_size :]
            updated_v = updated_v[:, -self.max_attention_size :]
            if action_register_length is not None:
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None
                out = _attention(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([updated_k, roped_action_key], dim=1),
                    torch.cat([updated_v, action_v], dim=1),
                )
            else:
                out = _attention(roped_query, updated_k, updated_v)
            updated_kv_cache = torch.stack([updated_k, updated_v], dim=0)

        out = _linear(self.o, out.flatten(2))
        return out, updated_kv_cache


class DreamZeroCausalWanTransformerBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        frame_seqlen: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_frame_per_block: int = 1,
        qk_norm=True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_tensor_parallel = use_tensor_parallel
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = DreamZeroCausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
            use_tensor_parallel=use_tensor_parallel,
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, qk_norm, eps, use_tensor_parallel=use_tensor_parallel
        )
        self.norm2 = WanLayerNorm(dim, eps)
        if use_tensor_parallel:
            self.ffn = nn.ModuleList(
                [
                    ColumnParallelLinear(dim, ffn_dim, gather_output=False),
                    nn.GELU(approximate="tanh"),
                    RowParallelLinear(
                        ffn_dim,
                        dim,
                        input_is_parallel=True,
                        skip_bias_add=False,
                        reduce_results=True,
                    ),
                ]
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, dim),
            )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _run_ffn(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_tensor_parallel:
            col = self.ffn[0](x)
            _debug_step_append(self, "ffn_col", col)
            act = self.ffn[1](col)
            _debug_step_append(self, "ffn_gelu", act)
            row = self.ffn[2](act)
            _debug_step_append(self, "ffn_row", row)
            return row
        col = _linear(self.ffn[0], x)
        _debug_step_append(self, "ffn_col", col)
        act = self.ffn[1](col)
        _debug_step_append(self, "ffn_gelu", act)
        if getattr(self, "_dreamzero_debug_row_collectives", False):
            row_layer = self.ffn[2]
            gathered_cols = get_tp_group().all_gather(
                col.contiguous(),
                separate_tensors=True,
            )
            if isinstance(gathered_cols, list):
                for rank_index, col_shard in enumerate(gathered_cols):
                    _debug_step_append(
                        self,
                        f"ffn_col_rank{rank_index}",
                        col_shard,
                    )
            gathered_acts = get_tp_group().all_gather(
                act.contiguous(),
                separate_tensors=True,
            )
            if isinstance(gathered_acts, list):
                for rank_index, act_shard in enumerate(gathered_acts):
                    _debug_step_append(
                        self,
                        f"ffn_gelu_rank{rank_index}",
                        act_shard,
                    )
            local_partial = F.linear(act, row_layer.weight, bias=None)
            gathered_local_partials = get_tp_group().all_gather(
                local_partial.contiguous(),
                separate_tensors=True,
            )
            reduced_no_bias = tensor_model_parallel_all_reduce(
                local_partial.clone(), tp_group=row_layer.tp_group
            )
            row_channel = DREAMZERO_DEBUG_FFN_ROW_CHANNEL
            _debug_step_append(self, "ffn_row_local_partial", local_partial)
            if isinstance(gathered_local_partials, list):
                for rank_index, partial in enumerate(gathered_local_partials):
                    _debug_step_append(
                        self,
                        f"ffn_row_local_partial_rank{rank_index}",
                        partial,
                    )
                    _debug_step_append(
                        self,
                        f"ffn_row_channel{row_channel}_local_partial_rank"
                        f"{rank_index}",
                        partial[..., row_channel],
                    )
            _debug_step_append(self, "ffn_row_reduced_no_bias", reduced_no_bias)
            _debug_step_append(
                self,
                f"ffn_row_channel{row_channel}_reduced_no_bias",
                reduced_no_bias[..., row_channel],
            )
            if row_layer.bias is not None:
                _debug_step_append(
                    self,
                    "ffn_row_bias_added",
                    reduced_no_bias + row_layer.bias,
                )
            local_partial_fp32 = F.linear(
                act.float(),
                row_layer.weight.float(),
                bias=None,
            )
            gathered_local_partials_fp32 = get_tp_group().all_gather(
                local_partial_fp32.contiguous(),
                separate_tensors=True,
            )
            reduced_fp32_no_bias = tensor_model_parallel_all_reduce(
                local_partial_fp32.clone(), tp_group=row_layer.tp_group
            )
            _debug_step_append(
                self,
                "ffn_row_local_partial_fp32",
                local_partial_fp32,
            )
            if isinstance(gathered_local_partials_fp32, list):
                for rank_index, partial in enumerate(gathered_local_partials_fp32):
                    _debug_step_append(
                        self,
                        f"ffn_row_local_partial_fp32_rank{rank_index}",
                        partial,
                    )
                    _debug_step_append(
                        self,
                        f"ffn_row_channel{row_channel}_local_partial_fp32_rank"
                        f"{rank_index}",
                        partial[..., row_channel],
                    )
            _debug_step_append(
                self,
                "ffn_row_reduced_fp32_no_bias",
                reduced_fp32_no_bias,
            )
            _debug_step_append(
                self,
                f"ffn_row_channel{row_channel}_reduced_fp32_no_bias",
                reduced_fp32_no_bias[..., row_channel],
            )
            if row_layer.bias is not None:
                _debug_step_append(
                    self,
                    "ffn_row_fp32_bias_added",
                    reduced_fp32_no_bias + row_layer.bias.float(),
                )
        row = _linear(self.ffn[2], act)
        _debug_step_append(self, "ffn_row", row)
        return row

    @staticmethod
    def _align_modulation(
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

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: dict | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        debug_steps = getattr(self, "_dreamzero_debug_steps", None)
        e_parts = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        e_parts = self._align_modulation(e_parts, x.shape[1])

        self_attn_input = (
            self.norm1(x) * (1 + e_parts[1].squeeze(2)) + e_parts[0].squeeze(2)
        )
        _debug_step_append(self, "self_attn_input", self_attn_input)
        y, updated_kv_cache = self.self_attn(
            x=self_attn_input,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
            enable_sequence_parallel=enable_sequence_parallel,
        )
        if debug_steps is not None:
            _debug_step_append(self, "self_attn", y)
        _debug_step_append(self, "self_residual_scale", e_parts[2].squeeze(2))
        self_residual_branch = _residual_branch(
            y,
            e_parts[2].squeeze(2),
            tensor_parallel=self.use_tensor_parallel,
        )
        _debug_step_append(self, "self_residual_branch", self_residual_branch)
        x = _residual_add(
            x,
            y,
            e_parts[2].squeeze(2),
            tensor_parallel=self.use_tensor_parallel,
        )
        _debug_step_append(self, "after_self_residual", x)
        cross_attn_input = self.norm3(x)
        _debug_step_append(self, "cross_attn_input", cross_attn_input)
        if debug_steps is not None:
            self.cross_attn._dreamzero_debug_steps = debug_steps
            self.cross_attn._dreamzero_debug_full_sequence = getattr(
                self, "_dreamzero_debug_full_sequence", False
            )
            self.cross_attn._dreamzero_debug_sp_gather = getattr(
                self, "_dreamzero_debug_sp_gather", False
            )
            self.cross_attn._dreamzero_debug_sp_action_register_length = getattr(
                self, "_dreamzero_debug_sp_action_register_length", None
            )
        try:
            cross = self.cross_attn(
                cross_attn_input,
                context,
                crossattn_cache=crossattn_cache,
            )
        finally:
            if debug_steps is not None:
                self.cross_attn._dreamzero_debug_steps = None
        if debug_steps is not None:
            _debug_step_append(self, "cross_attn", cross)
        cross_residual_branch = _residual_branch(
            cross,
            tensor_parallel=self.use_tensor_parallel,
        )
        _debug_step_append(self, "cross_residual_branch", cross_residual_branch)
        x = _residual_add(
            x,
            cross,
            tensor_parallel=self.use_tensor_parallel,
        )
        norm2_input = self.norm2(x) * (1 + e_parts[4].squeeze(2)) + e_parts[3].squeeze(2)
        _debug_step_append(self, "norm2_input", norm2_input)
        ffn_input = norm2_input
        _debug_step_append(self, "ffn_input", ffn_input)
        if enable_sequence_parallel and DREAMZERO_SP_FULL_FFN_DIAGNOSTIC:
            if (
                not getattr(self, "_dreamzero_full_ffn_diagnostic_logged", False)
                and get_sp_parallel_rank() == 0
            ):
                print(
                    "[DreamZero SP Diagnostic] running each FFN on the gathered "
                    "full sequence",
                    flush=True,
                )
                self._dreamzero_full_ffn_diagnostic_logged = True
            gathered_ffn_input = gather_full_sequence_parallel_tensor(ffn_input)
            if action_register_length is not None:
                full_ffn_input = remove_redundant_action_register(
                    gathered_ffn_input,
                    action_register_length,
                )
            else:
                full_ffn_input = flatten_dim_sp_into_sequence(gathered_ffn_input)

            self._dreamzero_debug_full_sequence = True
            try:
                full_y = self._run_ffn(full_ffn_input)
                _debug_step_append(self, "ffn_full_sequence", full_y)
            finally:
                self._dreamzero_debug_full_sequence = False

            local_video_length = ffn_input.shape[1] - (
                action_register_length or 0
            )
            local_video_begin = get_sp_parallel_rank() * local_video_length
            local_video_end = local_video_begin + local_video_length
            y = full_y[:, local_video_begin:local_video_end]
            if action_register_length is not None:
                y = torch.cat(
                    [y, full_y[:, -action_register_length:]],
                    dim=1,
                )
        else:
            y = self._run_ffn(ffn_input)
        _debug_step_append(self, "ffn", y)
        _debug_step_append(self, "ffn_residual_scale", e_parts[5].squeeze(2))
        ffn_residual_branch = _residual_branch(
            y,
            e_parts[5].squeeze(2),
            tensor_parallel=self.use_tensor_parallel,
        )
        _debug_step_append(self, "ffn_residual_branch", ffn_residual_branch)
        x = _residual_add(
            x,
            y,
            e_parts[5].squeeze(2),
            tensor_parallel=self.use_tensor_parallel,
        )
        return x, updated_kv_cache


class DreamZeroCausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, ...], eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        e_parts = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        target_len = x.shape[1]
        aligned = []
        for part in e_parts:
            part_len = part.shape[1]
            if part_len == target_len:
                aligned.append(part)
            elif part_len >= target_len:
                aligned.append(part[:, :target_len])
            else:
                repeat = (target_len + part_len - 1) // part_len
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :target_len])
        shift, scale = aligned
        return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(image_embeds)


class DreamZeroCausalWanModel(nn.Module):
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: tuple[int, int, int] = (1, 2, 2),
        frame_seqlen: int = 220,
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        max_chunk_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        num_frame_per_block: int = 1,
        action_dim: int = 32,
        num_registers: int = 8,
        max_state_dim: int = 64,
        max_num_embodiments: int = 32,
        hidden_size: int = 1024,
        diffusion_model_pretrained_path=None,
        num_action_per_block: int = 32,
        num_state_per_block: int = 1,
        concat_first_frame_latent: bool = True,
        use_tensor_parallel: bool = False,
    ):
        super().__init__()
        assert model_type in ["t2v", "i2v", "ti2v"]
        self.model_type = model_type
        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.local_num_heads = (
            divide(num_heads, get_tp_world_size()) if use_tensor_parallel else num_heads
        )
        self.num_layers = num_layers
        self.local_attn_size = (
            max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        )
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        self.concat_first_frame_latent = concat_first_frame_latent
        self.use_tensor_parallel = use_tensor_parallel

        max_num_embodiments = 1
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                DreamZeroCausalWanTransformerBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    frame_seqlen,
                    self.local_attn_size,
                    sink_size,
                    num_frame_per_block,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    num_action_per_block,
                    num_state_per_block,
                    use_tensor_parallel,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = DreamZeroCausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        self.freqs_action = rope_params(1024 * 10, head_dim)
        self.freqs_state = rope_params(1024, head_dim)
        self.freqs = [
            rope_params(1024, head_dim - 4 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
            rope_params(1024, 2 * (head_dim // 6)),
        ]
        if model_type in ("i2v", "ti2v"):
            self.img_emb = MLPProj(1280, dim)

        self.gradient_checkpointing = True
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True

    def _create_freqs(
        self, grid_size: torch.Tensor, start_frame: int
    ) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        frames, height, width = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame : start_frame + frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                self.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1)
        return freqs

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[dict] | None,
        current_start_frame: int,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        x = x.flatten(start_dim=2).transpose(1, 2)
        if enable_sequence_parallel:
            x, freqs = shard_sequence_parallel_sequence(x, freqs)
            sp_seq_len = x.shape[1]
        else:
            sp_seq_len = seq_len
        batch = x.shape[0]
        num_timestep_frames = timestep.shape[1]

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(
                action, timestep_action, embodiment_id
            )
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_length = 0
            action_register_length = None
            state_features = None

        if num_timestep_frames <= seq_len:
            repeat = (seq_len + num_timestep_frames - 1) // num_timestep_frames
            timestep = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(
                0,
                num_timestep_frames - 1,
                seq_len,
                device=timestep.device,
                dtype=torch.long,
            )
            timestep = timestep[:, indices]

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        e = e.unflatten(dim=0, sizes=(batch, -1))
        e0 = self.time_projection(e).unflatten(dim=2, sizes=(6, self.dim))
        if enable_sequence_parallel:
            e0 = shard_sequence_parallel_time_embedding(
                e0, sp_seq_len, action_register_length
            )

        context = self.text_embedding(context)
        if clip_feature is not None:
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        debug_layers = getattr(self, "_dreamzero_current_layer_debug", None)
        debug_layer_output_limit = max(
            0,
            int(
                os.getenv(
                    "DREAMZERO_DEBUG_DIT_LAYER_OUTPUT_LIMIT",
                    os.getenv("DREAMZERO_PRECCOMPARE_LAYER_LIMIT", "5"),
                )
            ),
        )
        updated_kv_caches: list[torch.Tensor] = []
        debug_step_blocks = _debug_block_indices()
        enable_debug_collectives = getattr(
            self, "_dreamzero_enable_debug_collectives", False
        )
        capture_sp_layer_debug = getattr(
            self, "_dreamzero_capture_sp_layer_debug", False
        )
        for block_index, block in enumerate(self.blocks):
            is_debug_step_block = (
                (enable_debug_collectives or capture_sp_layer_debug)
                and block_index in debug_step_blocks
                and block_index < debug_layer_output_limit
            )
            block._dreamzero_debug_row_collectives = is_debug_step_block
            block._dreamzero_debug_sp_gather = (
                capture_sp_layer_debug and is_debug_step_block
            )
            block._dreamzero_debug_sp_action_register_length = (
                action_register_length
            )
            if is_debug_step_block:
                block._dreamzero_debug_steps = []
            elif hasattr(block, "_dreamzero_debug_steps"):
                block._dreamzero_debug_steps = None
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                crossattn_cache=(
                    crossattn_cache[block_index]
                    if crossattn_cache is not None
                    else None
                ),
                current_start_frame=current_start_frame,
                enable_sequence_parallel=enable_sequence_parallel,
            )
            capture_layer = block_index < debug_layer_output_limit and (
                debug_layers is not None or capture_sp_layer_debug
            )
            if capture_layer:
                debug_x = x
                if capture_sp_layer_debug:
                    gathered_debug_x = gather_full_sequence_parallel_tensor(debug_x)
                    if action_register_length is not None:
                        debug_x = remove_redundant_action_register(
                            gathered_debug_x, action_register_length
                        )
                    else:
                        debug_x = flatten_dim_sp_into_sequence(gathered_debug_x)
            if debug_layers is not None and capture_layer:
                layer_debug = {
                    "index": block_index,
                    "x": debug_x.detach().cpu().clone(),
                    "video_token_length": (
                        debug_x.shape[1] - (action_register_length or 0)
                    ),
                    "action_token_length": action_length,
                    "action_register_length": action_register_length,
                }
                debug_steps = getattr(block, "_dreamzero_debug_steps", None)
                if debug_steps:
                    layer_debug["steps"] = debug_steps
                debug_layers.append(layer_debug)
            if hasattr(block, "_dreamzero_debug_steps"):
                block._dreamzero_debug_steps = None
            block._dreamzero_debug_row_collectives = False
            block._dreamzero_debug_sp_gather = False
            block._dreamzero_debug_sp_action_register_length = None
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, sp_seq_len : sp_seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        x_video = x[:, :sp_seq_len]
        if enable_sequence_parallel:
            video_e = e[:, :seq_len]
            sp_rank = get_sp_parallel_rank()
            e_video = video_e[
                :, sp_rank * sp_seq_len : (sp_rank + 1) * sp_seq_len
            ]
        else:
            e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        if enable_sequence_parallel:
            x_video = flatten_dim_sp_into_sequence(
                gather_full_sequence_parallel_tensor(x_video)
            )
        return x_video, action_noise_pred, updated_kv_caches

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor] | None,
        current_start_frame: int,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        enable_sequence_parallel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        if self.model_type == "i2v":
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len
        debug_layer_call_limit = int(
            os.getenv("DREAMZERO_DEBUG_DIT_LAYER_CALL_LIMIT", "1")
        )
        debug_call_index = int(getattr(self, "_dreamzero_debug_call_index", 0))
        debug_action_only = (
            os.getenv("DREAMZERO_DEBUG_DIT_ACTION_ONLY", "0") == "1"
        )
        eligible_debug_call = not debug_action_only or action is not None
        selected_debug_calls = _debug_layer_call_indices()
        selected_debug_call = (
            debug_call_index in selected_debug_calls
            if selected_debug_calls is not None
            else debug_call_index < debug_layer_call_limit
        )
        capture_layer_call = (
            DREAMZERO_DEBUG_DIT_LAYERS
            and eligible_debug_call
            and selected_debug_call
        )
        capture_layers = capture_layer_call and os.getenv("RANK", "0") == "0"
        debug_collectives = capture_layer_call and not enable_sequence_parallel
        if eligible_debug_call:
            self._dreamzero_debug_call_index = debug_call_index + 1
        self._dreamzero_current_layer_debug = [] if capture_layers else None
        self._dreamzero_enable_debug_collectives = debug_collectives
        self._dreamzero_capture_sp_layer_debug = (
            capture_layer_call and enable_sequence_parallel
        )

        if y is not None and self.concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(grid_size=grid_size, start_frame=current_start_frame)

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=current_start_frame,
            enable_sequence_parallel=enable_sequence_parallel,
        )
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        video_noise_pred = self.unpatchify(x_video, grid_size)
        self._dreamzero_last_layer_debug = self._dreamzero_current_layer_debug
        self._dreamzero_current_layer_debug = None
        self._dreamzero_capture_sp_layer_debug = False
        return video_noise_pred, action_noise_pred, updated_kv_caches

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        raise NotImplementedError("DreamZero training forward is outside Phase 2.3")

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        channels = self.out_dim
        grid = grid_size.tolist()
        assert x.shape[1] == math.prod(grid)
        x = x.view(batch, *grid, *self.patch_size, channels)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(
            batch,
            channels,
            *[axis * patch for axis, patch in zip(grid, self.patch_size)],
        )


EntryClass = DreamZeroCausalWanModel

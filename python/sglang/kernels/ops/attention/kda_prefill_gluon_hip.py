"""Fail-closed adapter for the gfx950 Kimi-K3 Gluon KDA prefill kernel."""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention import kda_whole_layer_gluon_hip

_HEADS = 12
_DIM = 128
_HIDDEN = 7168
_QKV = 3 * _HEADS * _DIM
_GATE = _HEADS * _DIM
_CONV = 3 * _HEADS * _DIM
_MIN_ROWS = 1024
_MAX_ROWS = 8192
_MAX_SEQUENCES = 32


def enabled() -> bool:
    return kda_whole_layer_gluon_hip.enabled()


def _tensor(
    tensor: object,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
    )


def can_prepare(layer) -> bool:
    """Check immutable model and weight requirements before enabling prefill."""
    if not kda_whole_layer_gluon_hip.available(layer.dt_bias.device):
        return False
    attn = layer.attn
    device = layer.dt_bias.device
    return (
        layer.use_full_rank_gate
        and not layer.all_reduce_fusion
        and layer.hidden_size == _HIDDEN
        and layer.local_num_heads == _HEADS
        and layer.head_dim == _DIM
        and layer.attn_tp_size == 8
        and tuple(layer.split_sizes) == (_QKV, _GATE)
        and (layer._bfa_fa_size, layer._bfa_b_size) == (_DIM, _HEADS)
        and getattr(attn, "bias", None) is None
        and attn.lower_bound == -5.0
        and layer.o_norm.eps == 1e-5
        and _tensor(
            layer._bfa_w,
            (_DIM + _HEADS + 4, _HIDDEN),
            torch.bfloat16,
            device,
        )
        and _tensor(layer._bfa_f_b_w, (_GATE, _DIM), torch.bfloat16, device)
        and _tensor(attn.conv_weights, (_CONV, 4), torch.float32, device)
        and isinstance(attn.A_log, torch.Tensor)
        and attn.A_log.numel() == _HEADS
        and attn.A_log.dtype == torch.float32
        and attn.A_log.device == device
        and attn.A_log.is_contiguous()
        and _tensor(attn.dt_bias, (_GATE,), torch.float32, device)
        and _tensor(layer.o_norm.weight, (_DIM,), torch.bfloat16, device)
    )


def prefill_layout(lengths: object, prefixes: object, rows: int) -> bool:
    return (
        isinstance(lengths, (tuple, list))
        and 1 <= len(lengths) <= _MAX_SEQUENCES
        and all(type(length) is int and length > 0 for length in lengths)
        and sum(lengths) == rows
        and isinstance(prefixes, (tuple, list))
        and len(prefixes) == len(lengths)
        and all(type(prefix) is int and prefix >= 0 for prefix in prefixes)
    )


def final_state_tracking(metadata: object) -> bool:
    """The generated kernel publishes only final convolution/recurrent state."""
    return metadata is not None and not getattr(metadata, "has_mamba_track_mask", True)


def covered(
    hidden_states: torch.Tensor,
    lengths: object,
    prefixes: object,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_tensor: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
) -> bool:
    """Validate every mutable runtime input before launching a stateful kernel."""
    if not kda_whole_layer_gluon_hip.available(hidden_states.device):
        return False
    rows = hidden_states.shape[0] if hidden_states.ndim == 2 else 0
    sequences = len(lengths) if isinstance(lengths, (tuple, list)) else 0
    device = hidden_states.device
    slots = state.shape[0] if isinstance(state, torch.Tensor) and state.ndim == 4 else 0
    return (
        _MIN_ROWS <= rows <= _MAX_ROWS
        and prefill_layout(lengths, prefixes, rows)
        and _tensor(hidden_states, (rows, _HIDDEN), torch.bfloat16, device)
        and _tensor(state_indices, (sequences,), torch.int32, device)
        and _tensor(cu_seqlens, (sequences + 1,), torch.int32, device)
        and isinstance(prefix_tensor, torch.Tensor)
        and tuple(prefix_tensor.shape) == (sequences,)
        and prefix_tensor.dtype in (torch.int32, torch.int64)
        and prefix_tensor.device == device
        and prefix_tensor.is_contiguous()
        and _tensor(conv_state, (slots, 3, _CONV), torch.bfloat16, device)
        and _tensor(state, (slots, _HEADS, _DIM, _DIM), torch.float32, device)
    )


def run(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    forget_a: torch.Tensor,
    beta: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    lower_bound: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.attention.kda_gluon.kernels.kimi_k3_kda_prefill import (
        fused_kda_prefill,
    )

    return fused_kda_prefill(
        qkv,
        gate,
        forget_a,
        beta,
        forget_weight,
        conv_weight,
        a_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        cu_seqlens,
        has_initial_state,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
    )

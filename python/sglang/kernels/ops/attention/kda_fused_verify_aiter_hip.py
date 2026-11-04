"""Fail-closed adapter for AITER's gfx950 Kimi-K3 fused KDA target verify."""

from __future__ import annotations

import functools
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_hip, print_warning_once

logger = logging.getLogger(__name__)

_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
# [cache, 4608, 3], or SGLang's [cache, 3, 4608] conv cache as transpose(-1, -2).
_CONV_SLOT_STRIDES = ((3, 1), (1, _CHANNELS))
_WARMED: set[tuple] = set()


@functools.cache
def _op():
    try:
        from aiter.ops.flydsl import linear_attention_kernels
    except (ImportError, ModuleNotFoundError) as error:
        print_warning_once(f"Kimi-K3 fused KDA verify unavailable ({error})")
        return None
    op = getattr(linear_attention_kernels, "flydsl_kda_verify_with_f_b", None)
    if op is None:
        print_warning_once(
            "Kimi-K3 fused KDA verify unavailable: this AITER build lacks "
            "flydsl_kda_verify_with_f_b; verify runs the unfused chain"
        )
    return op


def available(
    device: torch.device, state_dtype: torch.dtype, num_draft_tokens: int
) -> bool:
    """Whether the kernel runs a ``num_draft_tokens`` verify over a ``state_dtype`` pool."""
    if not is_hip() or not envs.SGLANG_OPT_HIP_K3_KDA_FUSED_VERIFY.get():
        return False
    op = _op()
    if op is None:
        return False
    from aiter.ops.flydsl.linear_attention_kernels import (
        is_flydsl_kda_decode_supported,
    )

    return (
        is_flydsl_kda_decode_supported(device)
        and state_dtype in op.state_dtypes
        and 1 <= num_draft_tokens <= op.max_tokens
    )


def _dense_slots(tensor: torch.Tensor, inner_strides: tuple[int, ...]) -> bool:
    return (
        tensor.stride()[-len(inner_strides) :] == inner_strides
        and tensor.stride(0) >= tensor.shape[1:].numel()
    )


def covered(
    *,
    f_a: torch.Tensor,
    mixed_qkv: torch.Tensor,
    raw_beta: torch.Tensor,
    output_gate: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    intermediate_states: torch.Tensor | None,
    replay_ring: tuple[torch.Tensor, ...] | None,
    num_draft_tokens: int,
) -> bool:
    """Shape and layout contract of flydsl_kda_verify_with_f_b for a dense verify."""
    tokens = mixed_qkv.shape[0]
    batch = cache_indices.shape[0]
    if (intermediate_states is None) == (replay_ring is None):
        return False
    if replay_ring is not None and replay_ring[0].shape[2] < num_draft_tokens:
        return False
    if intermediate_states is not None and (
        intermediate_states.dtype != state.dtype
        or intermediate_states.shape[1] < num_draft_tokens
        or not _dense_slots(intermediate_states, (_DIM * _DIM, _DIM, 1))
    ):
        return False
    return (
        batch > 0
        and tokens == batch * num_draft_tokens
        and f_a.shape == (tokens, _DIM)
        and f_a.dtype == torch.bfloat16
        and f_a.stride(-1) == 1
        and mixed_qkv.shape == (tokens, _CHANNELS)
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and raw_beta.shape == (1, tokens, _HEADS)
        and raw_beta.dtype == torch.bfloat16
        and raw_beta.stride(-1) == 1
        and output_gate.shape == (tokens, _HEADS, _DIM)
        and output_gate.dtype == torch.bfloat16
        and output_gate.stride(-1) == 1
        and conv_state.shape[1:] == (_CHANNELS, 3)
        and conv_state.stride()[1:] in _CONV_SLOT_STRIDES
        and state.shape[1:] == (_HEADS, _DIM, _DIM)
        and _dense_slots(state, (_DIM * _DIM, _DIM, 1))
        and cache_indices.dtype == torch.int32
        and cache_indices.stride(0) == 1
        and intermediate_indices.shape == (batch,)
        and intermediate_indices.dtype == torch.int32
        and intermediate_indices.stride(0) == 1
        and intermediate_conv_window.shape[1] >= num_draft_tokens
        and intermediate_conv_window.shape[2:] == (_CHANNELS, 3)
        and intermediate_conv_window.dtype == torch.bfloat16
    )


def run(
    *,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    num_draft_tokens: int,
    intermediate_states: torch.Tensor | None = None,
    replay_ring: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    return _op()(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=mixed_qkv,
        conv_weight=conv_weight,
        conv_bias=None,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        cache_indices=cache_indices,
        intermediate_indices=intermediate_indices,
        intermediate_conv_window=intermediate_conv_window,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        num_tokens=num_draft_tokens,
        intermediate_states=intermediate_states,
        replay_ring=replay_ring,
    )


def warmup(
    *,
    f_b_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
    state_dtype: torch.dtype,
    num_draft_tokens: int,
    replay_ring: bool,
    max_batch: int,
) -> None:
    """JIT-build every specialization a verify up to ``max_batch`` sequences uses.

    Builds must not happen during CUDA graph capture; the wave count depends on
    the batch, so each power-of-two batch up to ``max_batch`` is run once.
    """
    device = f_b_weight.device
    key = (
        device.index,
        float(norm_eps),
        float(lower_bound),
        state_dtype,
        num_draft_tokens,
        replay_ring,
        max_batch,
    )
    if key in _WARMED:
        return
    T = num_draft_tokens
    batch = 1
    while True:
        tokens = batch * T
        zeros = functools.partial(torch.zeros, device=device)
        ring = (
            (
                zeros(batch, _HEADS, T, _DIM, dtype=torch.bfloat16),
                zeros(batch, _HEADS, T, _DIM, dtype=torch.bfloat16),
                zeros(batch, _HEADS, T, _DIM, dtype=torch.float32),
                zeros(batch, _HEADS, T, dtype=torch.float32),
            )
            if replay_ring
            else None
        )
        indices = torch.arange(batch, dtype=torch.int32, device=device)
        run(
            f_a=zeros(tokens, _DIM, dtype=torch.bfloat16),
            f_b_weight=f_b_weight,
            mixed_qkv=zeros(tokens, _CHANNELS, dtype=torch.bfloat16),
            conv_weight=conv_weight,
            conv_state=zeros(batch, _CHANNELS, 3, dtype=torch.bfloat16),
            raw_beta=zeros(1, tokens, _HEADS, dtype=torch.bfloat16),
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            state=zeros(batch, _HEADS, _DIM, _DIM, dtype=state_dtype),
            cache_indices=indices,
            intermediate_indices=indices,
            intermediate_conv_window=zeros(batch, T, _CHANNELS, 3, dtype=torch.bfloat16),
            output_gate=zeros(tokens, _HEADS, _DIM, dtype=torch.bfloat16),
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            num_draft_tokens=T,
            intermediate_states=(
                None
                if replay_ring
                else zeros(batch, T, _HEADS, _DIM, _DIM, dtype=state_dtype)
            ),
            replay_ring=ring,
        )
        if batch >= max_batch:
            break
        batch = min(2 * batch, max_batch)
    torch.cuda.synchronize(device)
    _WARMED.add(key)
    logger.info(
        "Kimi-K3 fused KDA verify armed: %d draft tokens, %s recurrent state, %s",
        T,
        state_dtype,
        "ReplaySSM ring" if replay_ring else "state snapshots",
    )

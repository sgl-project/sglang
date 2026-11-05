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
_REJECTION_LOGGED = False


@functools.cache
def _op():
    try:
        from aiter.ops.flydsl.linear_attention_kernels import (
            flydsl_kda_verify_with_f_b,
        )
    except ImportError as error:
        print_warning_once(
            f"Kimi-K3 fused KDA verify unavailable ({error}); "
            "verify runs the unfused chain"
        )
        return None
    return flydsl_kda_verify_with_f_b


def available(
    *, device: torch.device, state_dtype: torch.dtype, num_draft_tokens: int
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
    num_warps: int | None = None,
) -> torch.Tensor:
    # None keeps AITER's batch-based wave count; older verify ops take no num_warps.
    wave_kwargs = {} if num_warps is None else {"num_warps": num_warps}
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
        **wave_kwargs,
    )


def materialize_f_b(layer: torch.nn.Module, f_a: torch.Tensor) -> torch.Tensor:
    """The deferred f_b projection of f_a [tokens, K], as [1, tokens, H, K]."""
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

    f_b_weight = layer._k3_hip_fused_verify_args[0]
    gate = kimi_k3_tiny_gemm(f_a, f_b_weight.flatten(0, 1))
    return gate.unflatten(-1, f_b_weight.shape[:2]).unsqueeze(0)


def run_layer(
    *,
    layer: torch.nn.Module,
    mixed_qkv: torch.Tensor,
    f_a: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    intermediate_state_cache: torch.Tensor | None,
    intermediate_conv_window_cache: torch.Tensor,
    num_draft_tokens: int,
    replay_ring: tuple[torch.Tensor, ...] | None,
) -> torch.Tensor | None:
    """Fused dense chain verify of a layer whose f_b the model deferred, or None."""
    f_b_weight, norm_weight, norm_eps, a_log = layer._k3_hip_fused_verify_args
    tokens = mixed_qkv.shape[0]
    batch = tokens // num_draft_tokens
    kwargs = dict(
        f_a=f_a,
        mixed_qkv=mixed_qkv,
        raw_beta=raw_beta,
        output_gate=layer._k3_onorm_gate[:tokens].view(
            -1, layer.num_v_heads, layer.head_v_dim
        ),
        conv_state=conv_states.transpose(-1, -2),
        state=ssm_states,
        cache_indices=cache_indices[:batch],
        intermediate_indices=intermediate_state_indices[:batch],
        intermediate_conv_window=intermediate_conv_window_cache.transpose(-1, -2),
        intermediate_states=intermediate_state_cache if replay_ring is None else None,
        replay_ring=replay_ring,
        num_draft_tokens=num_draft_tokens,
    )
    if not covered(**kwargs):
        _warn_rejected_once(
            f"tokens={tokens}, draft_tokens={num_draft_tokens}, "
            f"cache_indices={cache_indices.dtype}, "
            f"inter_indices={intermediate_state_indices.dtype}, "
            f"state={tuple(ssm_states.shape)}/{ssm_states.dtype}"
        )
        return None
    return run(
        **kwargs,
        f_b_weight=f_b_weight,
        conv_weight=layer.conv_weights,
        A_log=a_log,
        dt_bias=layer.dt_bias,
        lower_bound=float(layer.lower_bound),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


def _warn_rejected_once(batch: str) -> None:
    # Once per process, not per batch shape: the first shape is enough to debug.
    global _REJECTION_LOGGED
    if _REJECTION_LOGGED:
        return
    _REJECTION_LOGGED = True
    logger.warning(
        "K3 HIP fused KDA verify rejected a batch: %s; verify runs the unfused chain "
        "(logged once)",
        batch,
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
) -> None:
    """JIT-build every wave count a ``num_draft_tokens`` verify can dispatch.

    Builds must not happen during CUDA graph capture. They are keyed on the wave
    count, not the batch, so one batch-1 call per wave count builds them all.
    """
    device = f_b_weight.device
    key = (
        device.index,
        float(norm_eps),
        float(lower_bound),
        state_dtype,
        num_draft_tokens,
        replay_ring,
    )
    if key in _WARMED:
        return
    op = _op()
    try:
        builds = [(1, num_warps) for num_warps in op.num_warps]
    except AttributeError:
        # Without the num_warps tag, AITER picks 8 waves up to batch 16 and 4 above.
        builds = [(1, None), (17, None)]
    for batch, num_warps in builds:
        _run_zeros(
            batch=batch,
            num_warps=num_warps,
            f_b_weight=f_b_weight,
            conv_weight=conv_weight,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            state_dtype=state_dtype,
            num_draft_tokens=num_draft_tokens,
            replay_ring=replay_ring,
        )
    torch.cuda.synchronize(device)
    _WARMED.add(key)
    logger.info(
        "Kimi-K3 fused KDA verify armed: %d draft tokens, %s recurrent state, %s",
        num_draft_tokens,
        state_dtype,
        "ReplaySSM ring" if replay_ring else "state snapshots",
    )


def _run_zeros(
    *,
    batch: int,
    num_warps: int | None,
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
) -> None:
    """One verify over zero inputs, to JIT-build its specialization."""
    zeros = functools.partial(torch.zeros, device=f_b_weight.device)
    tokens = batch * num_draft_tokens
    ring = (
        (
            zeros(batch, _HEADS, num_draft_tokens, _DIM, dtype=torch.bfloat16),
            zeros(batch, _HEADS, num_draft_tokens, _DIM, dtype=torch.bfloat16),
            zeros(batch, _HEADS, num_draft_tokens, _DIM, dtype=torch.float32),
            zeros(batch, _HEADS, num_draft_tokens, dtype=torch.float32),
        )
        if replay_ring
        else None
    )
    snapshots = (
        None
        if replay_ring
        else zeros(batch, num_draft_tokens, _HEADS, _DIM, _DIM, dtype=state_dtype)
    )
    indices = torch.arange(batch, dtype=torch.int32, device=f_b_weight.device)
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
        intermediate_conv_window=zeros(
            batch, num_draft_tokens, _CHANNELS, 3, dtype=torch.bfloat16
        ),
        output_gate=zeros(tokens, _HEADS, _DIM, dtype=torch.bfloat16),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        num_draft_tokens=num_draft_tokens,
        intermediate_states=snapshots,
        replay_ring=ring,
        num_warps=num_warps,
    )

"""Opt-in Cake kernel stages for the MiniMax-H3 diffusion transformer.

``SGLANG_CAKE_ROUTES=minimax_h3_diffusion`` lets four stages of every
``MiniMaxH3DiTBlock`` take the Cake KernelSpecs registered in
``sglang.kernels.ops.diffusion.cake`` instead of the stock layer sequence:

* ``bf16_pre_attention`` -- ``norm1`` + indexed AdaLN + fused QKV GEMM + per-head
  Q/K RMSNorm + partial NeoX RoPE (``diffusion.minimax_h3_bf16_pre_attention``).
* ``varlen_attention`` -- the packed-varlen non-causal attention at the BCG
  break point (``diffusion.minimax_h3_varlen_attention``).
* ``out_proj`` -- output projection fused with the first gated residual
  (``diffusion.minimax_h3_out_proj``).
* ``fc1_swiglu`` -- ``norm2`` + indexed AdaLN + fused FC1 GEMM + SwiGLU
  (``diffusion.minimax_h3_fc1_swiglu``).

Every stage is admitted per call by the adapter's ``supports_*`` on the exact
tensors the engine is about to use; a stage that is not admitted (route off,
other arch, TP > 1 weight shards, quantized weights, an operand outside the
kernel contract, ...) returns ``None`` and the block runs its stock code,
unchanged.  The first "taken" and the first "fallback" of each distinct reason
per stage are logged so an e2e run can prove which kernel executed.

CUDA graphs (breakable CUDA graph runner): the three GEMM stages run inside the
captured segments.  They are one-shot kernels with caller-owned outputs and no
host sync once their JIT module is built, so a stage is used under capture only
after it completed one eager launch (the warm-up forward); otherwise the graph
keeps the stock kernels for that stage (logged once).  The attention stage runs
in the eager break (``_minimax_h3_attention_core_bcg``); inside a captured
region (token refiner, ``bcg_breakpoint=False``) it always falls back because
the one-shot FlashInfer entry builds its plan tables at call time.

The engine's operands are handed to the kernels as they are -- no copy, cast
or gather runs between the stages:

* AdaLN / gate tables are the ``[rows, 5376]`` column chunks of the
  ``[rows, 6 * 5376]`` modulation projection (any ``rows >= 1``, row stride
  ``6 * 5376``); the kernels read them through the row pitch.
* ``adaln_index`` / ``gate_index`` is the engine's int64 ``combined_indices``.
* RoPE takes the engine's ``(cos_sin_cache [S, 96], positions int64 [T])``
  pair and gathers the cache rows on the device.
* Attention consumes the strided ``[T, H, 128]`` views of the fused QKV
  projection (row stride ``3 * 7168``) or of the Cake pre-attention pack
  ``[T, H, 3, 128]`` in place and writes a contiguous output.
* ``norm1.eps`` and the Q/K-norm ``eps`` are passed separately.

Operands outside the kernel contract (a non-unit last stride, a row pitch that
is not a multiple of 8 elements, an int32 index, a mismatched row count, ...)
are rejected by the adapter's admission and the stage falls back to the stock
kernels with one log line per reason.

Not covered (stock path kept): Ulysses / ring sequence parallelism (the Cake
pack layout is the Ulysses send buffer, but the engine's all-to-all consumes
separate q/k/v), the VDN hybrid-window, video-sparse, sub-block-sparse and cube
sparse attention backends, MXFP8 / NVFP4 / FP8 quantized weights (the model
has no NVFP4/MXFP8 weight path at the Cake layouts), the SM120 quantized
attention entries (quantized operands change numerics), the dense attention
entry (needs a pre-scaled query copy; SM120 performance target), and TP > 1.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Callable, Optional, Sequence, Tuple

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

ROUTE = "minimax_h3_diffusion"
STAGE_PRE_ATTENTION = "bf16_pre_attention"
STAGE_ATTENTION = "varlen_attention"
STAGE_OUT_PROJ = "out_proj"
STAGE_FC1 = "fc1_swiglu"
STAGES = (STAGE_PRE_ATTENTION, STAGE_ATTENTION, STAGE_OUT_PROJ, STAGE_FC1)

_LOG_PREFIX = "[cake-route]"
_QKV_KINDS = 3
# FlashInfer validates on the host before launching; these are its rejections.
_CAKE_ERRORS = (RuntimeError, ValueError, NotImplementedError)

# (stage, event, reason kind) triples already logged; keeps the per-call path
# free of I/O while still naming every distinct fallback reason once.
_logged: set[tuple[str, str, str]] = set()


def _reason_kind(detail: str) -> str:
    """Digit-normalised prefix of a fallback detail (the text before the tensor
    dump), so one line is emitted per distinct reason, not per shape."""
    return re.sub(r"\d+", "N", detail.split(":", 1)[0])[:64]


# Per (stage, shape key) admission verdict; False after a FlashInfer rejection.
_verdicts: dict[tuple, bool] = {}
# Stages that completed one eager launch (safe to use under graph capture).
_warmed: set[str] = set()


def route_enabled() -> bool:
    return cake_route_enabled(ROUTE)


def reset_state_for_tests() -> None:
    _logged.clear()
    _verdicts.clear()
    _warmed.clear()


def _log_once(stage: str, event: str, detail: str) -> None:
    key = (stage, event, _reason_kind(detail) if event == "fallback" else "")
    if key in _logged:
        return
    _logged.add(key)
    if event == "taken":
        logger.info(
            "%s %s/%s: Cake kernel selected (%s)", _LOG_PREFIX, ROUTE, stage, detail
        )
    else:
        logger.info(
            "%s %s/%s: fallback to stock kernels (%s)",
            _LOG_PREFIX,
            ROUTE,
            stage,
            detail,
        )


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _summary(**tensors: Optional[torch.Tensor]) -> str:
    parts = []
    for name, t in tensors.items():
        if t is None:
            parts.append(f"{name}=None")
        else:
            text = f"{name}={tuple(t.shape)}/{str(t.dtype).removeprefix('torch.')}"
            if not t.is_contiguous():
                text += f"/stride{tuple(t.stride())}"
            parts.append(text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Lazy (admission, forwarder) pairs: FlashInfer is imported only when called.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _pre_attention_kernels() -> Tuple[Callable[..., bool], Callable[..., object]]:
    from sglang.kernels.cake_kernels.diffusion_minimax_h3_pre_attention import (
        supports_minimax_h3_bf16_pre_attention,
    )
    from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_bf16_pre_attention

    return supports_minimax_h3_bf16_pre_attention, cake_minimax_h3_bf16_pre_attention


@functools.lru_cache(maxsize=None)
def _attention_kernels() -> Tuple[Callable[..., bool], Callable[..., object]]:
    from sglang.kernels.cake_kernels.diffusion_minimax_h3_attention import (
        supports_minimax_h3_varlen_attention,
    )
    from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_varlen_attention

    return supports_minimax_h3_varlen_attention, cake_minimax_h3_varlen_attention


@functools.lru_cache(maxsize=None)
def _out_proj_kernels() -> Tuple[Callable[..., bool], Callable[..., object]]:
    from sglang.kernels.cake_kernels.diffusion_minimax_h3_proj import (
        supports_minimax_h3_out_proj,
    )
    from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_out_proj

    return supports_minimax_h3_out_proj, cake_minimax_h3_out_proj


@functools.lru_cache(maxsize=None)
def _fc1_kernels() -> Tuple[Callable[..., bool], Callable[..., object]]:
    from sglang.kernels.cake_kernels.diffusion_minimax_h3_proj import (
        supports_minimax_h3_fc1_swiglu,
    )
    from sglang.kernels.ops.diffusion.cake import cake_minimax_h3_fc1_swiglu

    return supports_minimax_h3_fc1_swiglu, cake_minimax_h3_fc1_swiglu


# ---------------------------------------------------------------------------
# Shared admission bookkeeping
# ---------------------------------------------------------------------------


def _tensor_key(t: Optional[torch.Tensor]) -> tuple:
    if t is None:
        return (None,)
    return (tuple(t.shape), tuple(t.stride()), t.dtype, str(t.device))


def _stage_open(stage: str, key: tuple, *, in_graph: bool) -> bool:
    """Route on, no cached rejection, and graph-capture safe for this stage."""
    if not route_enabled():
        return False
    if _verdicts.get(key) is False:
        return False
    if in_graph and _capturing() and stage not in _warmed:
        _log_once(
            stage,
            "fallback",
            "shape first seen inside CUDA-graph capture before an eager warm-up "
            "launch; the stock kernels are captured for this graph",
        )
        return False
    return True


def _reject(stage: str, key: tuple, detail: str) -> None:
    _verdicts[key] = False
    _log_once(stage, "fallback", detail)


def _rope_pair_ok(
    cos_sin_cache: object, positions: object, rows: int, device: torch.device
) -> bool:
    """The engine's ``(cos_sin_cache [S, 96], positions int64 [T])`` pair."""
    return (
        isinstance(cos_sin_cache, torch.Tensor)
        and isinstance(positions, torch.Tensor)
        and cos_sin_cache.ndim == 2
        and positions.ndim == 1
        and positions.dtype == torch.int64
        and int(positions.shape[0]) == rows
        and positions.device == device
        and cos_sin_cache.device == device
    )


# ---------------------------------------------------------------------------
# Stage 1: norm1 + AdaLN + QKV projection + QK-norm + RoPE
# ---------------------------------------------------------------------------


def pre_attention(
    x: torch.Tensor,
    *,
    x_norm_weight: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight: Optional[torch.Tensor],
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
    eps: float,
    qk_eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Fused pre-attention; ``(q, k, v)`` ``[T, H, 128]`` views or ``None``.

    The returned tensors are already Q/K-normalised and RoPE-rotated, i.e. what
    ``MiniMaxH3Attention.forward`` hands to the attention core.  ``None`` means
    "run the stock norm1 / AdaLN / qkv_proj / qk-norm / RoPE sequence".
    ``eps`` is ``norm1``'s epsilon, ``qk_eps`` the Q/K-norm epsilon.
    """
    stage = STAGE_PRE_ATTENTION
    if (
        qkv_weight is None
        or rope_cache is None
        or x.ndim != 2
        or qkv_weight.ndim != 2
        or q_norm_weight.ndim != 1
    ):
        return None
    key = (
        stage,
        _tensor_key(x),
        _tensor_key(qkv_weight),
        _tensor_key(adaln_scale),
        _tensor_key(adaln_shift),
        _tensor_key(adaln_index),
        float(eps),
        float(qk_eps),
    )
    if not _stage_open(stage, key, in_graph=True):
        return None
    head_dim = int(q_norm_weight.shape[0])
    heads = int(qkv_weight.shape[0]) // (_QKV_KINDS * head_dim)
    if heads <= 0 or heads * _QKV_KINDS * head_dim != int(qkv_weight.shape[0]):
        _reject(stage, key, f"qkv weight rows {tuple(qkv_weight.shape)} not 3*H*D")
        return None
    cos_sin_cache, positions = rope_cache
    if not _rope_pair_ok(cos_sin_cache, positions, int(x.shape[0]), x.device):
        _reject(
            stage,
            key,
            "rope cache is not a (cos_sin_cache [S, 96], int64 positions [T]) "
            f"pair: {_summary(cache=cos_sin_cache, positions=positions)}",
        )
        return None
    supports, forward = _pre_attention_kernels()
    # Destination-major pack for ulysses_degree=1: [1, T, H, 3, D].
    out = torch.empty(
        (1, int(x.shape[0]), heads, _QKV_KINDS, head_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    detail = _summary(x=x, qkv_weight=qkv_weight, adaln=adaln_scale, index=adaln_index)
    if _verdicts.get(key) is None:
        admitted = bool(
            supports(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                qkv_weight,
                q_norm_weight,
                k_norm_weight,
                cos_sin_cache,
                ulysses_degree=1,
                out=out,
                eps=eps,
                qk_eps=qk_eps,
                rope_positions=positions,
            )
        )
        if not admitted:
            _reject(stage, key, f"adapter admission rejected: {detail}")
            return None
        _verdicts[key] = True
    try:
        forward(
            x,
            x_norm_weight,
            adaln_scale,
            adaln_shift,
            adaln_index,
            qkv_weight,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            ulysses_degree=1,
            out=out,
            eps=eps,
            qk_eps=qk_eps,
            rope_positions=positions,
        )
    except _CAKE_ERRORS as error:
        _reject(stage, key, f"FlashInfer rejected the call ({error}): {detail}")
        return None
    if not _capturing():
        _warmed.add(stage)
    _log_once(stage, "taken", detail)
    qkv = out[0]  # [T, H, 3, D]
    return qkv[:, :, 0, :], qkv[:, :, 1, :], qkv[:, :, 2, :]


# ---------------------------------------------------------------------------
# Stage 2: packed-varlen attention (eager BCG break point)
# ---------------------------------------------------------------------------


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    cu_seqlens_host: Optional[Sequence[int]],
    softmax_scale: float,
) -> Optional[torch.Tensor]:
    """Cake varlen attention; BF16 ``[T, H, 128]`` or ``None`` (stock backend)."""
    stage = STAGE_ATTENTION
    if q.ndim != 3:
        return None
    key = (
        stage,
        _tensor_key(q),
        _tensor_key(k),
        _tensor_key(v),
        _tensor_key(cu_seqlens),
    )
    if not _stage_open(stage, key, in_graph=False):
        return None
    if _capturing():
        # One-shot entry: plan tables are built from cu_seqlens at call time.
        _log_once(
            stage,
            "fallback",
            "attention core is inside a CUDA-graph capture (no eager break point)",
        )
        return None
    supports, forward = _attention_kernels()
    # The strided q/k/v views (fused QKV chunks or pre-attention pack slices)
    # are consumed in place; the kernel writes a contiguous output.
    detail = _summary(q=q, k=k, v=v, cu_seqlens=cu_seqlens)
    if _verdicts.get(key) is None:
        if not bool(supports(q, k, v, cu_seqlens)):
            _reject(stage, key, f"adapter admission rejected: {detail}")
            return None
        _verdicts[key] = True
    try:
        out = forward(
            q,
            k,
            v,
            cu_seqlens,
            softmax_scale=softmax_scale,
            cu_seqlens_host=(
                None if cu_seqlens_host is None else tuple(cu_seqlens_host)
            ),
        )
    except _CAKE_ERRORS as error:
        _reject(stage, key, f"FlashInfer rejected the call ({error}): {detail}")
        return None
    _warmed.add(stage)
    _log_once(stage, "taken", detail)
    return out


# ---------------------------------------------------------------------------
# Stage 3: out-projection fused with the gated residual
# ---------------------------------------------------------------------------


def out_proj_gated_residual(
    attn_out: torch.Tensor,
    *,
    o_weight: Optional[torch.Tensor],
    gate: torch.Tensor,
    gate_index: torch.Tensor,
    residual: torch.Tensor,
) -> Optional[torch.Tensor]:
    """``residual + gate[idx] * (attn_out @ W^T)`` as one Cake launch, or ``None``.

    ``attn_out`` is the attention-core output ``[T, H, 128]`` (before the
    engine's ``reshape`` + ``out_proj``); the Ulysses receive layout at
    ``P = 1`` is its free ``[1, T, H, 128]`` view.
    """
    stage = STAGE_OUT_PROJ
    if o_weight is None or attn_out.ndim != 3 or not attn_out.is_contiguous():
        return None
    key = (
        stage,
        _tensor_key(attn_out),
        _tensor_key(o_weight),
        _tensor_key(gate),
        _tensor_key(gate_index),
        _tensor_key(residual),
    )
    if not _stage_open(stage, key, in_graph=True):
        return None
    supports, forward = _out_proj_kernels()
    packed = attn_out.unsqueeze(0)
    detail = _summary(attn_out=packed, o_weight=o_weight, gate=gate, index=gate_index)
    if _verdicts.get(key) is None:
        if not bool(supports(packed, o_weight, gate, gate_index, residual)):
            _reject(stage, key, f"adapter admission rejected: {detail}")
            return None
        _verdicts[key] = True
    out = torch.empty_like(residual)
    try:
        forward(packed, o_weight, gate, gate_index, residual, out=out)
    except _CAKE_ERRORS as error:
        _reject(stage, key, f"FlashInfer rejected the call ({error}): {detail}")
        return None
    if not _capturing():
        _warmed.add(stage)
    _log_once(stage, "taken", detail)
    return out


# ---------------------------------------------------------------------------
# Stage 4: norm2 + AdaLN + FC1 + SwiGLU
# ---------------------------------------------------------------------------


def fc1_swiglu(
    x: torch.Tensor,
    *,
    x_norm_weight: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_index: torch.Tensor,
    fc1_weight: Optional[torch.Tensor],
    eps: float,
) -> Optional[torch.Tensor]:
    """``silu(gate) * up`` of the modulated ``norm2(x)``; ``[T, FFN]`` or ``None``.

    The result is what the stock path hands to ``mlp.fc2``.
    """
    stage = STAGE_FC1
    if fc1_weight is None or x.ndim != 2 or fc1_weight.ndim != 2:
        return None
    key = (
        stage,
        _tensor_key(x),
        _tensor_key(fc1_weight),
        _tensor_key(adaln_scale),
        _tensor_key(adaln_shift),
        _tensor_key(adaln_index),
        float(eps),
    )
    if not _stage_open(stage, key, in_graph=True):
        return None
    supports, forward = _fc1_kernels()
    detail = _summary(x=x, fc1_weight=fc1_weight, adaln=adaln_scale, index=adaln_index)
    if _verdicts.get(key) is None:
        if not bool(
            supports(
                x,
                x_norm_weight,
                adaln_scale,
                adaln_shift,
                adaln_index,
                fc1_weight,
                eps=eps,
            )
        ):
            _reject(stage, key, f"adapter admission rejected: {detail}")
            return None
        _verdicts[key] = True
    # Caller-owned buffers keep the launch allocation-free for graph capture.
    out = torch.empty(
        (int(x.shape[0]), int(fc1_weight.shape[0]) // 2),
        dtype=torch.bfloat16,
        device=x.device,
    )
    workspace = torch.empty_like(x)
    try:
        forward(
            x,
            x_norm_weight,
            adaln_scale,
            adaln_shift,
            adaln_index,
            fc1_weight,
            out=out,
            workspace=workspace,
            eps=eps,
        )
    except _CAKE_ERRORS as error:
        _reject(stage, key, f"FlashInfer rejected the call ({error}): {detail}")
        return None
    if not _capturing():
        _warmed.add(stage)
    _log_once(stage, "taken", detail)
    return out


__all__ = (
    "ROUTE",
    "STAGES",
    "STAGE_ATTENTION",
    "STAGE_FC1",
    "STAGE_OUT_PROJ",
    "STAGE_PRE_ATTENTION",
    "fc1_swiglu",
    "out_proj_gated_residual",
    "pre_attention",
    "reset_state_for_tests",
    "route_enabled",
    "varlen_attention",
)

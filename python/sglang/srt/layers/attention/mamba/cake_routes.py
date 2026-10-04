"""Opt-in Cake routes for the Mamba2 mixer (``SGLANG_CAKE_ROUTES``).

Two engine call sites in :mod:`sglang.srt.layers.attention.mamba.mamba` can
take the Cake KernelSpecs registered in ``sglang.kernels.ops.mamba.cake``:

* ``mamba_ssd_prefill`` -- the SSD combined prefill
  (``mamba_chunk_scan_combined``) -> ``mamba.ssd_combined_fwd``.
* ``mamba_ssu`` -- the selective state update for decode and for the
  speculative target-verify batch -> ``mamba.selective_state_update``.

Each dispatcher below receives the stock kernel and *exactly* the keyword
arguments the mixer passes to it today.  With the route off, or when the
adapter admission rejects the real tensors, the stock kernel is called with
those arguments unchanged.  The first "taken" and the first "fallback" of each
distinct reason per route are logged so an e2e run can prove which kernel
executed and why a batch did not take the Cake kernel.

FlashInfer contract constraints that shape the wiring (see the adapter
:mod:`sglang.kernels.cake_kernels.mamba` for the full list):

* SSD: chunk 128 (the engine builds its chunk metadata for the model's
  ``mamba_chunk_size``, 256 for Nemotron-H, so :class:`Mamba2Metadata`
  carries a second, chunk-128 ``chunk_indices`` / ``chunk_offsets`` pair when
  the route is on), any packed token count (the kernel handles a partial last
  chunk), BF16 / FP16 / FP32 state (every ``--mamba-ssm-dtype``), varlen
  without a prefix passes ``initial_states=None`` plus the sequence count, and
  the engine's token-major ``[1, S, H, 64]`` output buffer is the kernel's
  ``out`` (no copy).
* SSD radix-cache tracking (``track_seq_idx`` / ``track_end_locs``): the
  Cake runner exposes selective checkpoints only at logical chunk ends
  (``checkpoint_token_indices`` + ``checkpoint_state_slots``); the mapping
  from the engine's track end locations onto that contract is not wired
  here, so a batch that requests track states keeps the stock kernel.
* SSU: Cake runs only on its promoted rows: ``(dim, dstate) = (128, 128)``
  for ``T in {1, 2}`` and the BF16-state MTP cache row ``(64, 128, T=6)``
  with ``disable_state_update`` + intermediate buffer (Nemotron-H / granite
  target-verify with six draft tokens).  Those rows take FP32 ``dt`` / ``D``
  / ``dt_bias`` (per-head broadcasts) and int64 indices; the engine holds
  BF16 broadcasts and int32 indices, so the route casts the broadcast base
  (``[H]`` parameters are cached once, ``dt`` / indices per call).  A
  ``T = 1`` decode of a ``headdim = 64`` model (Nemotron-H, granite) has no
  promoted row and stays on the stock kernel without consulting FlashInfer.
* A shape first seen inside CUDA-graph capture is not routed (the Cake SSU
  is nvcc-built on first use); the warm-up forward before capture admits it.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import re
from typing import Callable, Optional, Sequence

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

CAKE_ROUTE_SSD_PREFILL = "mamba_ssd_prefill"
CAKE_ROUTE_SSU = "mamba_ssu"
SSD_CHUNK_SIZE = 128
SSD_HEADDIM = 64
SSD_DSTATE = 128
_CAKE_LOG_PREFIX = "[cake-route]"

# (route, event, reason kind) triples already logged; keeps the per-call path
# free of I/O while still naming every distinct fallback reason once.
_cake_route_logged: set[tuple[str, str, str]] = set()


def _reason_kind(detail: str) -> str:
    """Digit-normalised prefix of a fallback detail (the text before the tensor
    dump), so one line is emitted per distinct reason, not per shape."""
    return re.sub(r"\d+", "N", detail.split(":", 1)[0])[:64]


# Shape keys FlashInfer rejected with NotImplementedError; skipped afterwards.
_cake_route_rejected: set[tuple] = set()
# SSU shape keys admitted (and run) eagerly; only these are routed under capture.
_ssu_warm: set[tuple] = set()
# SSU admission is a pure function of the shape key; memoised per key.
_ssu_admission: dict[tuple, bool] = {}
# FP32 copies of the per-head parameters (D / dt_bias) keyed by storage.
_fp32_params: dict[tuple, torch.Tensor] = {}


def _log_cake_route_once(route: str, event: str, detail: str) -> None:
    key = (route, event, _reason_kind(detail) if event == "fallback" else "")
    if key in _cake_route_logged:
        return
    _cake_route_logged.add(key)
    if event == "taken":
        logger.info("%s %s: Cake kernel selected (%s)", _CAKE_LOG_PREFIX, route, detail)
    else:
        logger.info(
            "%s %s: fallback to stock kernel (%s)", _CAKE_LOG_PREFIX, route, detail
        )


def reset_cake_route_state_for_tests() -> None:
    _cake_route_logged.clear()
    _cake_route_rejected.clear()
    _ssu_warm.clear()
    _ssu_admission.clear()
    _fp32_params.clear()


@functools.lru_cache(maxsize=None)
def _cake_ssd_kernels() -> tuple[Callable[..., bool], Callable[..., tuple]]:
    """Lazy (admission, forwarder) pair for ``mamba.ssd_combined_fwd``."""
    from sglang.kernels.cake_kernels.mamba import supports_ssd_combined
    from sglang.kernels.ops.mamba.cake import cake_ssd_combined_fwd

    return supports_ssd_combined, cake_ssd_combined_fwd


@functools.lru_cache(maxsize=None)
def _cake_ssu_kernels() -> tuple[Callable[..., bool], Callable[..., torch.Tensor]]:
    """Lazy (admission, forwarder) pair for ``mamba.selective_state_update``."""
    from sglang.kernels.cake_kernels.mamba import supports_selective_state_update
    from sglang.kernels.ops.mamba.cake import cake_selective_state_update

    return supports_selective_state_update, cake_selective_state_update


def _tensor_summary(**tensors: Optional[torch.Tensor]) -> str:
    parts = []
    for name, t in tensors.items():
        if t is None:
            parts.append(f"{name}=None")
        else:
            parts.append(
                f"{name}={tuple(t.shape)}/{str(t.dtype).removeprefix('torch.')}"
            )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# SSD combined prefill
# ---------------------------------------------------------------------------


class _TrackStatesInPlace:
    """Sentinel the SSD route returns in the ``track_states`` slot when the Cake
    runner wrote the chunk-unaligned radix-cache track rows itself (selective
    checkpoints straight into the state pool); the backend then skips its own
    copy of those rows and only performs the chunk-aligned slot copies."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "SSD_TRACK_STATES_IN_PLACE"


SSD_TRACK_STATES_IN_PLACE = _TrackStatesInPlace()


@dataclasses.dataclass(frozen=True)
class CakeTrackCheckpoints:
    """The engine's prefill track rows expressed for the Cake SSD runner.

    ``boundaries`` are the absolute packed token positions the chunk-128
    metadata must expose as logical chunk ends; ``token_indices`` /
    ``state_slots`` the runner's per-sequence int32 checkpoint pair (``-1``
    where no checkpoint is wanted), both ``None`` when no row needs a
    checkpoint (every tracked row is chunk-aligned and takes the engine's
    final-state slot copy, the default-configuration case).
    """

    boundaries: tuple[int, ...]
    token_indices: Optional[torch.Tensor]
    state_slots: Optional[torch.Tensor]


def cake_ssd_track_checkpoints(
    track_mask: Sequence[bool],
    track_lens: Sequence[int],
    extend_lens: Sequence[int],
    prefix_lens: Sequence[int],
    state_chunk_size: int,
    track_slots: torch.Tensor,
    device: torch.device,
) -> Optional[CakeTrackCheckpoints]:
    """Map the engine's prefill track rows onto Cake selective checkpoints.

    The radix cache snapshots each tracked sequence's SSM state at its last
    ``state_chunk_size`` boundary (``build_prefill_track_plan``). Chunk-aligned
    rows take the sequence's final state through the engine's slot copy (no
    checkpoint). For chunk-unaligned rows the stock kernel reads its chunk grid
    or runs a recompute pass; the Cake runner instead writes one checkpoint per
    sequence at an exclusive absolute token boundary, provided that boundary
    is a logical chunk end of the chunk-128 metadata.  ``None`` when a tracked
    row wants the state before its first token (chunk index 0): the stock
    kernel owns that case, so the batch stays on the stock path.
    """
    from sglang.srt.layers.attention.mamba.prefill_track_metadata import (
        build_prefill_track_plan,
    )

    plan = build_prefill_track_plan(
        list(track_mask),
        list(track_lens),
        list(extend_lens),
        list(prefix_lens),
        state_chunk_size,
        mamba2=True,
    )
    starts = list(itertools.accumulate(extend_lens, initial=0))
    tokens = [-1] * len(extend_lens)
    for row in plan.unaligned_rows:
        chunk = plan.chunk_indices[row]
        if chunk <= 0:
            return None
        tokens[row] = starts[row] + chunk * state_chunk_size
    if not plan.unaligned_rows:
        return CakeTrackCheckpoints((), None, None)
    boundaries = tuple(t for t in tokens if t >= 0)
    token_indices = torch.tensor(tokens, dtype=torch.int32).to(
        device, non_blocking=True
    )
    slots = torch.full((len(extend_lens),), -1, dtype=torch.int32, device=device)
    rows = torch.tensor(plan.unaligned_rows, dtype=torch.int64).to(
        device, non_blocking=True
    )
    slots[rows] = track_slots.index_select(0, rows).to(torch.int32)
    return CakeTrackCheckpoints(boundaries, token_indices, slots)


def cake_ssd_chunk_metadata(
    extend_seq_lens: Sequence[int],
    device: torch.device,
    extra_boundaries: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logical-chunk metadata of a packed prefill batch for the Cake chunk size.

    Same semantics as ``Mamba2Metadata._query_start_loc_to_chunk_indices_offsets``
    (a logical chunk starts at every physical 128-token chunk start and at
    every sequence start) computed from the host-side sequence lengths, so
    the metadata build does not synchronise the stream.  ``extra_boundaries``
    are further absolute token positions that must be logical chunk ends (the
    radix-cache track positions the Cake runner checkpoints; see
    :func:`cake_ssd_track_checkpoints`); one already on the 128 grid or at a
    sequence start adds nothing.  Returns int32 ``(chunk_indices,
    chunk_offsets)`` on ``device``.
    """
    starts: set[int] = {int(b) for b in extra_boundaries}
    start = 0
    for length in extend_seq_lens:
        starts.add(start)
        start += int(length)
    total = start
    chunk_indices: list[int] = []
    chunk_offsets: list[int] = []
    for chunk in range(-(-total // SSD_CHUNK_SIZE)):
        lo, hi = chunk * SSD_CHUNK_SIZE, (chunk + 1) * SSD_CHUNK_SIZE
        for offset in sorted({0} | {b - lo for b in starts if lo < b < hi}):
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        torch.tensor(chunk_indices, dtype=torch.int32).to(device, non_blocking=True),
        torch.tensor(chunk_offsets, dtype=torch.int32).to(device, non_blocking=True),
    )


def ssd_prefill(
    stock: Callable[..., tuple],
    *,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    seq_idx: torch.Tensor,
    chunk_indices: Optional[torch.Tensor],
    chunk_offsets: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    initial_states: Optional[torch.Tensor],
    track_seq_idx: Optional[torch.Tensor],
    track_end_locs: Optional[torch.Tensor],
    out: torch.Tensor,
    state_dtype: torch.dtype,
    cake_chunk_indices: Optional[torch.Tensor],
    cake_chunk_offsets: Optional[torch.Tensor],
    track_states_out: Optional[torch.Tensor] = None,
    cake_track_checkpoints: Optional[CakeTrackCheckpoints] = None,
) -> tuple:
    """Run the prefill SSD scan; ``mamba_ssd_prefill`` may take the Cake kernel.

    Returns the stock ``(intermediate_states, varlen_state, track_states)``
    triple.  ``cake_chunk_indices`` / ``cake_chunk_offsets`` are the chunk-128
    metadata prepared once per forward by :class:`Mamba2Metadata`;
    ``cake_track_checkpoints`` the radix-cache track rows mapped onto Cake
    checkpoints (same place) and ``track_states_out`` the layer's SSM state
    pool they are written into.  For a tracked batch the Cake path returns
    :data:`SSD_TRACK_STATES_IN_PLACE` in the ``track_states`` slot: the
    chunk-unaligned rows (if any) were checkpointed in-kernel and only the
    engine's aligned slot copies remain.  Everything else is the stock call's
    argument list.
    """
    stock_kwargs = dict(
        chunk_size=chunk_size,
        D=D,
        z=None,
        dt_bias=dt_bias,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        initial_states=initial_states,
        return_varlen_states=True,
        return_final_states=False,
        return_track_states=True,
        track_seq_idx=track_seq_idx,
        track_end_locs=track_end_locs,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        out=out,
        state_dtype=state_dtype,
    )
    if not cake_route_enabled(CAKE_ROUTE_SSD_PREFILL):
        return stock(x, dt, A, B, C, **stock_kwargs)
    result = _cake_ssd_prefill(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        initial_states=initial_states,
        track_seq_idx=track_seq_idx,
        out=out,
        state_dtype=state_dtype,
        cake_chunk_indices=cake_chunk_indices,
        cake_chunk_offsets=cake_chunk_offsets,
        track_states_out=track_states_out,
        cake_track_checkpoints=cake_track_checkpoints,
    )
    if result is None:
        return stock(x, dt, A, B, C, **stock_kwargs)
    return result


def _cake_ssd_prefill(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    seq_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_states: Optional[torch.Tensor],
    track_seq_idx: Optional[torch.Tensor],
    out: torch.Tensor,
    state_dtype: torch.dtype,
    cake_chunk_indices: Optional[torch.Tensor],
    cake_chunk_offsets: Optional[torch.Tensor],
    track_states_out: Optional[torch.Tensor] = None,
    cake_track_checkpoints: Optional[CakeTrackCheckpoints] = None,
) -> Optional[tuple]:
    """``None`` means "use the stock call"."""
    route = CAKE_ROUTE_SSD_PREFILL
    seqlen = x.shape[1]
    nheads = x.shape[2]
    detail = _tensor_summary(x=x, dt=dt, B=B, seq_idx=seq_idx, out=out)
    detail += f" state={str(state_dtype).removeprefix('torch.')}"
    num_seqs = int(cu_seqlens.shape[0]) - 1
    # ``track_seq_idx`` is set for every forward of a radix-cache-tracked batch
    # (even with no row to recompute); the mapped checkpoints decide whether
    # the Cake runner can write the tracked rows itself.
    tracking = track_seq_idx is not None
    checkpoints: dict = {}
    if tracking:
        if cake_track_checkpoints is None:
            _log_cake_route_once(
                route,
                "fallback",
                f"radix-cache track states requested but not mapped onto Cake "
                f"checkpoints (host track plan unavailable, or a row wants its "
                f"pre-first-token state): {detail}",
            )
            return None
        if cake_track_checkpoints.token_indices is not None:
            if (
                track_states_out is None
                or track_states_out.dim() != 4
                or not track_states_out.is_contiguous()
                or track_states_out.dtype != state_dtype
            ):
                _log_cake_route_once(
                    route,
                    "fallback",
                    f"track-state pool is not a contiguous 4-D {state_dtype} "
                    f"tensor (checkpoint target): {detail}",
                )
                return None
            checkpoints = dict(
                checkpoint_token_indices=cake_track_checkpoints.token_indices,
                checkpoint_state_slots=cake_track_checkpoints.state_slots,
                checkpoint_states=track_states_out,
            )
    if cake_chunk_indices is None or cake_chunk_offsets is None:
        _log_cake_route_once(
            route, "fallback", f"no chunk-128 metadata for this batch: {detail}"
        )
        return None
    if state_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        _log_cake_route_once(
            route,
            "fallback",
            f"state dtype not BF16/FP16/FP32 (--mamba-ssm-dtype): {detail}",
        )
        return None
    key = (
        seqlen,
        nheads,
        B.shape[2],
        x.dtype,
        state_dtype,
        initial_states is None,
        bool(checkpoints),
    )
    if key in _cake_route_rejected:
        return None
    supports, cake_fwd = _cake_ssd_kernels()
    # ``initial_states=None`` is the stock "no prefix" call; the Cake runner
    # takes it as such and learns the packed sequence count from ``num_seqs``.
    # The engine's token-major output buffer is the kernel's ``out``.
    dt_limit = (0.0, float("inf"))
    admitted = supports(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_limit=dt_limit,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=cake_chunk_indices,
        chunk_offsets=cake_chunk_offsets,
        out=out,
        num_seqs=num_seqs,
        chunk_size=SSD_CHUNK_SIZE,
        **checkpoints,
    )
    if not admitted:
        _log_cake_route_once(route, "fallback", f"adapter admission rejected: {detail}")
        return None
    try:
        _, final_states = cake_fwd(
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=True,
            dt_limit=dt_limit,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=cake_chunk_indices,
            chunk_offsets=cake_chunk_offsets,
            out=out,
            num_seqs=num_seqs,
            return_final_states=True,
            **checkpoints,
        )
    except NotImplementedError as error:  # FlashInfer host-side refusal
        _cake_route_rejected.add(key)
        _log_cake_route_once(
            route,
            "fallback",
            f"FlashInfer refused the configuration ({error}): {detail}",
        )
        return None
    _log_cake_route_once(route, "taken", detail)
    # Stock contract: (intermediate_states, varlen_state, track_states); the
    # per-sequence final states are the varlen states the mixer scatters back.
    # With tracking, the unaligned track rows were checkpointed into the pool
    # in-kernel; the sentinel tells the backend to skip its copy of them.
    return None, final_states, (SSD_TRACK_STATES_IN_PLACE if tracking else None)


# ---------------------------------------------------------------------------
# Selective state update (decode / target verify)
# ---------------------------------------------------------------------------


def _ssu_static_row(
    state: torch.Tensor, x: torch.Tensor, disable_state_update: bool, has_buffer: bool
) -> Optional[str]:
    """Shape-only pre-check against the promoted Cake SSU rows.

    Returns a reason string when no promoted row can match, so the per-call
    dtype casts and the adapter admission are skipped for shapes Cake never
    runs (e.g. the ``T = 1`` decode of a ``headdim = 64`` model).
    """
    dim, dstate = int(state.shape[-2]), int(state.shape[-1])
    if x.ndim == 3:
        if (dim, dstate) not in ((128, 128), (64, 128)):
            return f"no promoted T=1 row for (dim, dstate)=({dim}, {dstate})"
        return None
    if x.ndim != 4:
        return f"unsupported x rank {x.ndim}"
    steps = int(x.shape[1])
    if (dim, dstate) == (128, 128) and steps in (1, 2) and not disable_state_update:
        return None
    if (
        (dim, dstate, steps) == (64, 128, 6)
        and disable_state_update
        and has_buffer
        and state.dtype == torch.bfloat16
    ):
        return None
    return (
        f"no promoted MTP row for (dim, dstate, T)=({dim}, {dstate}, {steps}) "
        f"state={str(state.dtype).removeprefix('torch.')} "
        f"disable_state_update={disable_state_update}"
    )


def _fp32_broadcast(tensor: torch.Tensor) -> torch.Tensor:
    """FP32 view of a trailing-axis stride-0 broadcast, keeping the broadcast."""
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.stride(-1) != 0:
        return tensor.to(torch.float32)
    base = tensor[..., 0]
    return base.to(torch.float32).unsqueeze(-1).expand_as(tensor)


def _fp32_param_broadcast(tensor: torch.Tensor) -> torch.Tensor:
    """Like :func:`_fp32_broadcast` for a ``[H, dim]`` expand of a parameter;
    the FP32 ``[H]`` copy is cached on the parameter storage."""
    if tensor.dtype == torch.float32:
        return tensor
    if tensor.ndim != 2 or tensor.stride(1) != 0:
        return tensor.to(torch.float32)
    base = tensor[:, 0]
    key = (base.data_ptr(), base.stride(0), tuple(base.shape), base.dtype)
    cached = _fp32_params.get(key)
    if cached is None:
        cached = base.to(torch.float32).contiguous()
        _fp32_params[key] = cached
    return cached.unsqueeze(1).expand_as(tensor)


def _int64(indices: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if indices is None or indices.dtype == torch.int64:
        return indices
    return indices.to(torch.int64)


def _ssu_raw_abi_row(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    D: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    dst_indices: Optional[torch.Tensor],
) -> bool:
    """``True`` when the call is the headdim-64 single-token decode row and the
    engine's own storage (BF16 ``dt``/``D``/``dt_bias`` broadcasts, int32 slot
    tables, fused-projection views) is the ABI the Cake programs read directly,
    so no per-call conversion copies are needed."""
    return (
        x.ndim == 3
        and tuple(state.shape[-2:]) == (64, 128)
        and dt.dtype == torch.bfloat16
        and D.dtype == torch.bfloat16
        and dt_bias is not None
        and dt_bias.dtype == torch.bfloat16
        and indices is not None
        and indices.dtype == torch.int32
        and (dst_indices is None or dst_indices.dtype == torch.int32)
    )


def selective_state_update(
    stock: Callable[..., None],
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    **stock_kwargs,
) -> None:
    """Run the selective state update; ``mamba_ssu`` may take the Cake kernel.

    ``stock_kwargs`` are exactly the keyword arguments the mixer passes to the
    stock dispatcher (``z``, ``dt_bias``, ``dt_softplus``,
    ``state_batch_indices``, ``out``, and for target verify
    ``disable_state_update``, ``intermediate_states_buffer``, ``cache_steps``,
    ``retrieve_parent_token``, ``intermediate_state_indices``).
    """
    if cake_route_enabled(CAKE_ROUTE_SSU) and _cake_selective_state_update(
        state, x, dt, A, B, C, D, stock_kwargs
    ):
        return
    stock(state, x, dt, A, B, C, D, **stock_kwargs)


def _cake_selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    kw: dict,
) -> bool:
    """``True`` when the Cake kernel ran (``state`` / ``out`` / buffer updated)."""
    route = CAKE_ROUTE_SSU
    disable_state_update = bool(kw.get("disable_state_update", False))
    buffer = kw.get("intermediate_states_buffer")
    detail = _tensor_summary(
        x=x, dt=dt, B=B, state=state, indices=kw.get("state_batch_indices")
    )
    if kw.get("retrieve_parent_token") is not None:
        _log_cake_route_once(
            route, "fallback", f"tree verify (retrieve_parent_token): {detail}"
        )
        return False
    reason = _ssu_static_row(state, x, disable_state_update, buffer is not None)
    if reason is not None:
        _log_cake_route_once(route, "fallback", f"{reason}: {detail}")
        return False
    key = (
        tuple(x.shape),
        tuple(B.shape),
        tuple(state.shape[1:]),
        state.dtype,
        x.dtype,
        disable_state_update,
        buffer is not None,
    )
    if key in _cake_route_rejected:
        return False
    if torch.cuda.is_current_stream_capturing() and key not in _ssu_warm:
        _log_cake_route_once(
            route,
            "fallback",
            f"shape first seen inside CUDA-graph capture (no eager warm-up): {detail}",
        )
        return False
    supports, cake_ssu = _cake_ssu_kernels()
    dt_bias = kw.get("dt_bias")
    if _ssu_raw_abi_row(
        state, x, dt, D, dt_bias, kw.get("state_batch_indices"), kw.get("dst_state_batch_indices")
    ):
        # The headdim-64 decode programs read the engine's BF16 coefficient
        # broadcasts, int32 slot tables and fused-projection views in place.
        dt_fi, D_fi, dt_bias_fi = dt, D, dt_bias
        indices_fi = kw.get("state_batch_indices")
        buffer_indices_fi = kw.get("intermediate_state_indices")
    else:
        # The other promoted rows take FP32 per-head broadcasts and int64
        # indices; the engine holds BF16 broadcasts of its parameters and
        # int32 slot indices.
        dt_fi = _fp32_broadcast(dt)
        D_fi = _fp32_param_broadcast(D)
        dt_bias_fi = None if dt_bias is None else _fp32_param_broadcast(dt_bias)
        indices_fi = _int64(kw.get("state_batch_indices"))
        buffer_indices_fi = _int64(kw.get("intermediate_state_indices"))
    cache_steps = kw.get("cache_steps")
    cache_steps = 0 if cache_steps is None else int(cache_steps)
    algorithm = "horizontal" if x.ndim == 4 and x.shape[0] >= 32 else "auto"
    cake_kwargs = dict(
        z=kw.get("z"),
        dt_bias=dt_bias_fi,
        dt_softplus=bool(kw.get("dt_softplus", False)),
        state_batch_indices=indices_fi,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=buffer,
        intermediate_state_indices=buffer_indices_fi,
        cache_steps=cache_steps,
        algorithm=algorithm,
        pad_slot_id=int(kw.get("pad_slot_id", -1)),
    )
    admitted = _ssu_admission.get(key)
    if admitted is None:
        admitted = bool(supports(state, x, dt_fi, A, B, C, D_fi, **cake_kwargs))
        _ssu_admission[key] = admitted
    if not admitted:
        _log_cake_route_once(route, "fallback", f"adapter admission rejected: {detail}")
        return False
    try:
        cake_ssu(state, x, dt_fi, A, B, C, D_fi, out=kw.get("out"), **cake_kwargs)
    except NotImplementedError as error:  # FlashInfer host-side refusal
        _cake_route_rejected.add(key)
        _log_cake_route_once(
            route,
            "fallback",
            f"FlashInfer refused the configuration ({error}): {detail}",
        )
        return False
    _ssu_warm.add(key)
    _log_cake_route_once(route, "taken", f"{detail} algorithm={algorithm}")
    return True

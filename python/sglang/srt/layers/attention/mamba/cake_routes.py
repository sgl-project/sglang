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
those arguments unchanged.  The first "taken" and the first "fallback" per
route are logged so an e2e run can prove which kernel executed.

FlashInfer contract constraints that shape the wiring (see the adapter
:mod:`sglang.kernels.cake_kernels.mamba` for the full list):

* SSD: chunk 128 (the engine builds its chunk metadata for the model's
  ``mamba_chunk_size``, 256 for Nemotron-H, so :class:`Mamba2Metadata`
  carries a second, chunk-128 ``chunk_indices`` / ``chunk_offsets`` pair when
  the route is on), ``seqlen % 128 == 0`` (an unaligned prefill batch falls
  back; chunked-prefill batches of a 128-multiple size are covered), BF16 or
  FP16 state (``--mamba-ssm-dtype bfloat16``), varlen always needs
  ``initial_states`` (a cached zero buffer stands in when no sequence has a
  prefix), and the caller-owned ``out`` is head-major chunked
  ``[1, H, 64, nchunks, 128]`` while the engine's output is token-major, so
  the result is copied once into the engine buffer.
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

import functools
import logging
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

# (route, event) pairs already logged; keeps the per-call path free of I/O.
_cake_route_logged: set[tuple[str, str]] = set()
# Shape keys FlashInfer rejected with NotImplementedError; skipped afterwards.
_cake_route_rejected: set[tuple] = set()
# SSU shape keys admitted (and run) eagerly; only these are routed under capture.
_ssu_warm: set[tuple] = set()
# SSU admission is a pure function of the shape key; memoised per key.
_ssu_admission: dict[tuple, bool] = {}
# FP32 copies of the per-head parameters (D / dt_bias) keyed by storage.
_fp32_params: dict[tuple, torch.Tensor] = {}
# Zero initial states for varlen SSD batches without a prefix, per shape.
_zero_states: dict[tuple, torch.Tensor] = {}


def _log_cake_route_once(route: str, event: str, detail: str) -> None:
    key = (route, event)
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
    _zero_states.clear()


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


def cake_ssd_chunk_metadata(
    extend_seq_lens: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logical-chunk metadata of a packed prefill batch for the Cake chunk size.

    Same semantics as ``Mamba2Metadata._query_start_loc_to_chunk_indices_offsets``
    (a logical chunk starts at every physical 128-token chunk start and at
    every sequence start) computed from the host-side sequence lengths, so
    the metadata build does not synchronise the stream.  Returns int32
    ``(chunk_indices, chunk_offsets)`` on ``device``.
    """
    starts: set[int] = set()
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


def _zero_initial_states(
    num_seqs: int, nheads: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (num_seqs, nheads, dtype, device)
    states = _zero_states.get(key)
    if states is None:
        states = torch.zeros(
            (num_seqs, nheads, SSD_HEADDIM, SSD_DSTATE), dtype=dtype, device=device
        )
        _zero_states[key] = states
    return states


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
) -> tuple:
    """Run the prefill SSD scan; ``mamba_ssd_prefill`` may take the Cake kernel.

    Returns the stock ``(intermediate_states, varlen_state, track_states)``
    triple.  ``cake_chunk_indices`` / ``cake_chunk_offsets`` are the chunk-128
    metadata prepared once per forward by :class:`Mamba2Metadata`; everything
    else is the stock call's argument list.
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
) -> Optional[tuple]:
    """``None`` means "use the stock call"."""
    route = CAKE_ROUTE_SSD_PREFILL
    seqlen = x.shape[1]
    nheads = x.shape[2]
    detail = _tensor_summary(x=x, dt=dt, B=B, seq_idx=seq_idx, out=out)
    detail += f" state={str(state_dtype).removeprefix('torch.')}"
    if seqlen % SSD_CHUNK_SIZE:
        _log_cake_route_once(
            route, "fallback", f"seqlen {seqlen} is not a multiple of 128: {detail}"
        )
        return None
    if track_seq_idx is not None:
        _log_cake_route_once(
            route,
            "fallback",
            f"radix-cache track states requested (not mapped onto Cake "
            f"checkpoints): {detail}",
        )
        return None
    if cake_chunk_indices is None or cake_chunk_offsets is None:
        _log_cake_route_once(
            route, "fallback", f"no chunk-128 metadata for this batch: {detail}"
        )
        return None
    if state_dtype not in (torch.bfloat16, torch.float16):
        _log_cake_route_once(
            route,
            "fallback",
            f"state dtype not BF16/FP16 (--mamba-ssm-dtype): {detail}",
        )
        return None
    key = (seqlen, nheads, B.shape[2], x.dtype, state_dtype, initial_states is None)
    if key in _cake_route_rejected:
        return None
    supports, cake_fwd = _cake_ssd_kernels()
    num_seqs = int(cu_seqlens.shape[0]) - 1
    if initial_states is None:
        # The Cake varlen runner requires initial states; a zero buffer is the
        # "no prefix" case the stock kernel expresses with ``None``.
        initial_states = _zero_initial_states(num_seqs, nheads, state_dtype, x.device)
    # FlashInfer owns the head-major chunked output layout; the token-major
    # view it returns is copied into the engine's preallocated buffer below.
    cake_out = torch.empty(
        (1, nheads, SSD_HEADDIM, seqlen // SSD_CHUNK_SIZE, SSD_CHUNK_SIZE),
        dtype=torch.bfloat16,
        device=x.device,
    )
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
        out=cake_out,
        chunk_size=SSD_CHUNK_SIZE,
    )
    if not admitted:
        _log_cake_route_once(route, "fallback", f"adapter admission rejected: {detail}")
        return None
    try:
        out_view, final_states = cake_fwd(
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
            out=cake_out,
            return_final_states=True,
        )
    except NotImplementedError as error:  # FlashInfer host-side refusal
        _cake_route_rejected.add(key)
        _log_cake_route_once(
            route,
            "fallback",
            f"FlashInfer refused the configuration ({error}): {detail}",
        )
        return None
    # Forced by the FI contract: Cake's ``out`` is [1, H, 64, nchunks, 128];
    # the engine consumes the token-major [1, S, H, 64] buffer it passed.
    out.copy_(out_view)
    _log_cake_route_once(route, "taken", detail)
    # Stock contract: (intermediate_states, varlen_state, track_states); the
    # per-sequence final states are the varlen states the mixer scatters back.
    return None, final_states, None


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
        if (dim, dstate) != (128, 128):
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
    # Promoted rows take FP32 per-head broadcasts and int64 indices; the engine
    # holds BF16 broadcasts of its parameters and int32 slot indices.
    dt_fi = _fp32_broadcast(dt)
    D_fi = _fp32_param_broadcast(D)
    dt_bias = kw.get("dt_bias")
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

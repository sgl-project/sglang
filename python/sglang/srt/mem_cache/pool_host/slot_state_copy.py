"""Stream-ordered copy of Mamba side-state rows from the device into the pinned host pool
(SGLANG_HICACHE_MAMBA_SLOT_STATE_ASYNC, default off).

MambaPoolHost._transfer_slot_states backs the registered side states (the PLE short-conv and
n-gram rows of a checkpoint slot) up with ``.to("cpu")`` copies: the host index tensor and then
the gathered rows come back to pageable memory, so the scheduler thread waits for every kernel
queued before them on the stream (the conv/temporal backup and the KV backups in front of it).
Here one kernel per state gathers the rows by device index and stores them into the pinned
host buffer through UVA, as the JIT HiCache kernels and the NVFP4 host copy already address
the host pools. The copy is queued on the same stream ahead of the backup's completion event,
the event every reader of a backed-up slot waits on, which is the ordering the conv/temporal
rows of the same slots already have. No staging buffer: no host memory beyond the pool.

Bytes are copied as they are (int32 or uint8 words): bitwise to the blocking path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_HICACHE_MAMBA_SLOT_STATE_ASYNC", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_BLOCK = 4096
# off_reason: set once the path is found unusable (layout, warmup or launch failure); the stock copy runs from then on.
_state = {"kernel": None, "on_logged": False, "off_reason": None}


class _PinnedRows:
    """A pinned host tensor as a Triton pointer argument (registered host memory is mapped
    under UVA; the device address equals the host one on CUDA)."""

    __slots__ = ("_tensor", "dtype")

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self.dtype = tensor.dtype

    def data_ptr(self) -> int:
        return self._tensor.data_ptr()


def _kernel():
    if _state["kernel"] is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _slot_rows_to_host(SRC, DST, SRC_IDX, DST_IDX, ROW: tl.constexpr, BLOCK: tl.constexpr):
            i = tl.program_id(0)
            cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            live = cols < ROW
            s = tl.load(SRC_IDX + i).to(tl.int64)
            d = tl.load(DST_IDX + i).to(tl.int64)
            tl.store(DST + d * ROW + cols, tl.load(SRC + s * ROW + cols, mask=live), mask=live)

        _state["kernel"] = _slot_rows_to_host
    return _state["kernel"]


def _words(t: torch.Tensor) -> Optional[torch.Tensor]:
    """The tensor's bytes as [slots, row words] int32 (uint8 when a row or base is not 4 B aligned)."""
    if not t.is_contiguous() or t.dim() < 1:
        return None
    flat = t.view(torch.uint8).reshape(t.shape[0], -1)
    if flat.shape[1] % 4 == 0 and t.data_ptr() % 4 == 0:
        return flat.view(torch.int32)
    return flat


def _disable(reason: str) -> None:
    """Turn the path off for the process; MambaPoolHost keeps its stock blocking copy."""
    if _state["off_reason"] is None:
        _state["off_reason"] = reason
        logger.warning(
            "Mamba slot-state backup: kept the blocking copy (SGLANG_HICACHE_MAMBA_SLOT_STATE_ASYNC: %s)",
            reason,
        )


def _layout_problem(state: torch.Tensor, host: torch.Tensor) -> Optional[str]:
    if not (state.is_cuda and state.is_contiguous() and host.device.type == "cpu"):
        return "a state is not a contiguous CUDA tensor beside a host buffer"
    if not (host.is_contiguous() and host.is_pinned()):
        return "a host buffer is not pinned and contiguous"
    if state.dtype != host.dtype or tuple(state.shape[1:]) != tuple(host.shape[1:]):
        return "a state and its host buffer differ in row layout"
    src, dst = _words(state), _words(host)
    if src is None or dst is None or src.dtype != dst.dtype or src.shape[1] != dst.shape[1]:
        return "a state and its host buffer differ in word layout (base alignment)"
    return None


def applies(states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]) -> bool:
    """CUDA device states and pinned, contiguous host buffers with the same per-slot row bytes and
    the same copy words; False (stock copy) once any check or an earlier warmup or launch failed."""
    if not ENABLED or not states or _state["off_reason"] is not None:
        return False
    if len(states) != len(host_buffers):
        _disable("states and host buffers differ in count")
        return False
    for state, host in zip(states, host_buffers):
        problem = _layout_problem(state, host)
        if problem is not None:
            _disable(problem)
            return False
    return True


def backup_rows(
    states: Sequence[torch.Tensor],
    host_buffers: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
) -> bool:
    """host_buffer[host_indices[i]] = state[device_indices[i]] for every state, queued on the
    current stream; the host thread never waits. False when a launch failed: the path is then
    off and the caller runs its stock copy (rewriting rows already queued here is harmless)."""
    n = device_indices.numel()
    if n == 0:
        return True
    try:
        _launch(states, host_buffers, device_indices, host_indices, n)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the stock copy
        _disable(f"copy failed, stock copy used: {type(exc).__name__}: {exc}")
        return False
    if not _state["on_logged"]:
        _state["on_logged"] = True
        logger.info(
            "Mamba slot-state backup: stream-ordered device-to-pinned copy, no host wait "
            "(SGLANG_HICACHE_MAMBA_SLOT_STATE_ASYNC; %d states, %s bytes per slot)",
            len(states),
            [s[0].numel() * s.element_size() for s in states],
        )
    return True


def load_rows(
    states: Sequence[torch.Tensor],
    host_buffers: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
) -> bool:
    """state[device_indices[i]] = host_buffer[host_indices[i]] for every state, queued on the
    current stream (the load leg of the same copy, read from pinned memory through UVA), so the
    host never waits on device host indices behind the load fence. Used under host-only Mamba
    checkpoints (SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT). False: the caller runs its stock copy."""
    n = device_indices.numel()
    if n == 0:
        return True
    try:
        device = states[0].device
        dev_idx = device_indices.reshape(-1).to(device=device, dtype=torch.long)
        if host_indices.device.type == "cuda":
            host_idx = host_indices.reshape(-1).to(device=device, dtype=torch.long)
        else:
            host_idx = (
                host_indices.reshape(-1)
                .to(dtype=torch.long)
                .pin_memory()
                .to(device=device, non_blocking=True)
            )
        kernel = _kernel()
        for state, host in zip(states, host_buffers):
            src, dst = _words(host), _words(state)
            row = dst.shape[1]
            grid = (n, triton_cdiv(row, _BLOCK))
            kernel[grid](_PinnedRows(src), dst, host_idx, dev_idx, ROW=row, BLOCK=_BLOCK, num_warps=4)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the stock copy
        _disable(f"load failed, stock copy used: {type(exc).__name__}: {exc}")
        return False
    if not _state.get("load_logged"):
        _state["load_logged"] = True
        logger.info(
            "Mamba slot-state load: stream-ordered pinned-to-device copy, no host wait "
            "(SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT; %d states)",
            len(states),
        )
    return True


def _launch(states, host_buffers, device_indices, host_indices, n: int) -> None:
    device = states[0].device
    dev_idx = device_indices.reshape(-1).to(device=device, dtype=torch.long)
    if host_indices.device.type == "cuda":
        host_idx = host_indices.reshape(-1).to(device=device, dtype=torch.long)
    else:
        # Pinned staging keeps the H2D off the stream's queue (a pageable one may wait on it).
        host_idx = (
            host_indices.reshape(-1)
            .to(dtype=torch.long)
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
    kernel = _kernel()
    for state, host in zip(states, host_buffers):
        src, dst = _words(state), _words(host)
        row = src.shape[1]
        grid = (n, triton_cdiv(row, _BLOCK))
        kernel[grid](src, _PinnedRows(dst), dev_idx, host_idx, ROW=row, BLOCK=_BLOCK, num_warps=4)


def warm(states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]) -> None:
    """Compile the copy for these buffers at pool construction, so the first eviction backup
    does not wait on a JIT compile. Nothing is launched. A failure turns the path off (stock copy)."""
    if not applies(states, host_buffers):
        return
    try:
        kernel = _kernel()
        idx = torch.zeros(1, dtype=torch.long, device=states[0].device)
        for state, host in zip(states, host_buffers):
            src, dst = _words(state), _words(host)
            row = src.shape[1]
            kernel.warmup(src, _PinnedRows(dst), idx, idx, ROW=row, BLOCK=_BLOCK, num_warps=4,
                          grid=(1, triton_cdiv(row, _BLOCK)))
            if envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.get():  # the load leg (load_rows)
                kernel.warmup(_PinnedRows(dst), src, idx, idx, ROW=row, BLOCK=_BLOCK, num_warps=4,
                              grid=(1, triton_cdiv(row, _BLOCK)))
    except Exception as exc:  # noqa: BLE001 - fail closed: the stock copy needs no compile
        _disable(f"warmup failed: {type(exc).__name__}: {exc}")


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# SGLANG_HICACHE_LOAD_BACK_DEVICE_ORDERED (default off): the load leg. The stock leg brings the moved
# host indices back with a D2H on the load stream, which is fenced on the in-flight forward, so
# start_loading waited for that forward; here the same kernel gathers the rows from the pinned host
# buffer through UVA into the device state, queued on the load stream. Bitwise: bytes as they are.
LOAD_ENABLED = os.environ.get("SGLANG_HICACHE_LOAD_BACK_DEVICE_ORDERED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_load_state = {"on_logged": False, "off_reason": None}


def _disable_load(reason: str) -> None:
    """Turn the load path off for the process; MambaPoolHost keeps its stock blocking copy."""
    if _load_state["off_reason"] is None:
        _load_state["off_reason"] = reason
        logger.warning(
            "Mamba slot-state load-back: kept the blocking copy (SGLANG_HICACHE_LOAD_BACK_DEVICE_ORDERED: %s)",
            reason,
        )


def load_applies(states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]) -> bool:
    """applies() for the load leg, under its own switch and failure state."""
    if not LOAD_ENABLED or not states or _load_state["off_reason"] is not None:
        return False
    if len(states) != len(host_buffers):
        _disable_load("states and host buffers differ in count")
        return False
    for state, host in zip(states, host_buffers):
        problem = _layout_problem(state, host)
        if problem is not None:
            _disable_load(problem)
            return False
    return True


def load_rows_device_ordered(
    states: Sequence[torch.Tensor],
    host_buffers: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
) -> bool:
    """state[device_indices[i]] = host_buffer[host_indices[i]] for every state, queued on the
    current stream; the host thread never waits. False when the indices disagree in count or a
    launch failed: the caller then runs its stock copy, which rewrites the same rows.
    (SGLANG_HICACHE_LOAD_BACK_DEVICE_ORDERED's leg; ``load_rows`` above is the host-only leg.)"""
    n = device_indices.numel()
    if host_indices.numel() != n:
        return False
    if n == 0:
        return True
    try:
        _launch_load(states, host_buffers, device_indices, host_indices, n)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the stock copy
        _disable_load(f"copy failed, stock copy used: {type(exc).__name__}: {exc}")
        return False
    if not _load_state["on_logged"]:
        _load_state["on_logged"] = True
        logger.info(
            "Mamba slot-state load-back: stream-ordered pinned-to-device copy on the load stream, "
            "no host wait (SGLANG_HICACHE_LOAD_BACK_DEVICE_ORDERED; %d states, %s bytes per slot)",
            len(states),
            [s[0].numel() * s.element_size() for s in states],
        )
    return True


def _load_indices(states, device_indices, host_indices):
    device = states[0].device
    dev_idx = device_indices.reshape(-1).to(device=device, dtype=torch.long)
    if host_indices.device.type == "cuda":
        host_idx = host_indices.reshape(-1).to(device=device, dtype=torch.long)
    else:
        # The caching host allocator keeps the pinned block until this stream has copied it.
        host_idx = (
            host_indices.reshape(-1)
            .to(dtype=torch.long)
            .pin_memory()
            .to(device=device, non_blocking=True)
        )
    return dev_idx, host_idx


def _launch_load(states, host_buffers, device_indices, host_indices, n: int) -> None:
    dev_idx, host_idx = _load_indices(states, device_indices, host_indices)
    kernel = _kernel()
    for state, host in zip(states, host_buffers):
        src, dst = _words(host), _words(state)
        row = src.shape[1]
        grid = (n, triton_cdiv(row, _BLOCK))
        kernel[grid](_PinnedRows(src), dst, host_idx, dev_idx, ROW=row, BLOCK=_BLOCK, num_warps=4)


def _scratch_row(like: torch.Tensor) -> torch.Tensor:
    """One device row of ``like``'s word dtype at an address in the same 16-byte alignment class,
    so a launch into it uses the specialization a load-back into ``like`` uses."""
    row = like.shape[1]
    pad = 16 // like.element_size()
    buf = torch.empty(row + pad, dtype=like.dtype, device=like.device)
    aligned = like.data_ptr() % 16 == 0
    for offset in range(pad):
        view = buf[offset : offset + row]
        if (view.data_ptr() % 16 == 0) == aligned:
            return view.view(1, row)
    raise RuntimeError("no scratch row in the state's alignment class")


def warm_load(states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]) -> None:
    """warm() for the load direction, run once at pool construction: launches per state copy host
    row 0 into a scratch row. A variant's first launch loads its module, which waits for all
    device work; in serving that is the forward in flight. A failure turns the load path off."""
    if not load_applies(states, host_buffers):
        return
    try:
        kernel = _kernel()
        device = states[0].device
        idx = torch.zeros(2, dtype=torch.long, device=device)
        # Triton compiles a variant per 16-byte alignment class of each index pointer; warm both.
        idx_classes = (idx[0:1], idx[1:2])
        for state, host in zip(states, host_buffers):
            src = _words(host)
            row = src.shape[1]
            scratch = _scratch_row(_words(state))
            for src_idx in idx_classes:
                for dst_idx in idx_classes:
                    kernel[(1, triton_cdiv(row, _BLOCK))](
                        _PinnedRows(src), scratch, src_idx, dst_idx, ROW=row, BLOCK=_BLOCK, num_warps=4
                    )
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - fail closed: the stock copy needs no compile
        _disable_load(f"warmup failed: {type(exc).__name__}: {exc}")

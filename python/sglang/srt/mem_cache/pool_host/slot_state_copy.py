"""Stream-ordered copies of Mamba side-state rows between the device and the pinned
host pool through UVA, queued ahead of the backup's completion event; bitwise to the
blocking copy, which runs for any layout the kernel cannot serve or after a failed
warmup or launch."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_BLOCK = 4096
# off_reason: once set, the blocking copy runs for the rest of the process.
_state = {"kernel": None, "on_logged": False, "load_logged": False, "off_reason": None}


class _PinnedRows:
    """A pinned host tensor as a Triton pointer (under UVA, device address = host)."""

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
        def _slot_rows_copy(
            SRC, DST, SRC_IDX, DST_IDX, ROW: tl.constexpr, BLOCK: tl.constexpr
        ):
            i = tl.program_id(0)
            cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            live = cols < ROW
            s = tl.load(SRC_IDX + i).to(tl.int64)
            d = tl.load(DST_IDX + i).to(tl.int64)
            tl.store(
                DST + d * ROW + cols,
                tl.load(SRC + s * ROW + cols, mask=live),
                mask=live,
            )

        _state["kernel"] = _slot_rows_copy
    return _state["kernel"]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _words(t: torch.Tensor) -> Optional[torch.Tensor]:
    """[slots, row words] int32, or uint8 when a row or base is not 4 B aligned."""
    if not t.is_contiguous() or t.dim() < 1:
        return None
    flat = t.view(torch.uint8).reshape(t.shape[0], -1)
    if flat.shape[1] % 4 == 0 and t.data_ptr() % 4 == 0:
        return flat.view(torch.int32)
    return flat


def _disable(reason: str) -> None:
    """Turn the path off for the process; MambaPoolHost keeps its blocking copy."""
    if _state["off_reason"] is None:
        _state["off_reason"] = reason
        logger.warning(
            "Mamba slot-state copy: kept the blocking copy (%s)",
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
    if (
        src is None
        or dst is None
        or src.dtype != dst.dtype
        or src.shape[1] != dst.shape[1]
    ):
        return "a state and its host buffer differ in word layout (base alignment)"
    return None


def applies(
    states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]
) -> bool:
    """Whether the kernel serves these buffers; False after any failure."""
    if not states or _state["off_reason"] is not None:
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


def _indices(device: torch.device, device_indices, host_indices):
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
    return dev_idx, host_idx


def _launch(states, host_buffers, device_indices, host_indices, n: int, *, to_device):
    dev_idx, host_idx = _indices(states[0].device, device_indices, host_indices)
    kernel = _kernel()
    for state, host in zip(states, host_buffers):
        dev_words, host_words = _words(state), _words(host)
        row = dev_words.shape[1]
        grid = (n, _cdiv(row, _BLOCK))
        if to_device:
            kernel[grid](
                _PinnedRows(host_words),
                dev_words,
                host_idx,
                dev_idx,
                ROW=row,
                BLOCK=_BLOCK,
                num_warps=4,
            )
        else:
            kernel[grid](
                dev_words,
                _PinnedRows(host_words),
                dev_idx,
                host_idx,
                ROW=row,
                BLOCK=_BLOCK,
                num_warps=4,
            )


def backup_rows(
    states: Sequence[torch.Tensor],
    host_buffers: Sequence[torch.Tensor],
    device_indices: torch.Tensor,
    host_indices: torch.Tensor,
) -> bool:
    """host_buffer[host_indices[i]] = state[device_indices[i]] on the current stream.
    False: the launch failed; the caller runs its blocking copy."""
    n = device_indices.numel()
    if n == 0:
        return True
    try:
        _launch(states, host_buffers, device_indices, host_indices, n, to_device=False)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the blocking copy
        _disable(f"copy failed, blocking copy used: {type(exc).__name__}: {exc}")
        return False
    if not _state["on_logged"]:
        _state["on_logged"] = True
        logger.info(
            "Mamba slot-state backup: stream-ordered device-to-pinned copy, no host wait "
            "(%d states, %s bytes per slot)",
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
    """state[device_indices[i]] = host_buffer[host_indices[i]] on the current stream.
    False: the caller runs its blocking copy."""
    n = device_indices.numel()
    if n == 0:
        return True
    try:
        _launch(states, host_buffers, device_indices, host_indices, n, to_device=True)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the blocking copy
        _disable(f"load failed, blocking copy used: {type(exc).__name__}: {exc}")
        return False
    if not _state["load_logged"]:
        _state["load_logged"] = True
        logger.info(
            "Mamba slot-state load: stream-ordered pinned-to-device copy, no host wait "
            "(SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT; %d states)",
            len(states),
        )
    return True


def warm(states: Sequence[torch.Tensor], host_buffers: Sequence[torch.Tensor]) -> None:
    """Compile the copies at pool construction, off the first eviction's path."""
    if not applies(states, host_buffers):
        return
    try:
        kernel = _kernel()
        idx = torch.zeros(1, dtype=torch.long, device=states[0].device)
        load_leg = envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.get()
        for state, host in zip(states, host_buffers):
            dev_words, host_words = _words(state), _words(host)
            row = dev_words.shape[1]
            grid = (1, _cdiv(row, _BLOCK))
            kernel.warmup(
                dev_words,
                _PinnedRows(host_words),
                idx,
                idx,
                ROW=row,
                BLOCK=_BLOCK,
                num_warps=4,
                grid=grid,
            )
            if load_leg:
                kernel.warmup(
                    _PinnedRows(host_words),
                    dev_words,
                    idx,
                    idx,
                    ROW=row,
                    BLOCK=_BLOCK,
                    num_warps=4,
                    grid=grid,
                )
    except Exception as exc:  # noqa: BLE001 - fail closed: the blocking copy needs no compile
        _disable(f"warmup failed: {type(exc).__name__}: {exc}")

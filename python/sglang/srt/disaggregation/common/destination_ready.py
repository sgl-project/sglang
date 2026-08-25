"""Destination-GPU epoch readiness primitive for disaggregation transports.

This module is deliberately transport- and payload-agnostic.  A future
metadata protocol may place a release-written epoch next to a destination KV
slot, then call :func:`wait_for_destination_ready_epoch` on the destination
GPU before making that slot consumable.  It does not validate payload bytes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The result buffer is ``[observed_epoch, status]``.  It is intentionally GPU
# resident until the wrapper has synchronized the caller's current stream.
_READY = 0
_TIMEOUT = 1
_STALE = 2
_REGRESSED = 3
_FUTURE = 4
_MAX_SPINS = 10_000_000


class ReadyEpochError(RuntimeError):
    """Base error for a destination epoch that was not consumable."""


class ReadyEpochTimeoutError(ReadyEpochError):
    """The ready epoch remained zero for the configured spin budget."""


class ReadyEpochStaleError(ReadyEpochError):
    """The ready epoch was stale, regressed, or skipped the expected epoch."""


@triton.jit
def _ld_acquire_sys_u64(ptr):
    return tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u64 $0, [$1];",
        "=l,l",
        [ptr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["expected_epoch"])
def _wait_ready_epoch_kernel(
    ready_epoch,
    result,
    expected_epoch,
    max_spins,
):
    observed = tl.full((), 0, tl.uint64)
    largest_observed = tl.full((), 0, tl.uint64)
    status = tl.full((), 1, tl.uint64)
    spin = tl.full((), 0, tl.int32)
    done = tl.full((), 0, tl.int1)

    while (spin < max_spins) & (done == 0):
        observed = _ld_acquire_sys_u64(ready_epoch)
        if observed == expected_epoch:
            status = tl.full((), 0, tl.uint64)
            done = tl.full((), 1, tl.int1)
        elif observed > expected_epoch:
            status = tl.full((), 4, tl.uint64)
            done = tl.full((), 1, tl.int1)
        elif observed < largest_observed:
            status = tl.full((), 3, tl.uint64)
            done = tl.full((), 1, tl.int1)
        else:
            largest_observed = observed
            spin += 1

    if done == 0:
        if observed != 0:
            status = tl.full((), 2, tl.uint64)
        else:
            status = tl.full((), 1, tl.uint64)

    tl.store(result, observed)
    tl.store(result + 1, status)


def _validate_ready_epoch(ready_epoch: torch.Tensor) -> None:
    if not isinstance(ready_epoch, torch.Tensor):
        raise TypeError("ready_epoch must be a torch.Tensor")
    if ready_epoch.dtype != torch.uint64:
        raise ValueError("ready_epoch must have dtype torch.uint64")
    if ready_epoch.shape != (1,):
        raise ValueError("ready_epoch must have shape (1,)")
    if not ready_epoch.is_cuda:
        raise ValueError("ready_epoch must be a CUDA tensor")
    if not ready_epoch.is_contiguous():
        raise ValueError("ready_epoch must be contiguous")


def _validate_positive_int(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if not 0 < value <= maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")


def wait_for_destination_ready_epoch(
    ready_epoch: torch.Tensor,
    expected_epoch: int,
    *,
    max_spins: int = 100_000,
) -> int:
    """Wait on the current CUDA stream until ``ready_epoch`` equals an epoch.

    The kernel performs system-scope acquire loads and writes only its observed
    epoch and status to a temporary GPU tensor.  The wrapper synchronizes the
    current stream before reading that control result.  It performs no D2H
    payload visibility check and no global barrier.
    """

    _validate_positive_int("expected_epoch", expected_epoch, (1 << 63) - 1)
    _validate_positive_int("max_spins", max_spins, _MAX_SPINS)
    _validate_ready_epoch(ready_epoch)

    result = torch.empty(2, dtype=torch.uint64, device=ready_epoch.device)
    stream = torch.cuda.current_stream(ready_epoch.device)
    _wait_ready_epoch_kernel[(1,)](
        ready_epoch,
        result,
        expected_epoch,
        max_spins,
        num_warps=1,
    )
    stream.synchronize()
    observed, status = (int(value) for value in result.cpu().tolist())

    if status == _READY:
        return observed
    if status == _TIMEOUT:
        raise ReadyEpochTimeoutError(
            f"ready epoch {expected_epoch} timed out after {max_spins} spins"
        )
    if status in (_STALE, _REGRESSED, _FUTURE):
        raise ReadyEpochStaleError(
            f"ready epoch {expected_epoch} rejected with observed epoch {observed}"
        )
    raise ReadyEpochError(f"ready epoch {expected_epoch} returned status {status}")

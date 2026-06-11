"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class _AsyncNanWarner:
    """Non-fatal NaN monitor: device-side detection accumulates into a GPU
    counter copied to pinned host memory without any stream sync; the host
    reads the (slightly stale) value on a later call and warns, throttled."""

    WARN_INTERVAL_S = 30.0

    def __init__(self):
        self._dev = None
        self._host = None
        self._reported = 0
        self._last_warn_time = 0.0

    def check(self, tensor: torch.Tensor, msg: str):
        if not tensor.is_cuda:
            return
        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        # Report hits enqueued on earlier steps (pinned-memory read, no sync).
        seen = int(self._host[0])
        now = time.monotonic()
        if seen > self._reported and now - self._last_warn_time >= self.WARN_INTERVAL_S:
            logger.warning(
                "NaN detected in %s (%d batches so far); values were sanitized "
                "before sampling. This usually indicates numerical overflow "
                "(e.g. fp16 activations) or an upstream bug producing NaN.",
                msg,
                seen,
            )
            self._reported = seen
            self._last_warn_time = now

        # Enqueue this step's detection (async, no sync).
        self._dev.add_(torch.isnan(tensor).any().to(torch.int32))
        self._host.copy_(self._dev, non_blocking=True)


_nan_warner = _AsyncNanWarner()


def maybe_warn_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Non-fatal counterpart of maybe_detect_nan, active when the assert gate
    is OFF: warn (throttled, sync-free) instead of crashing. Callers are
    expected to sanitize the tensor themselves."""
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        # The hard assert path already covers detection.
        return
    if tensor is None:
        return
    _nan_warner.check(tensor, msg)


def maybe_detect_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    # A None tensor means there is nothing to probe, e.g. hidden_states on
    # capture_hidden_mode=NULL paths (STANDALONE speculative decoding).
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


def maybe_detect_oob(indices: Optional[torch.Tensor], low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point.

    Low/high asserted separately so the message names which failed (low =
    negative/sentinel, high = out of range).
    """
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    torch._assert_async(
        indices.min() >= low,
        f"index < {low} (negative / unmasked sentinel?): {msg}",
    )
    torch._assert_async(
        indices.max() < high,
        f"index >= {high} (out of range): {msg}",
    )


def maybe_detect_page_aligned(
    indices: Optional[torch.Tensor], page_size: int, msg: str
):
    """Async page-alignment check on slot ids."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0 or page_size <= 1:
        return
    torch._assert_async(
        (indices % page_size == 0).all(),
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )

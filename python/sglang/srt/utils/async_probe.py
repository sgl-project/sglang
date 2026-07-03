"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _xpu_assert_sync(condition: torch.Tensor, msg: str) -> bool:
    """XPU has no torch._assert_async; fall back to a synchronous check.

    Returns True if the (XPU) tensor was handled here so the caller can skip
    the async assert; False for non-XPU tensors which take the async path.
    """
    if condition.device.type == "xpu":
        if not bool(condition.item()):
            raise AssertionError(msg)
        return True
    return False


class _AsyncNanWarner:
    """One-shot NaN monitor: device-side detection lands in pinned host
    memory without any stream sync; the host reads the (slightly stale) flag
    on a later call, warns once, and stops detecting."""

    def __init__(self):
        self._dev = None
        self._host = None
        self._warned = False

    def check(self, tensor: torch.Tensor, msg: str):
        if self._warned or not tensor.is_cuda:
            return
        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        # Report a hit enqueued on an earlier step (pinned read, no sync).
        if int(self._host[0]):
            logger.warning(
                "NaN detected in %s; values were sanitized before sampling. "
                "This usually indicates numerical overflow (e.g. fp16 "
                "activations) or an upstream bug producing NaN. "
                "Logged once; further occurrences are silent.",
                msg,
            )
            self._warned = True
            return

        # Enqueue this step's detection (async, no sync).
        self._dev.add_(torch.isnan(tensor).any().to(torch.int32))
        self._host.copy_(self._dev, non_blocking=True)


_nan_warner = _AsyncNanWarner()


def maybe_warn_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Non-fatal counterpart of maybe_detect_nan: throttled sync-free warning
    instead of crashing. Callers sanitize the tensor themselves."""
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        # The hard assert path already covers detection.
        return
    if tensor is None:
        return
    _nan_warner.check(tensor, msg)


def sanitize_nan_logits(logits: torch.Tensor, msg: str = ""):
    """Detect NaN (assert in CI, throttled warning in prod), then sanitize in
    place: NaN logits (e.g. fp16 activation overflow) are undefined behavior
    in sampling kernels and can come back as out-of-vocab token ids. +-1e30
    rather than dtype min/max because callers divide logits by temperature,
    which would overflow dtype min/max to +-Inf and softmax back to NaN."""
    maybe_detect_nan(logits, msg)
    if not envs.SGLANG_SANITIZE_NAN_LOGITS.get():
        return
    maybe_warn_nan(logits, msg)
    torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)


def maybe_assert_async(cond: torch.Tensor, msg: str = ""):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    torch._assert_async(cond, msg)


def maybe_detect_nan(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async NaN check — no GPU-CPU sync, error surfaces at next sync point."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    # A None tensor means there is nothing to probe, e.g. hidden_states on
    # capture_hidden_mode=NULL paths (STANDALONE speculative decoding).
    if tensor is None:
        return
    condition = ~torch.any(torch.isnan(tensor))
    if _xpu_assert_sync(condition, f"NaN detected! {msg}"):
        return
    torch._assert_async(condition, f"NaN detected! {msg}")


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    condition = ~torch.any(torch.isinf(tensor))
    if _xpu_assert_sync(condition, f"Inf detected! {msg}"):
        return
    torch._assert_async(condition, f"Inf detected! {msg}")


def maybe_detect_in_closed_range(
    tensor: Optional[torch.Tensor], low: float, high: float, msg: str = ""
):
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None or tensor.numel() == 0:
        return
    torch._assert_async(
        ((tensor >= low) & (tensor <= high)).all(),
        f"value outside [{low}, {high}]: {msg}",
    )


def maybe_detect_oob(indices: Optional[torch.Tensor], low: int, high: int, msg: str):
    """Async OOB check — no GPU-CPU sync, error surfaces at next sync point.

    Low/high asserted separately so the message names which failed (low =
    negative/sentinel, high = out of range).
    """
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if indices is None or indices.numel() == 0:
        return
    low_ok = indices.min() >= low
    if not _xpu_assert_sync(
        low_ok, f"index < {low} (negative / unmasked sentinel?): {msg}"
    ):
        torch._assert_async(
            low_ok,
            f"index < {low} (negative / unmasked sentinel?): {msg}",
        )
    high_ok = indices.max() < high
    if not _xpu_assert_sync(high_ok, f"index >= {high} (out of range): {msg}"):
        torch._assert_async(
            high_ok,
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
    condition = (indices % page_size == 0).all()
    if _xpu_assert_sync(
        condition, f"page-misaligned indices (page_size={page_size}): {msg}"
    ):
        return
    torch._assert_async(
        condition,
        f"page-misaligned indices (page_size={page_size}): {msg}",
    )

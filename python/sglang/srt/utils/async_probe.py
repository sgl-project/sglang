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
    """Throttled NaN monitor: device-side detection lands in pinned host
    memory without any stream sync; the host reads the (slightly stale)
    counter on a later call and warns at most once per interval, reporting how
    many events that line covers.

    Counting rather than latching on the first hit: a NaN burst is the signal
    operators need (which node, how often, still ongoing?), and a one-shot
    latch hides every occurrence after the first for the process's lifetime."""

    def __init__(self, interval_s: float = 60.0):
        self._dev = None
        self._host = None
        self._interval_s = interval_s
        self._reported_ct = 0
        self._last_warn_time = None

    def check(self, tensor: torch.Tensor, msg: str):
        if not tensor.is_cuda:
            return
        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        # Report hits enqueued on earlier steps (pinned read, no sync).
        seen_ct = int(self._host[0])
        num_new = seen_ct - self._reported_ct
        now = time.monotonic()
        if num_new > 0 and (
            self._last_warn_time is None
            or now - self._last_warn_time >= self._interval_s
        ):
            logger.warning(
                "NaN detected in %s on %d forward pass(es) since the last such "
                "warning (%d total); values were sanitized before sampling. "
                "This usually indicates numerical overflow (e.g. fp16 "
                "activations) or an upstream bug producing NaN. Further "
                "occurrences are reported at most every %.0fs.",
                msg,
                num_new,
                seen_ct,
                self._interval_s,
            )
            self._reported_ct = seen_ct
            self._last_warn_time = now

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


def detect_full_nan_rows(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """Bool mask [#rows] of rows that are entirely NaN, or None when
    SGLANG_ABORT_ON_NAN_LOGITS is off.

    Must be read *before* sanitize_nan_logits: sanitization maps a full-NaN
    row to a constant -1e30 row, i.e. a uniform distribution, so sampling it
    returns a uniformly random token id. That is indistinguishable from a
    healthy sample downstream — the request streams vocabulary noise as a 200
    response and never reaches EOS. A partial-NaN row still holds real
    evidence about the next token, so only full rows are reported here.
    """
    if not envs.SGLANG_ABORT_ON_NAN_LOGITS.get():
        return None
    return torch.isnan(logits).all(dim=-1)


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
    torch._assert_async(~torch.any(torch.isnan(tensor)), f"NaN detected! {msg}")


def maybe_detect_inf(tensor: Optional[torch.Tensor], msg: str = ""):
    """Async Inf check — fp16 overflow surfaces as Inf before NaN."""
    if not envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return
    if tensor is None:
        return
    torch._assert_async(~torch.any(torch.isinf(tensor)), f"Inf detected! {msg}")


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

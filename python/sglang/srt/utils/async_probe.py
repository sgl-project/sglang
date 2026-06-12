"""Async invariant probes — fire torch._assert_async without CPU sync.

All probes are gated on SGLANG_ENABLE_ASYNC_ASSERT (default off in prod).
When the gate is on, a violation surfaces as an assertion at the next CUDA
sync point instead of as a silent NaN cascade or illegal-address crash.
"""

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_hip

logger = logging.getLogger(__name__)


class _AsyncNanWarner:
    """One-shot NaN monitor: device-side detection lands in pinned host
    memory without any stream sync; the host reads the (slightly stale) flag
    on a later call, warns once, and stops detecting.

    On ROCm/HIP the device probe (isnan/any/cast/add/copy) is sampled once
    every ``_PROBE_INTERVAL`` calls rather than on every call. ``check`` runs
    on the sampler hot path (once per decode step, every model); on AMD,
    enqueuing those extra kernels on every step measurably lowered
    single-batch decode ``fwd_occupancy`` (more non-overlapped per-step launch
    overhead). The probe only feeds a throttled diagnostic warning --
    sanitization itself is done by ``nan_to_num_`` in ``sanitize_nan_logits``
    independent of this -- so sampling it on a cadence keeps the hot path
    cheap while still surfacing a persistent NaN within at most
    ``_PROBE_INTERVAL`` steps. On CUDA the cadence is 1 (probe every call),
    preserving the original detection latency."""

    # AMD only: probe roughly every N calls. The cheap host-flag read still
    # happens every call, so a landed hit is reported on the very next call
    # after a probe. CUDA keeps the original every-call behavior (interval 1).
    _PROBE_INTERVAL = 128 if is_hip() else 1

    def __init__(self):
        self._dev = None
        self._host = None
        self._warned = False
        self._calls = 0

    def check(self, tensor: torch.Tensor, msg: str):
        if self._warned or not tensor.is_cuda:
            return

        # Report a hit enqueued on an earlier probe (pinned read, no sync).
        if self._host is not None and int(self._host[0]):
            logger.warning(
                "NaN detected in %s; values were sanitized before sampling. "
                "This usually indicates numerical overflow (e.g. fp16 "
                "activations) or an upstream bug producing NaN. "
                "Logged once; further occurrences are silent.",
                msg,
            )
            self._warned = True
            return

        # Only enqueue the device-side detection on a cadence to keep the
        # per-step kernel-launch overhead off the decode hot path.
        sample = self._calls % self._PROBE_INTERVAL == 0
        self._calls += 1
        if not sample:
            return

        if self._dev is None:
            self._dev = torch.zeros(1, dtype=torch.int32, device=tensor.device)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

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

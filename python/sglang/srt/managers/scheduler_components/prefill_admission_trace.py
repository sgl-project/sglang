import logging
import time
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PrefillAdmissionTrace:
    """One scheduler pass. A pass can end without launching a batch."""

    scan_id: int
    queued: int = 0
    visited: int = 0
    admitted: int = 0
    head_wait_ms: float | None = None
    skips: Counter[str] = field(default_factory=Counter)
    stop: str = "queue_exhausted"

    def observe_head(self, req):
        ts = req.time_stats.wait_queue_entry_time
        self.head_wait_ms = round((time.perf_counter() - ts) * 1000, 1) if ts else None

    def finish(self, last_forward, batch):
        if batch is not None:
            batch.prefill_admission_scan_id = self.scan_id
        logger.info(
            "DP_ADMISSION scan=%d after_fct=%d queue=%d visited=%d admitted=%d "
            "head_wait_ms=%s skips=%s stop=%s",
            self.scan_id,
            last_forward,
            self.queued,
            self.visited,
            self.admitted,
            self.head_wait_ms,
            ",".join(f"{k}:{v}" for k, v in self.skips.items()) or "none",
            self.stop,
        )

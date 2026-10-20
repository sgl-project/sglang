"""Process-local counters for timeout observations in the shared KV protocol."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from prometheus_client import Counter


def get_kv_transfer_timeout_counter(*, enabled: bool) -> Optional[Counter]:
    return _get_timeout_counter() if enabled else None


@lru_cache(maxsize=1)
def _get_timeout_counter() -> Counter:
    # Import after the launcher sets PROMETHEUS_MULTIPROC_DIR, as other collectors do.
    from prometheus_client import Counter

    counter = Counter(
        "sglang:kv_transfer_timeouts_total",
        "KV timeout observations per sender/receiver, not unique failed requests.",
        ["stage"],
    )
    for stage in ("bootstrap", "transfer"):
        counter.labels(stage=stage).inc(0)
    return counter

"""page_size is DERIVED from the backend, never a free sweep axis.

SGLang's ``overrides.py`` (``_mla_backend_page_constraints`` / ``_fa4_page_constraint``)
silently snaps ``--page-size`` per backend on selection. Sweeping (backend x page_size)
as an independent cross-product would waste GPU-hours on combinations the dispatcher
overrides anyway. So Attune resolves page_size from the chosen backend using the same
table the engine enforces.

The multi-value backends keep a small OPT-IN secondary sweep (the interim design's
carve-out): when a backend permits more than one page size, the prefix-cache-vs-throughput
tradeoff can matter, so those specific page sizes are worth benchmarking — but only for
those backends, not as a global axis.

Values mirror SGLang main (verify against the current ``overrides.py`` before landing).
"""

from __future__ import annotations

from typing import List

# backend -> allowed page sizes. First entry is the default snap.
PAGE_SIZE_TABLE = {
    "flashmla": [64],
    "cutlass_mla": [128],
    "trtllm_mla": [32, 64],
    "tokenspeed_mla": [32, 64],
    "cutedsl_mla": [32, 64],
    "trtllm_mha": [16, 32, 64],
    "fa4": [128],  # non-MLA fa4 path
    "hpc_ops": [64],  # SM90 only
}
# Backends with no hard snap fall back to the engine default (commonly 1 for paged decode).
DEFAULT_PAGE_SIZE = 1


def default_page_size(backend: str) -> int:
    """The single page size the dispatcher would snap this backend to."""
    return PAGE_SIZE_TABLE.get(backend, [DEFAULT_PAGE_SIZE])[0]


def page_sizes_to_probe(backend: str, secondary_sweep: bool = False) -> List[int]:
    """Page sizes Attune should benchmark for this backend.

    Default: just the snapped value (one cell). With ``secondary_sweep`` and a
    multi-value backend, probe all permitted sizes (the opt-in carve-out).
    """
    allowed = PAGE_SIZE_TABLE.get(backend, [DEFAULT_PAGE_SIZE])
    if secondary_sweep and len(allowed) > 1:
        return list(allowed)
    return [allowed[0]]

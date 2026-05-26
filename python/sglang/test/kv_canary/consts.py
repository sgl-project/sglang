from __future__ import annotations

from typing import Final

# SWA e2e pool sizing for 8 reqs × ~7K prompt + 2K decode, SWA window 1024.
# FULL pool = max-total-tokens; must fit 8 × (7000 + 2048) = 72_384 to avoid preempt.
# SWA  pool = max-total-tokens × ratio;
#   ≥ 8 × 1024 = 8192   (else deadlock — in-flight footprint exceeds capacity);
#   < 8 × 7000 = 56_000 (else allocator never recycles → swa_full_idx_divergence stays 0).
# Pick FULL=81920 (72_384 + headroom), SWA=16384 (2× in-flight floor, well under 56K).
SWA_POOL_SERVER_ARGS: Final[tuple[str, ...]] = (
    "--max-total-tokens",
    "81920",
    "--swa-full-tokens-ratio",
    "0.2",
)

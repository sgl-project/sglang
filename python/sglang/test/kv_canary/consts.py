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

DSV4_DEEPEP_CONFIG: Final[str] = (
    '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
)

DSV4_POOL_SERVER_ARGS: Final[tuple[str, ...]] = (
    "--trust-remote-code",
    "--tp",
    "4",
    "--dp",
    "4",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--cuda-graph-max-bs-decode",
    "128",
    "--max-running-requests",
    "256",
    "--deepep-config",
    DSV4_DEEPEP_CONFIG,
    "--mem-fraction-static",
    "0.7",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
)

DSV4_POOL_SERVER_ENV: Final[dict[str, str]] = {
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

from __future__ import annotations

from typing import Final

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
    "--cuda-graph-max-bs",
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

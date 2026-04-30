from __future__ import annotations

import signal
from collections.abc import Iterable
from typing import Any

from sglang.srt.relaykv import RelayKVConfig, make_shadow_plan


KV_BYTES_PER_TOKEN = 28672
SEQ_LEN = 2535
BUDGET_BLOCK_SIZE = 128

AVAILABLE_KV_BUDGET_MIB = (512.0, 1024.0, 2048.0)
WORKING_BUDGET_TOKENS = (1024, 2048, 4096)
RECENT_WINDOWS = (512, 768, 1024, 2048)
ANCHOR_BLOCKS = (0, 4, 8)
RETRIEVAL_TOP_K = (0, 2, 4, 8)

COLUMNS = (
    "budget_source",
    "available_kv_budget_mib",
    "kv_working_budget_tokens",
    "seq_len",
    "planned_resident_tokens",
    "planned_cold_tokens",
    "estimated_resident_ratio",
    "recent_window_tokens",
    "anchor_blocks",
    "budget_block_size",
    "anchor_budget_tokens",
    "retrieval_budget_tokens",
    "retrieval_block_budget",
    "retrieval_top_k_requested",
    "retrieval_top_k_effective",
    "budget_overflow",
    "budget_policy_reason",
)


def _plans() -> Iterable[dict[str, Any]]:
    for available_mib in AVAILABLE_KV_BUDGET_MIB:
        for recent_window in RECENT_WINDOWS:
            for anchor_blocks in ANCHOR_BLOCKS:
                for retrieval_top_k in RETRIEVAL_TOP_K:
                    yield _row(
                        make_shadow_plan(
                            SEQ_LEN,
                            RelayKVConfig(
                                enabled=True,
                                mode="shadow",
                                available_kv_budget_mib=available_mib,
                                recent_window=recent_window,
                                anchor_blocks=anchor_blocks,
                                budget_block_size=BUDGET_BLOCK_SIZE,
                                retrieval_top_k=retrieval_top_k,
                            ),
                            kv_bytes_per_token=KV_BYTES_PER_TOKEN,
                        )
                    )

    for working_budget_tokens in WORKING_BUDGET_TOKENS:
        for recent_window in RECENT_WINDOWS:
            for anchor_blocks in ANCHOR_BLOCKS:
                for retrieval_top_k in RETRIEVAL_TOP_K:
                    yield _row(
                        make_shadow_plan(
                            SEQ_LEN,
                            RelayKVConfig(
                                enabled=True,
                                mode="shadow",
                                kv_working_budget_tokens=working_budget_tokens,
                                recent_window=recent_window,
                                anchor_blocks=anchor_blocks,
                                budget_block_size=BUDGET_BLOCK_SIZE,
                                retrieval_top_k=retrieval_top_k,
                            ),
                            kv_bytes_per_token=KV_BYTES_PER_TOKEN,
                        )
                    )


def _row(plan: Any) -> dict[str, Any]:
    return {
        "budget_source": plan.kv_working_budget_source,
        "available_kv_budget_mib": _format_mib(plan.available_kv_budget_mib),
        "kv_working_budget_tokens": plan.kv_working_budget_tokens,
        "seq_len": plan.seq_len,
        "planned_resident_tokens": plan.planned_resident_tokens,
        "planned_cold_tokens": plan.planned_cold_tokens,
        "estimated_resident_ratio": f"{plan.estimated_resident_ratio:.4f}",
        "recent_window_tokens": plan.recent_window_tokens,
        "anchor_blocks": plan.anchor_blocks,
        "budget_block_size": plan.budget_block_size,
        "anchor_budget_tokens": plan.anchor_budget_tokens,
        "retrieval_budget_tokens": plan.retrieval_budget_tokens,
        "retrieval_block_budget": plan.retrieval_block_budget,
        "retrieval_top_k_requested": plan.retrieval_top_k_requested,
        "retrieval_top_k_effective": plan.retrieval_top_k_effective,
        "budget_overflow": str(plan.budget_overflow).lower(),
        "budget_policy_reason": plan.budget_policy_reason,
    }


def _format_mib(value: float) -> str:
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return str(value)


def _print_markdown_table(rows: Iterable[dict[str, Any]]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |")


def main() -> None:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    _print_markdown_table(_plans())


if __name__ == "__main__":
    main()

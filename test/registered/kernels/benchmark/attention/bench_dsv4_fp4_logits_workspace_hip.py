"""MI35x endurance benchmark for the adaptive DSV4 FP4 logits workspace.

This isolates the score-output/top-k lifetime that caused #37660: contexts grow
monotonically while every row chunk reuses one stable allocation.  It reports
allocated/reserved memory and latency for each geometry; run the full model
nightlies for end-to-end score-kernel measurements.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from sglang.srt.layers.attention.dsv4.fp4_logits_workspace import (
    FP4LogitsWorkspace,
    fp4_logits_width_for_context,
    plan_fp4_logits_workspace,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=60,
    stage="jit-kernel-benchmark",
    runner_config="amd",
    disabled="Manual long-context workspace endurance benchmark",
)


def benchmark(
    *,
    contexts: list[int],
    row_counts: list[int],
    budget_mb: int,
    iterations: int,
) -> list[dict]:
    device = torch.device("cuda")
    max_width = fp4_logits_width_for_context(max(contexts), 256)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    plan = plan_fp4_logits_workspace(
        max_seq_len=max_width,
        max_query_rows=max(row_counts),
        runtime_headroom_bytes=free_bytes,
        free_memory_fraction=0.2,
        max_workspace_bytes=budget_mb << 20,
    )
    workspace = FP4LogitsWorkspace(plan=plan, device=device)
    results = []
    try:
        for context_len in contexts:
            width = fp4_logits_width_for_context(context_len, 256)
            for rows in row_counts:
                chunk_rows = min(rows, workspace.rows_per_chunk(width))
                torch.cuda.synchronize()
                allocated_before = torch.cuda.memory_allocated(device)
                reserved_before = torch.cuda.memory_reserved(device)
                started = time.perf_counter()
                checksum = None
                for _ in range(iterations):
                    for start in range(0, rows, chunk_rows):
                        count = min(chunk_rows, rows - start)
                        with workspace.acquire(count, width) as logits:
                            # Model the producer write and immediate top-k
                            # consumer without allocating another full rectangle.
                            logits.zero_()
                            checksum = torch.topk(
                                logits,
                                k=min(512, width),
                                dim=-1,
                            ).indices
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000 / iterations
                results.append(
                    {
                        "context": context_len,
                        "rows": rows,
                        "width": width,
                        "chunk_rows": chunk_rows,
                        "chunks": -(-rows // chunk_rows),
                        "latency_ms": elapsed_ms,
                        "workspace_mb": workspace.capacity_bytes / (1 << 20),
                        "allocated_delta_mb": (
                            torch.cuda.memory_allocated(device) - allocated_before
                        )
                        / (1 << 20),
                        "reserved_delta_mb": (
                            torch.cuda.memory_reserved(device) - reserved_before
                        )
                        / (1 << 20),
                        "checksum": int(checksum[0, 0].item()),
                    }
                )
    finally:
        workspace.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=[8192, 131072, 524288, 1048576],
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 64, 256, 512])
    parser.add_argument("--budget-mb", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if not is_hip() or not is_gfx95_supported():
        print("[skip] requires a gfx95 HIP GPU")
        return
    print(
        json.dumps(
            benchmark(
                contexts=args.contexts,
                row_counts=args.rows,
                budget_mb=args.budget_mb,
                iterations=args.iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

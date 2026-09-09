"""Compare completed GPU results and retain a CPU archive on correctness failure."""

import os
import time
from pathlib import Path

import torch

from .oracle import assert_combine_matches, assert_dispatch_matches


def compare(batch, rank, received, counters, combined, *, identity=False):
    try:
        assert_dispatch_matches(batch, rank, received, counters)
        assert_combine_matches(batch, rank, combined, identity=identity)
    except AssertionError:
        root = Path(os.environ.get("NCCL_EP_REPORT_DIR", "."))
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"mismatch-rank{rank}-{time.time_ns()}.pt"
        torch.save(
            {
                "tokens": tuple(x.cpu() for x in batch.tokens),
                "ids": tuple(x.cpu() for x in batch.expert_ids),
                "weights": tuple(x.cpu() for x in batch.weights),
                "num_experts": batch.num_experts,
                "rank": rank,
                "identity": identity,
                "received": received.cpu(),
                "counters": counters.cpu(),
                "combined": combined.cpu(),
            },
            path,
        )
        print(f"Saved correctness failure: {path}", flush=True)
        raise

"""Measure CPU SHM receive latency, optionally including an H2D transfer.

Run from the repository root with PYTHONPATH=python:
    python benchmark/multimodal/bench_shm_receiver.py --device cuda

Sender allocation/copy is excluded from both measurements. The clone mode
uses the existing portable receive path as the baseline.
"""

import argparse
import json
import pickle
import statistics
import time

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import mm_utils


def measure(*, feature, device):
    sender = mm_utils.ShmPointerMMData(feature)
    try:
        payload = pickle.dumps(sender)
        started = time.perf_counter()
        receiver = pickle.loads(payload)
        tensor = receiver.materialize()
        received = time.perf_counter()
        if device == "cuda":
            result = tensor.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
        else:
            result = tensor
        finished = time.perf_counter()
        assert torch.equal(result.cpu(), feature)
        return (received - started) * 1000, (finished - started) * 1000
    finally:
        sender.close_and_unlink()


def main(args):
    torch.set_num_threads(1)
    if args.device == "cuda":
        torch.empty(1, device="cuda")
    for size_mb in args.sizes_mb:
        feature = torch.ones(size_mb * 1024 * 1024 // 4)
        for mode, use_views in (("clone", False), ("view", True)):
            with envs.SGLANG_ENABLE_MM_SHM_ZERO_COPY.override(use_views):
                measure(feature=feature, device=args.device)
                results = [
                    measure(feature=feature, device=args.device)
                    for _ in range(args.repeats)
                ]
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "feature_mb": size_mb,
                            "device": args.device,
                            "receive_ms_median": statistics.median(
                                row[0] for row in results
                            ),
                            "receive_and_transfer_ms_median": statistics.median(
                                row[1] for row in results
                            ),
                        }
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    main(parser.parse_args())

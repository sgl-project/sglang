"""Compare inline CPU hashing with bounded offload and an event-loop heartbeat.

Run from the repository root with PYTHONPATH=python:
    python benchmark/multimodal/bench_hash_executor.py --sizes-mb 1 16 64
"""

import argparse
import asyncio
import json
import statistics
import time

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.hash_executor import MultimodalHashExecutor


async def measure(*, feature, requests, executor):
    items = [
        MultimodalDataItem(modality=Modality.IMAGE, feature=feature)
        for _ in range(requests)
    ]
    gaps = []
    finished = False

    async def heartbeat():
        previous = time.perf_counter()
        while not finished:
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            gaps.append(now - previous)
            previous = now

    ticker = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    started = time.perf_counter()
    if executor is None:
        for item in items:
            item.set_pad_value()
    else:
        await asyncio.gather(*(executor.set_pad_values([item]) for item in items))
    elapsed = time.perf_counter() - started
    await asyncio.sleep(0.002)
    finished = True
    await ticker
    assert len({(item.hash, item.pad_value) for item in items}) == 1
    return elapsed * 1000, max(gaps) * 1000


async def main(args):
    torch.set_num_threads(1)
    executor = MultimodalHashExecutor()
    try:
        for size_mb in args.sizes_mb:
            feature = torch.ones(size_mb * 1024 * 1024 // 4)
            for mode, pool in (("inline", None), ("offload", executor)):
                await measure(feature=feature, requests=args.requests, executor=pool)
                results = [
                    await measure(
                        feature=feature, requests=args.requests, executor=pool
                    )
                    for _ in range(args.repeats)
                ]
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "feature_mb": size_mb,
                            "requests": args.requests,
                            "wall_ms_median": statistics.median(
                                row[0] for row in results
                            ),
                            "max_heartbeat_gap_ms_median": statistics.median(
                                row[1] for row in results
                            ),
                        }
                    )
                )
    finally:
        executor.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    asyncio.run(main(parser.parse_args()))

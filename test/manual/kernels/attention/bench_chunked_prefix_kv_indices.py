"""Compare the 1D and 2D prefix-index kernels on reused CUDA-graph buffers.

Run from the repository root with SGLang's GPU dependencies installed:
    PYTHONPATH=python python test/manual/kernels/attention/bench_chunked_prefix_kv_indices.py --output /tmp/prefix-indices.json

This measures the index-copy operator, not full-model serving performance.
"""

import argparse
import hashlib
import inspect
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton

from sglang.kernels.ops.attention.kv_indices import (
    create_chunked_prefix_cache_kv_indices,
    create_chunked_prefix_cache_kv_indices_2d,
)


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    source = Path(inspect.getsourcefile(create_chunked_prefix_cache_kv_indices_2d.fn))
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "base_commit": commit,
        "working_tree_status": dirty,
        "kernel_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "protocol": "5 eager warmups; 20 calls per graph; 3 graph warmups; median of 7 timed graph replays; baseline then candidate; reused buffers",
        "cases": [],
    }
    torch.manual_seed(0)
    print("| Batch | Prefix-chunk lengths | Before (us) | After (us) | Change |")
    print("|---:|---|---:|---:|---:|")
    for tokens in (17, 16384, 49152, 114688, 131072):
        for lengths, starts, pools in (
            ([tokens], [7], [0]),
            ([17, tokens - 3, 0], [2, 5, 0], [2, 0, 1]),
        ):
            batch, width = len(lengths), tokens + 16
            table = torch.arange(batch * width, dtype=torch.int32, device="cuda").view(
                batch, width
            )
            reqs = torch.tensor(pools, dtype=torch.int32, device="cuda")
            begin = torch.tensor(starts, dtype=torch.int32, device="cuda")
            lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            indptr = torch.tensor(
                [0] + lengths, dtype=torch.int32, device="cuda"
            ).cumsum(0, dtype=torch.int32)
            out = torch.empty(sum(lengths), dtype=torch.int32, device="cuda")
            common = (table, reqs, begin, lens, indptr, out, table.stride(0))

            def before():
                create_chunked_prefix_cache_kv_indices[(batch,)](*common)

            def after():
                create_chunked_prefix_cache_kv_indices_2d[
                    (batch, triton.cdiv(max(lengths), 1024))
                ](*common, BLOCK_SIZE=1024)

            reference = torch.cat(
                [table[p, s : s + n] for p, s, n in zip(pools, starts, lengths)]
            )
            for fn in (before, after):
                out.fill_(-1)
                fn()
                torch.testing.assert_close(out, reference, rtol=0, atol=0)
            old, new = measure(before), measure(after)
            change = 100 * (new["median_us"] / old["median_us"] - 1)
            result["cases"].append(
                {
                    "batch": batch,
                    "lengths": lengths,
                    "starts": starts,
                    "pools": pools,
                    "before": old,
                    "after": new,
                    "change_percent": change,
                    "correctness": "exact int32 match against reference",
                }
            )
            print(
                f"| {batch} | {lengths} | {old['median_us']:.3f} | {new['median_us']:.3f} | {change:+.1f}% |",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Results:", args.output)


if __name__ == "__main__":
    main()

"""Measure how short neighboring requests affect a long paged decode request."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import flashinfer
import torch
from flashinfer.jit import env as jit_env


def measure(functions, repeats, replays):
    for fn in functions:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in functions:
            fn()
    for _ in range(4):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / replays / len(functions))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-length", type=int, default=168960)
    parser.add_argument("--short-length", type=int, default=64)
    parser.add_argument("--heads", default="4,8,16")
    parser.add_argument("--batches", default="1,2,4,8,9,16")
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(0)
    header = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/trtllm/fmha/fmhaKernels.cuh"
    header_text = header.read_text()
    metadata = {
        "torch": torch.__version__,
        "flashinfer": flashinfer.__version__,
        "device": properties.name,
        "sm_count": properties.multi_processor_count,
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "launcher_excerpt": "\n".join(header_text.splitlines()[590:725]),
        "args": vars(args),
        "mode": "static diagnostic, unchanged library, CUDA graphs",
        "accuracy": "PENDING; diagnostic does not promote a new implementation",
    }
    workspace = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    counter = torch.zeros(8 * 1024**2, device="cuda", dtype=torch.uint8)
    batches = list(map(int, args.batches.split(",")))
    long_pages = (args.long_length + 63) // 64
    short_pages = (args.short_length + 63) // 64
    page_count = long_pages + max(batches) * short_pages
    banks = [
        tuple(
            torch.randn(page_count, 1, 64, 256, device="cuda").to(torch.float8_e4m3fn)
            for _ in range(2)
        )
        for _ in range(args.banks)
    ]
    records = []
    for heads in map(int, args.heads.split(",")):
        # One query pool makes the long request identical in every batch.
        q_pool = torch.randn(max(batches), 7, heads, 256, device="cuda").to(
            torch.float8_e4m3fn
        )
        cases = [(k, "mixed") for k in batches]
        cases += [
            (k, "short")
            for k in sorted(set(batches + [k - 1 for k in batches if k > 1]))
        ]
        # Deterministic permutation keeps a length or batch from owning one time range.
        generator = torch.Generator().manual_seed(args.seed + heads)
        order = torch.randperm(len(cases), generator=generator).tolist()
        for index in order:
            batch, kind = cases[index]
            q = q_pool[:batch].reshape(batch * 7, heads, 256)
            lengths = [args.short_length] * batch
            if kind == "mixed":
                lengths[0] = args.long_length
            seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            table = torch.zeros(batch, long_pages, device="cuda", dtype=torch.int32)
            for row in range(batch):
                if kind == "mixed" and row == 0:
                    table[row] = torch.arange(
                        long_pages, device="cuda", dtype=torch.int32
                    )
                else:
                    table[row, :short_pages] = torch.arange(
                        long_pages + row * short_pages,
                        long_pages + (row + 1) * short_pages,
                        device="cuda",
                        dtype=torch.int32,
                    )
            outputs = [
                torch.empty(q.shape, device="cuda", dtype=torch.bfloat16) for _ in banks
            ]
            functions = [
                lambda kv=kv, out=out, q=q, table=table, seq=seq: (
                    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                        q,
                        kv,
                        workspace,
                        table,
                        seq,
                        262144,
                        bmm1_scale=1 / 16,
                        bmm2_scale=1.0,
                        out=out,
                        q_len_per_req=7,
                        multi_ctas_kv_counter_buffer=counter,
                    )
                )
                for kv, out in zip(banks, outputs)
            ]
            samples = measure(functions, args.repeats, args.replays)
            record = {
                "heads": heads,
                "batch": batch,
                "kind": kind,
                "lengths": lengths,
                "samples_us": samples,
                "median_us": statistics.median(samples),
                "min_us": min(samples),
                "max_us": max(samples),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(args.output).write_text(
                json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()

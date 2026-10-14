#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from common import require_single_visible_gpu

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead


def benchmark(
    output_head: BeamTrieOutputHead,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    warmups: int,
    iterations: int,
) -> dict:
    def project(states: torch.Tensor, selected_weight: torch.Tensor) -> torch.Tensor:
        return torch.matmul(states, selected_weight.T)

    def transform(logits: torch.Tensor) -> torch.Tensor:
        return logits

    for _ in range(warmups):
        output_head._stream_normalizer(hidden_states, weight, project, transform)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    elapsed_ms = []
    result = None
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = output_head._stream_normalizer(
            hidden_states, weight, project, transform
        )
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    assert result is not None
    return {
        "mean_ms": statistics.fmean(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
        "times_ms": elapsed_ms,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark trie-head full-vocabulary normalization."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rows", type=int, nargs="+", default=[2000, 8000, 20000])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=176245)
    parser.add_argument("--token-start", type=int, default=151669)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--num-codebooks", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=args.token_start,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
    )
    weight = torch.empty(
        args.vocab_size, args.hidden_size, dtype=dtype, device=device
    ).normal_(mean=0.0, std=0.02)

    report = {
        "timestamp_unix": time.time(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "rows": {},
    }

    for rows in args.rows:
        hidden_states = torch.empty(
            rows, args.hidden_size, dtype=dtype, device=device
        ).normal_(mean=0.0, std=0.02)
        row_report = {}
        outputs = {}
        chunked_label = f"chunked_{args.chunk_size}"
        for label, chunk_size in ((chunked_label, args.chunk_size), ("full", 0)):
            result = benchmark(
                BeamTrieOutputHead(
                    config, vocab_size=args.vocab_size, chunk_size=chunk_size
                ),
                hidden_states,
                weight,
                args.warmups,
                args.iterations,
            )
            outputs[label] = result.pop("result")
            row_report[label] = result

        difference = (outputs["full"].float() - outputs[chunked_label].float()).abs()
        row_report["parity"] = {
            "exact_equal": torch.equal(outputs["full"], outputs["chunked_32768"]),
            "max_abs": difference.max().item(),
            "mean_abs": difference.mean().item(),
        }
        row_report["full_speedup"] = (
            row_report[chunked_label]["mean_ms"] / row_report["full"]["mean_ms"]
        )
        report["rows"][str(rows)] = row_report
        del hidden_states, outputs
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

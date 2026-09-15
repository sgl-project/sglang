"""Compare default, H200, and H20-3e block-FP8 configs on an H20-3e.

Example:
    python benchmark/kernels/deepseek/benchmark_h20_block_fp8.py \
        --output h20_block_fp8.jsonl

This measures individual GEMMs, not end-to-end model throughput. Run on an
otherwise idle GPU. Inputs are synthetic; no model weights are needed.
"""

import argparse
import json
import pathlib
import statistics
from unittest.mock import patch

import torch
import triton

from sglang.kernels.ops.quantization import fp8_kernel

SHAPES = [
    (1792, 5120),
    (25600, 6144),
    (4096, 1280),
    (5120, 1024),
    (5120, 288),
    (576, 5120),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=98765)
    args = parser.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA H20-3e"
    assert args.trials > 0 and args.rep_ms > 0 and all(m > 0 for m in args.batch_sizes)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    config_dir = pathlib.Path(fp8_kernel.__file__).parent / "configs"
    with args.output.open("x") as output:

        def emit(row):
            text = json.dumps(row)
            output.write(text + "\n")
            output.flush()
            print(text, flush=True)

        emit(
            {
                "kind": "environment",
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "seed": args.seed,
                "batch_sizes": args.batch_sizes,
            }
        )
        for n, k in SHAPES:
            b = (torch.randn(n, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
            bs = torch.rand(n // 32, k // 32, device="cuda") + 0.1
            b_dequant = b.float() * bs.repeat_interleave(32, 0).repeat_interleave(32, 1)
            tables = {"default": None}
            for device in ("NVIDIA_H200", "NVIDIA_H20-3e"):
                path = (
                    config_dir
                    / f"N={n},K={k},device_name={device},dtype=fp8_w8a8,block_shape=[32, 32].json"
                )
                if path.exists():
                    tables[device] = {
                        int(m): cfg for m, cfg in json.loads(path.read_text()).items()
                    }
            assert "NVIDIA_H20-3e" in tables, (
                "Install the H20-3e configuration files before benchmarking"
            )
            for m in args.batch_sizes:
                a = (torch.randn(m, k, device="cuda") * 0.2).to(torch.float8_e4m3fn)
                scales = (torch.rand(k // 32, m, device="cuda") + 0.1).T
                reference = (
                    (a.float() * scales.repeat_interleave(32, 1)) @ b_dequant.T
                ).to(torch.bfloat16)
                results = {}
                # Compile and validate every variant before timing any variant.
                for label, table in tables.items():
                    with patch.object(
                        fp8_kernel, "get_w8a8_block_fp8_configs", return_value=table
                    ):
                        actual = fp8_kernel.w8a8_block_fp8_matmul_triton(
                            a, b, scales, bs, [32, 32], torch.bfloat16
                        )
                    relative_l2 = (
                        torch.linalg.vector_norm(actual.float() - reference.float())
                        / torch.linalg.vector_norm(reference.float())
                    ).item()
                    assert torch.isfinite(actual).all() and relative_l2 < 0.01
                    results[label] = {
                        "config": None
                        if table is None
                        else table[min(table, key=lambda key: abs(key - m))],
                        "relative_l2": relative_l2,
                        "trials_us": [],
                    }
                for trial in range(args.trials):
                    labels = list(tables)
                    # Rotate variant order to reduce systematic timing-order bias.
                    labels = (
                        labels[trial % len(labels) :] + labels[: trial % len(labels)]
                    )
                    for label in labels:
                        with patch.object(
                            fp8_kernel,
                            "get_w8a8_block_fp8_configs",
                            return_value=tables[label],
                        ):

                            def invoke():
                                return fp8_kernel.w8a8_block_fp8_matmul_triton(
                                    a, b, scales, bs, [32, 32], torch.bfloat16
                                )

                            elapsed = (
                                triton.testing.do_bench_cudagraph(
                                    invoke, rep=args.rep_ms
                                )
                                * 1000
                            )
                        results[label]["trials_us"].append(elapsed)
                for result in results.values():
                    result["median_us"] = statistics.median(result["trials_us"])
                emit(
                    {"kind": "measurement", "m": m, "n": n, "k": k, "results": results}
                )
        emit({"kind": "complete"})


if __name__ == "__main__":
    main()

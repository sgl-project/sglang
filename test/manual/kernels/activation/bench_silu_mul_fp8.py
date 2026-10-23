# SPDX-License-Identifier: Apache-2.0
"""Compare separate BF16 SiLU-mul + FP8 quantization with the fused producer.

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python .venv/bin/python test/manual/kernels/activation/bench_silu_mul_fp8.py
Uses cold-L2 rotating CUDA graphs; does not change runtime configuration.
"""

import argparse
import json
from pathlib import Path

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation.silu_mul_fp8 import silu_mul_fp8
from sglang.kernels.ops.diffusion import fused_silu_mul_bitexact
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8


def separate(gate, up):
    return sglang_per_token_quant_fp8(fused_silu_mul_bitexact(gate, up))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/silu-mul-fp8-bench.json")
    args = parser.parse_args()
    assert torch.cuda.get_device_capability() == (8, 9), "SM89 reference required"
    torch.manual_seed(42)
    results = []
    for rows, hidden in [(4096, 12288), (4096, 4096), (22, 12288)]:
        gate = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        expected = separate(gate, up)
        actual = silu_mul_fp8(gate, up)
        assert torch.equal(expected[0].view(torch.uint8), actual[0].view(torch.uint8))
        assert torch.equal(expected[1], actual[1])
        for repeat in range(3):
            record = {"rows": rows, "hidden": hidden, "repeat": repeat}
            order = [("separate", separate), ("fused", silu_mul_fp8)]
            if repeat % 2:
                order.reverse()
            for name, fn in order:
                record[name + "_ms"] = (
                    marker.do_bench(
                        fn,
                        input_args=(gate, up),
                        warmup_iters=3,
                        replay_iters=30,
                        disable_log_bandwidth=True,
                    ).times[0]
                    * 1000
                )
            results.append(record)
            print(json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

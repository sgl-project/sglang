"""Benchmark DSA query quantization plus head-gate scale application.

This compares the old decode sequence:
  act_quant(query) + compiled(weights.unsqueeze(-1) * q_scale * softmax_scale)

against the fused Triton helper:
  act_quant_apply_scale(query, weights, softmax_scale)

For GLM5 DSA decode the shape is [B, 32, 128], where B is the decode
batch/CUDA-graph bucket size.
"""

from typing import Tuple

import torch
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.srt.layers.attention.dsa.triton_kernel import (
    act_quant,
    act_quant_apply_scale,
)


INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
BLOCK_SIZE = 128
SOFTMAX_SCALE = INDEX_HEAD_DIM**-0.5
SCALE_FMT = "ue8m0"

BATCH_SIZE_RANGE = get_benchmark_range(
    full_range=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    ci_range=[1, 32, 256],
)

LINE_VALS = ["split_compile", "fused"]
LINE_NAMES = ["act_quant + torch.compile scale", "fused act_quant_apply_scale"]
STYLES = [("red", "--"), ("green", "-")]


def apply_scale_only(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    return weights.unsqueeze(-1) * q_scale * softmax_scale


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=BATCH_SIZE_RANGE,
        line_arg="provider",
        line_vals=LINE_VALS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="dsa-act-quant-apply-scale-performance",
        args={},
    )
)
def benchmark(batch_size: int, provider: str) -> Tuple[float, float, float]:
    torch.manual_seed(0)
    query = torch.randn(
        (batch_size, INDEX_N_HEADS, INDEX_HEAD_DIM),
        dtype=torch.bfloat16,
        device=DEFAULT_DEVICE,
    )
    weights = torch.randn(
        (batch_size, INDEX_N_HEADS),
        dtype=torch.float32,
        device=DEFAULT_DEVICE,
    )

    compiled_apply_scale = torch.compile(
        apply_scale_only, fullgraph=True, dynamic=True
    )

    if provider == "split_compile":
        q_probe, scale_probe = act_quant(query, BLOCK_SIZE, SCALE_FMT)
        compiled_apply_scale(weights, scale_probe, SOFTMAX_SCALE)
        torch.cuda.synchronize()

        def fn():
            q, q_scale = act_quant(query, BLOCK_SIZE, SCALE_FMT)
            scaled_weights = compiled_apply_scale(weights, q_scale, SOFTMAX_SCALE)
            return q, scaled_weights

    elif provider == "fused":
        act_quant_apply_scale(query, weights, SOFTMAX_SCALE, BLOCK_SIZE, SCALE_FMT)
        torch.cuda.synchronize()

        def fn():
            return act_quant_apply_scale(
                query, weights, SOFTMAX_SCALE, BLOCK_SIZE, SCALE_FMT
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=DEFAULT_QUANTILES
    )
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


def check_correctness() -> None:
    for batch_size in BATCH_SIZE_RANGE:
        torch.manual_seed(0)
        query = torch.randn(
            (batch_size, INDEX_N_HEADS, INDEX_HEAD_DIM),
            dtype=torch.bfloat16,
            device=DEFAULT_DEVICE,
        )
        weights = torch.randn(
            (batch_size, INDEX_N_HEADS),
            dtype=torch.float32,
            device=DEFAULT_DEVICE,
        )
        q_ref, q_scale = act_quant(query, BLOCK_SIZE, SCALE_FMT)
        weights_ref = weights.unsqueeze(-1) * q_scale * SOFTMAX_SCALE
        q, scaled_weights = act_quant_apply_scale(
            query, weights, SOFTMAX_SCALE, BLOCK_SIZE, SCALE_FMT
        )
        torch.cuda.synchronize()
        assert torch.equal(q, q_ref)
        torch.testing.assert_close(scaled_weights, weights_ref, rtol=0, atol=0)


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmarking DSA act_quant + q-scale head-gate application")
    print(
        f"shape=[B, {INDEX_N_HEADS}, {INDEX_HEAD_DIM}], "
        f"block_size={BLOCK_SIZE}, scale_fmt={SCALE_FMT}"
    )
    print("=" * 80)
    check_correctness()
    benchmark.run(print_data=True)

"""Benchmark the W4AFP8 DeepEP low-latency requant against its previous geometry.

``legacy-geometry`` launches the current kernel with its previous launch
parameters (1024-element tile, 8 warps, 32 programs per expert); ``tuned`` goes
through the wrapper.  ``skew`` concentrates rows on one hot expert, which the
m-grid cannot see because ``expected_m`` is a dispatch-wide average.
"""

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.moe.ep_moe_kernels import (
    _fp8_per_token_quant_to_per_tensor_quant_kernel,
    fp8_per_token_to_per_tensor_quant_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FP8 = torch.float8_e4m3fn
K_SCALE_BLOCK_SIZE = 128
LEGACY_G_BLOCK = 8  # 1024 hidden elements
LEGACY_WARPS = 8
LEGACY_M_GRID = 32


def _expected_m(num_experts, dispatched_rows):
    """``dispatch_a`` reports ``(rows + num_experts) // num_experts``, one high
    at exact averages; benchmark with what production would pass."""
    return (dispatched_rows + num_experts) // num_experts


def _row_counts(num_experts, rows, skew):
    """One hot expert at ``skew * rows``, the rest share the fixed remainder; a
    dispatch redistributes rows, so skew cannot exceed ``num_experts``."""
    if skew == 1:
        return [rows] * num_experts
    total = num_experts * rows
    counts = [(total - rows * skew) // (num_experts - 1)] * num_experts
    counts[0] = rows * skew
    return counts


def _build(num_experts, m, k, rows, skew=1):
    x = (torch.randn(num_experts, m, k, device="cuda") * 4).to(FP8)
    # DeepEP returns the last two scale dims column-major (for TMA).
    x_scale = (
        torch.rand(num_experts, m, k // K_SCALE_BLOCK_SIZE, device="cuda")
        .add_(0.5)
        .permute(0, 2, 1)
        .contiguous()
        .permute(0, 2, 1)
    )
    counts = _row_counts(num_experts, rows, skew)
    masked_m = torch.tensor(counts, dtype=torch.int32, device="cuda")
    output_scale = torch.tensor([2.0], dtype=torch.float32, device="cuda")
    output = torch.empty((num_experts, m, k), dtype=FP8, device="cuda")
    return (
        x,
        x_scale,
        masked_m,
        output_scale,
        output,
        _expected_m(num_experts, sum(counts)),
    )


def _tuned(x, x_scale, masked_m, output_scale, output, expected_m):
    fp8_per_token_to_per_tensor_quant_triton(
        x=x,
        x_scale=x_scale,
        masked_m=masked_m,
        output_scale=output_scale,
        output=output,
        expected_rows=expected_m,
    )
    return output


def _legacy_geometry(x, x_scale, masked_m, output_scale, output, expected_m):
    num_groups = x.size(2) // K_SCALE_BLOCK_SIZE
    grid = (triton.cdiv(num_groups, LEGACY_G_BLOCK), LEGACY_M_GRID, x.size(0))
    _fp8_per_token_quant_to_per_tensor_quant_kernel[grid](
        x,
        x_scale,
        *x_scale.stride(),
        masked_m,
        output_scale,
        output,
        x.size(1),
        x.size(2),
        x.size(0),
        # row_cap = m keeps every row on its own expert, as the old launch did.
        x.size(1),
        K_SCALE_BLOCK_SIZE=K_SCALE_BLOCK_SIZE,
        G_BLOCK_SIZE=LEGACY_G_BLOCK,
        HAS_G_TAIL=(num_groups % LEGACY_G_BLOCK != 0),
        EXPERT_BLOCK=triton.next_power_of_2(x.size(0)),
        num_warps=LEGACY_WARPS,
    )
    return output


FN_MAP = {"tuned": _tuned, "legacy-geometry": _legacy_geometry}

# (hidden, local experts, padded rows): DeepSeek-V3 at EP8, then a 3584 hidden
# size at a low and a high local-expert count.
SHAPES = [(7168, 8, 1024), (3584, 8, 1024), (3584, 56, 256)]


@marker.parametrize("hidden,num_experts,m", SHAPES, [(7168, 8, 1024), (3584, 56, 256)])
@marker.parametrize("rows", [8, 32, 128, 256], [8, 32, 256])
@marker.parametrize("skew", [1, 4, 16], [1, 16])
@marker.benchmark("impl", ["tuned", "legacy-geometry"])
def benchmark(hidden: int, num_experts: int, m: int, rows: int, skew: int, impl: str):
    if skew > num_experts:
        marker.skip("one expert cannot hold more than the whole dispatch")
    if rows * skew > m:
        marker.skip("more live rows than the payload holds")
    args = _build(num_experts, m, hidden, rows, skew)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=args,
        graph_clone_args=(0, 1),
        memory_args=None,
        # Tensor-size bandwidth would be off by the padding factor.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()

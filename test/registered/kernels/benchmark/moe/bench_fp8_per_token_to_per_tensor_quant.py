"""Benchmark the W4AFP8 DeepEP low-latency requant against its previous geometry.

``legacy-geometry`` launches the current kernel with the launch parameters it
shipped with (a 1024-element tile, 8 warps, 32 programs per expert regardless of
how many rows are live), so this isolates the launch geometry and not the
scale-loading rewrite; ``tuned`` goes through the wrapper, which sizes the tile
from the part's warp width and the m-grid from the dispatcher's expected rows.

``skew`` covers the case the m-grid cannot see: ``expected_m`` is a global
average, so a hot expert holding ``skew`` times its share serializes its rows
over an m-grid sized for the average.
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
    """What the dispatcher would report for ``dispatched_rows`` in total.

    `_DeepEPDispatcherImplLowLatency.dispatch_a` computes
    ``(dispatched_rows + num_experts) // num_experts``, so an exact average
    arrives one higher; the production launch has to be benchmarked with that.
    It is an average over the whole dispatch, so it is the same number however
    those rows are distributed -- which is what makes ``skew`` interesting.
    """
    return (dispatched_rows + num_experts) // num_experts


def _row_counts(num_experts, rows, skew):
    """Live rows per expert: one hot expert at ``skew * rows``, the rest share.

    A batch dispatches a fixed number of rows, so skew redistributes them rather
    than adding any: the total stays at ``num_experts * rows`` and the average
    the m-grid is sized from does not move, however lopsided the routing is.
    That caps the achievable skew at ``num_experts`` -- one expert taking every
    row.
    """
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
        K_SCALE_BLOCK_SIZE=K_SCALE_BLOCK_SIZE,
        G_BLOCK_SIZE=LEGACY_G_BLOCK,
        HAS_G_TAIL=(num_groups % LEGACY_G_BLOCK != 0),
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
        # Only `rows` of the padded payload are live, so a tensor-size-derived
        # bandwidth would be off by the padding factor.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()

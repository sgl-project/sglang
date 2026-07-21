import sys

import torch
from sgl_kernel import rmsnorm as sgl_rmsnorm

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.jit_kernel.rmsnorm_per_token_group_quant_fp8 import (
    can_use_rmsnorm_per_token_group_quant_fp8,
    rmsnorm_per_token_group_quant_fp8_out,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-5
GROUP_SIZE = 128
TRACE_HIDDEN_SIZE = 2048
TRACE_ROW_STRIDE = 2624
HIDDEN_SIZES = [128, 640, 1536, TRACE_HIDDEN_SIZE, 4096]

# Sweep M at the traced reduction size, then sweep hidden size at the traced M.
# This covers both scaling dimensions without turning the benchmark into their
# full Cartesian product.
CASES = [
    (m, TRACE_HIDDEN_SIZE, "contiguous") for m in [1, 2, 4, 6, 8, 16, 32, 64, 128, 512]
]
CASES.extend(
    (6, hidden_size, "contiguous")
    for hidden_size in HIDDEN_SIZES
    if hidden_size != TRACE_HIDDEN_SIZE
)
CASES.append((6, TRACE_HIDDEN_SIZE, "trace_stride_2624"))


def _alloc_outputs(
    num_tokens: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_q = torch.empty(num_tokens, hidden_size, device=DEVICE, dtype=fp8_dtype)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_tokens, hidden_size),
        device=DEVICE,
        group_size=GROUP_SIZE,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    output_norm = torch.empty(num_tokens, hidden_size, device=DEVICE, dtype=DTYPE)
    return output_q, output_s, output_norm


def _unfused(
    backing: torch.Tensor,
    hidden_size: int,
    weight: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
) -> None:
    x = backing[:, :hidden_size]
    sgl_rmsnorm(x, weight, EPS, out=output_norm)
    per_token_group_quant_8bit_v2(
        output_norm,
        output_q,
        output_s,
        GROUP_SIZE,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
    )


def _fused(
    backing: torch.Tensor,
    hidden_size: int,
    weight: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    output_norm: torch.Tensor,
) -> None:
    x = backing[:, :hidden_size]
    rmsnorm_per_token_group_quant_fp8_out(
        x, weight, output_q, output_s, output_norm, EPS
    )


PROVIDERS = {"unfused": _unfused, "fused": _fused}


@marker.parametrize("num_tokens,hidden_size,layout", CASES)
@marker.benchmark("provider", ["unfused", "fused"])
def benchmark(num_tokens: int, hidden_size: int, layout: str, provider: str):
    torch.manual_seed(num_tokens + hidden_size)
    if layout == "trace_stride_2624":
        assert hidden_size == TRACE_HIDDEN_SIZE
        row_stride = TRACE_ROW_STRIDE
    else:
        row_stride = hidden_size

    # marker clones dense tensors. Clone the backing allocation and recreate
    # the logical view in the provider so trace_stride_2624 remains strided.
    backing = torch.randn(num_tokens, row_stride, device=DEVICE, dtype=DTYPE)
    x = backing[:, :hidden_size]
    assert x.stride() == (row_stride, 1)

    weight = torch.randn(hidden_size, device=DEVICE, dtype=DTYPE)
    output_q, output_s, output_norm = _alloc_outputs(num_tokens, hidden_size)
    graph_clone_args = (0, 2, 5) if provider == "unfused" else (0, 2)
    return marker.do_bench(
        PROVIDERS[provider],
        input_args=(
            backing,
            hidden_size,
            weight,
            output_q,
            output_s,
            output_norm,
        ),
        graph_clone_args=graph_clone_args,
        # Report effective algorithmic traffic. The unfused path writes and
        # then rereads output_norm, so count that intermediate read explicitly.
        memory_args=(x, weight),
        memory_output=(output_q, output_s, output_norm),
        extra_memory_args=(output_norm,) if provider == "unfused" else None,
    )


if __name__ == "__main__":
    if not can_use_rmsnorm_per_token_group_quant_fp8(DTYPE, TRACE_HIDDEN_SIZE):
        print(
            "[skip] RMSNorm + FP8 group-quant benchmark requires an SM10x "
            "Blackwell GPU with CUDA 12.9+."
        )
        sys.exit(0)
    benchmark.run()

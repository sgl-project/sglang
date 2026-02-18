import itertools
import os
from typing import Any, Dict, List

import torch
import triton
from sgl_kernel.test_utils import create_per_token_group_quant_test_data

from sglang.jit_kernel.benchmark.utils import (
    get_benchmark_range,
)
from sglang.jit_kernel.per_token_group_quant_8bit import (
    per_token_group_quant_8bit as sglang_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.bench_utils import bench_kineto

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

NUM_TESTS = 300 if IS_CI else 30

GROUP_SIZE_RANGE = [128]
DST_DTYPE_RANGE = [fp8_type_]

# ---- GEMM-like branch (num_ranks=None) ----
NUM_TOKENS_RANGE_GEMM = get_benchmark_range(
    full_range=[1, 4, 16, 64, 256, 768, 2048, 8192, 16384],
    ci_range=[768],
)
HIDDEN_DIM_RANGE_GEMM = [1536, 7168, 16384]
NUM_RANKS_RANGE_GEMM = [None]


FLAGS_GEMM_FULL: List[Dict[str, Any]] = [
    dict(
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
]
FLAGS_GEMM_CI: List[Dict[str, Any]] = [
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
]
FLAGS_RANGE_GEMM = get_benchmark_range(
    full_range=FLAGS_GEMM_FULL, ci_range=FLAGS_GEMM_CI
)

CONFIGS_GEMM = list(
    itertools.product(
        NUM_TOKENS_RANGE_GEMM,
        HIDDEN_DIM_RANGE_GEMM,
        GROUP_SIZE_RANGE,
        NUM_RANKS_RANGE_GEMM,
        DST_DTYPE_RANGE,
        FLAGS_RANGE_GEMM,
    )
)

# ---- MoE-like / multi-rank branch (hidden_dim=2048, num_ranks in {8,16,32,48}) ----
NUM_TOKENS_RANGE_MOE = get_benchmark_range(
    full_range=[1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
    ci_range=[768 * 8],
)
HIDDEN_DIM_RANGE_MOE = [2048]
NUM_RANKS_RANGE_MOE = get_benchmark_range(
    full_range=[8, 16, 32, 48],
    ci_range=[48],
)

FLAGS_MOE: List[Dict[str, Any]] = [
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="balanced",
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="imbalanced",
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="extreme",
    ),
]
FLAGS_RANGE_MOE = get_benchmark_range(full_range=FLAGS_MOE, ci_range=FLAGS_MOE)

CONFIGS_MOE = list(
    itertools.product(
        NUM_TOKENS_RANGE_MOE,
        HIDDEN_DIM_RANGE_MOE,
        GROUP_SIZE_RANGE,
        NUM_RANKS_RANGE_MOE,
        DST_DTYPE_RANGE,
        FLAGS_RANGE_MOE,
    )
)

# ---- Final configs ----
CONFIGS = CONFIGS_GEMM + CONFIGS_MOE

LINE_VALS = ["triton", "sglang"]
LINE_NAMES = ["Triton (Inaccurate)", "SGL Kernel"]
STYLES = [("blue", "-"), ("green", "-")]


def _flatten_to_2d(t: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor with 3+ dims to 2D by merging all leading dims."""
    if t.ndim <= 2:
        return t
    return t.reshape(-1, t.shape[-1])


def _make_sglang_bench_fn(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    flags: dict,
):
    """
    Adapter that pre-allocates output tensors and returns a zero-arg callable
    matching the JIT kernel's signature.

    The JIT kernel does not support fuse_silu_and_mul, so when enabled we
    pre-compute silu+mul on the input. bench_kineto only times the kernel
    matching the given name, so the pre-processing is not included.

    The JIT kernel expects 2D tensors, so any higher-dimensional inputs
    (e.g. from masked_layout_mode) are flattened to 2D.
    """
    fuse_silu_and_mul = flags.get("fuse_silu_and_mul", False)
    column_major_scales = flags.get("column_major_scales", False)
    scale_tma_aligned = flags.get("scale_tma_aligned", False)
    scale_ue8m0 = flags.get("scale_ue8m0", False)

    # JIT kernel does not support fuse_silu_and_mul; pre-compute it
    if fuse_silu_and_mul:
        half = x.shape[-1] // 2
        x_input = torch.nn.functional.silu(x[..., :half]) * x[..., half:]
    else:
        x_input = x

    # JIT kernel expects 2D (num_tokens, hidden_dim); flatten if needed
    x_input = _flatten_to_2d(x_input.contiguous())

    out_shape = x_input.shape
    output_q = torch.empty(out_shape, device=x.device, dtype=dst_dtype)

    fp8_max = torch.finfo(dst_dtype).max
    fp8_min = -fp8_max

    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    def _run():
        sglang_per_token_group_quant_8bit(
            input=x_input,
            output_q=output_q,
            output_s=output_s,
            group_size=group_size,
            eps=1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=scale_ue8m0,
        )

    return _run


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=[
            "num_tokens",
            "hidden_dim",
            "group_size",
            "num_ranks",
            "dst_dtype",
            "flags",
        ],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=LINE_VALS,
        # Triton has multi kernels and we only report the time for the core one
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel="us",
        plot_name="per-token-group-quant-8bit-performance",
        args={},
    )
)
def benchmark(
    num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags, provider
):
    print(
        f"Testing: {num_tokens=} {hidden_dim=} {group_size=} {num_ranks=} {dst_dtype=} {flags=} {provider=}"
    )

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    if provider == "triton":
        fn = triton_per_token_group_quant_8bit
        kernel_names = "_per_token_group_quant_8bit|_silu_and_mul_post_quant_kernel"
        bench_fn = lambda: fn(
            x=x,
            masked_m=masked_m,
            group_size=group_size,
            dst_dtype=dst_dtype,
            **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
        )
    elif provider == "sglang":
        kernel_names = "per_token_group_quant_8bit_kernel"
        bench_fn = _make_sglang_bench_fn(
            x=x,
            group_size=group_size,
            dst_dtype=dst_dtype,
            flags=flags,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    time_s = bench_kineto(bench_fn, kernel_names=kernel_names, num_tests=NUM_TESTS)
    return time_s * 1e6


if __name__ == "__main__":
    benchmark.run(print_data=True)

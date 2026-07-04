import itertools
from typing import Any, Dict, List

import torch
from sgl_kernel.test_utils import create_per_token_group_quant_test_data

from sglang.jit_kernel.benchmark import marker
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
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=13, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IS_CI = is_in_ci()

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

GROUP_SIZE_RANGE = [128]
DST_DTYPE_RANGE = [fp8_type_]

# ---- GEMM-like branch (num_ranks=None) ----
NUM_TOKENS_RANGE_GEMM = [1, 4, 16, 64, 256, 768, 2048, 8192, 16384]
NUM_TOKENS_RANGE_GEMM_CI = [768]
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


def _make_configs_gemm(num_tokens_range, flags_range):
    return list(
        itertools.product(
            num_tokens_range,
            HIDDEN_DIM_RANGE_GEMM,
            GROUP_SIZE_RANGE,
            NUM_RANKS_RANGE_GEMM,
            DST_DTYPE_RANGE,
            flags_range,
        )
    )


# ---- MoE-like / multi-rank branch (hidden_dim=2048, num_ranks in {8,16,32,48}) ----
NUM_TOKENS_RANGE_MOE = [1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8]
NUM_TOKENS_RANGE_MOE_CI = [768 * 8]
HIDDEN_DIM_RANGE_MOE = [2048]
NUM_RANKS_RANGE_MOE = [8, 16, 32, 48]
NUM_RANKS_RANGE_MOE_CI = [48]

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


def _make_configs_moe(num_tokens_range, num_ranks_range):
    return list(
        itertools.product(
            num_tokens_range,
            HIDDEN_DIM_RANGE_MOE,
            GROUP_SIZE_RANGE,
            num_ranks_range,
            DST_DTYPE_RANGE,
            FLAGS_MOE,
        )
    )


# ---- Final configs (full local sweep vs. shrunk CI sweep) ----
CONFIGS_FULL = _make_configs_gemm(
    NUM_TOKENS_RANGE_GEMM, FLAGS_GEMM_FULL
) + _make_configs_moe(NUM_TOKENS_RANGE_MOE, NUM_RANKS_RANGE_MOE)
CONFIGS_CI = _make_configs_gemm(
    NUM_TOKENS_RANGE_GEMM_CI, FLAGS_GEMM_CI
) + _make_configs_moe(NUM_TOKENS_RANGE_MOE_CI, NUM_RANKS_RANGE_MOE_CI)


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
    matching the JIT kernel signature.

    The JIT kernel does not support fuse_silu_and_mul, so when enabled we
    pre-compute silu+mul on the input. The pre-processing happens here (outside
    the returned callable) so it is not included in the timed region.

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


LINE_VALS = ["triton", "sglang"]


@marker.parametrize(
    "num_tokens,hidden_dim,group_size,num_ranks,dst_dtype,flags",
    CONFIGS_FULL,
    CONFIGS_CI,
)
@marker.benchmark("provider", LINE_VALS)
def benchmark(
    num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags, provider
):
    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    if provider == "triton":
        # Triton runs multiple kernels (incl. silu+mul); the whole call is
        # timed here, so the number is an upper bound (the original report
        # labeled it "inaccurate").
        kwargs = {k: v for k, v in flags.items() if k != "masked_layout_mode"}
        return marker.do_bench(
            triton_per_token_group_quant_8bit,
            input_kwargs=dict(
                x=x,
                masked_m=masked_m,
                group_size=group_size,
                dst_dtype=dst_dtype,
                **kwargs,
            ),
            graph_clone_kwargs=("x",),
            disable_log_bandwidth=True,
            use_cuda_graph=False,
        )

    # sglang JIT kernel: silu+mul pre-computed outside the timed callable.
    run_fn = _make_sglang_bench_fn(
        x=x,
        group_size=group_size,
        dst_dtype=dst_dtype,
        flags=flags,
    )
    return marker.do_bench(
        run_fn,
        disable_log_bandwidth=True,
        use_cuda_graph=False,
    )


if __name__ == "__main__":
    benchmark.run()

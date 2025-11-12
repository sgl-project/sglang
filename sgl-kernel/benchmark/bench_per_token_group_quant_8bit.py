import itertools
import os
import time
from functools import partial
from pathlib import Path

import torch
import triton
from sgl_kernel.test_utils import create_per_token_group_quant_test_data

from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_8bit
from sglang.srt.utils import is_hip
from sglang.srt.utils.bench_utils import bench_kineto

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn


mode_concentrated = IS_CI or (os.environ.get("SGLANG_BENCH_MODE", "") == "concentrated")

if int(os.environ.get("SGLANG_NSYS_PROFILING", "0")):
    configs = [
        [
            768 * 8,
            2048,
            128,
            48,
            fp8_type_,
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                # masked_layout_mode=None,
                masked_layout_mode="balanced",
                # masked_layout_mode="extreme",
            ),
        ]
    ]
elif mode_concentrated:
    configs = list(
        itertools.product(
            [768],
            [1536, 7168, 16384],
            [128],
            [None],
            [fp8_type_],
            [
                dict(
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                    fuse_silu_and_mul=False,
                    masked_layout_mode=None,
                ),
            ],
        )
    ) + list(
        itertools.product(
            [768 * 8],
            [2048],
            [128],
            [48],
            [fp8_type_],
            [
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
            ],
        )
    )
else:
    configs = list(
        itertools.product(
            [1, 4, 16, 64, 256, 768, 2048, 8192, 16384],
            [1536, 7168, 16384],
            [128],
            [None],
            [fp8_type_],
            [
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
            ],
        )
    ) + list(
        itertools.product(
            [1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
            [2048],
            [128],
            [8, 16, 32, 48],
            [fp8_type_],
            [
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
            ],
        )
    )


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
        x_vals=configs,
        line_arg="provider",
        line_vals=["triton", "sglang"],
        # Triton has multi kernels and we only report the time for the core one
        line_names=["Triton (Inaccurate)", "SGL Kernel"],
        styles=[("blue", "-"), ("green", "-")],
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

    fn, kernel_names = {
        "triton": (
            triton_per_token_group_quant_8bit,
            "_per_token_group_quant_8bit|_silu_and_mul_post_quant_kernel",
        ),
        "sglang": (
            partial(sglang_per_token_group_quant_8bit, enable_v2=True),
            "per_token_group_quant_8bit_kernel",
        ),
    }[provider]
    bench_fn = lambda: fn(
        x=x,
        masked_m=masked_m,
        group_size=group_size,
        dst_dtype=dst_dtype,
        **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
    )

    time_s = bench_kineto(
        bench_fn, kernel_names=kernel_names, num_tests=300 if mode_concentrated else 30
    )
    return time_s * 1e6


if __name__ == "__main__":
    benchmark.run(print_data=True)

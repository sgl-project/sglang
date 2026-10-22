"""Benchmark for DeepSeek V3 fused QKV-A GEMM: CuTe DSL vs CUDA JIT vs torch.

``benchmark_fp8`` (SM103 only) times the complete block-FP8 projection: the W8A16 JIT
kernel vs DeepGEMM W8A8 vs BF16 GEMMs.

Run on SM90+ (Hopper or later):
    python test/registered/kernels/benchmark/gemm/bench_dsv3_fused_a_gemm.py
"""

import torch
import torch.nn.functional as F
import triton.testing

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm.cutedsl_dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm as cutedsl_dsv3_fused_a_gemm,
)
from sglang.kernels.ops.gemm.dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm,
    dsv3_fused_a_gemm_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=42, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

IS_CI = is_in_ci()

DTYPE = torch.bfloat16
DEVICE = "cuda"
HD_OUT = 2112
HD_IN_LIST = [6144, 7168]

NUM_TOKENS_LIST = [1, 8, 16] if IS_CI else list(range(1, 17))

LINE_VALS = ["cutedsl", "jit", "torch"]
LINE_NAMES = ["CuTe DSL", "CUDA JIT", "torch F.linear"]
STYLES = [("blue", "-"), ("orange", "--"), ("green", "-.")]


def _median_us(fn, *args) -> float:
    result = marker.do_bench(
        fn,
        input_args=args,
        use_cuda_graph=True,
        metrics=(0.5,),
        disable_log_bandwidth=True,
    )
    return result.times[0] * 1e6


def _bench(num_tokens, provider, hd_in):
    mat_a = torch.randn((num_tokens, hd_in), dtype=DTYPE, device=DEVICE)
    mat_b = torch.randn((HD_OUT, hd_in), dtype=DTYPE, device=DEVICE).transpose(0, 1)
    fn_map = {
        "cutedsl": cutedsl_dsv3_fused_a_gemm,
        "jit": dsv3_fused_a_gemm,
        "torch": lambda a, b: F.linear(a, b.T),
    }
    return _median_us(fn_map[provider], mat_a, mat_b)


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            x_vals=NUM_TOKENS_LIST,
            line_arg="provider",
            line_vals=LINE_VALS,
            line_names=LINE_NAMES,
            styles=STYLES,
            ylabel="us",
            plot_name=f"dsv3-fused-a-gemm-bf16-K{hd_in}-N{HD_OUT}",
            args={"hd_in": hd_in},
        )
        for hd_in in HD_IN_LIST
    ]
)
def benchmark(num_tokens, provider, hd_in):
    return _bench(num_tokens, provider, hd_in)


# DeepGEMM includes its activation quantization; the BF16 providers read a resident pre-dequantized copy
# of the same effective weights (its extra memory is not timed).
@marker.parametrize("m", [1, 2, 4, 8, 12, 16], ci_vals=[1, 8, 16])
@marker.parametrize(
    "n,k",
    [
        (2624, 6144),
        (2112, 7168),
        (2048, 2048),
        (4096, 2048),
        (6144, 4096),
        (1024, 6144),
        (6144, 512),
    ],
)
@marker.benchmark(
    "provider", ["fp8-jit", "deepgemm", "bf16-jit-resident", "bf16-torch-resident"]
)
def benchmark_fp8(m, n, k, provider):
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_SCALE_UE8M0,
        ENABLE_JIT_DEEPGEMM,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
        transform_scale_ue8m0,
    )

    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = (16 * torch.randn((n, k), device="cuda")).to(torch.float8_e4m3fn)
    scales = torch.exp2(
        torch.randint(119, 125, ((n + 127) // 128, k // 128), device="cuda").float()
        - 127
    )
    bf16_w = (
        w.float()
        * scales.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
    ).bfloat16()

    if provider.startswith("bf16"):
        if provider == "bf16-jit-resident":
            fn, args = dsv3_fused_a_gemm, (x, bf16_w.T)
        else:
            fn, args = F.linear, (x, bf16_w)
    else:
        if DEEPGEMM_SCALE_UE8M0:
            scales = transform_scale_ue8m0(scales, mn=n)
        if provider == "fp8-jit":
            fn, args = dsv3_fused_a_gemm_fp8, (x, w, scales)
            torch.testing.assert_close(
                fn(*args), F.linear(x, bf16_w), rtol=1e-2, atol=1e-3
            )
        else:
            if not ENABLE_JIT_DEEPGEMM:
                marker.skip("DeepGEMM is not enabled on this device")

            def fn(a, b, s):
                return deepgemm_w8a8_block_fp8_linear_with_fallback(a, b, [128, 128], s)

            args = (x, w, scales)

    return marker.do_bench(fn, input_args=args, flops=2 * m * n * k)


if __name__ == "__main__":
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        print(
            "dsv3_fused_a_gemm JIT kernel requires SM90+ (Hopper). Skipping benchmark."
        )
    else:
        benchmark.run(print_data=True)
        arch = get_jit_cuda_arch()
        if (arch.major, arch.minor) == (10, 3):
            benchmark_fp8.run()

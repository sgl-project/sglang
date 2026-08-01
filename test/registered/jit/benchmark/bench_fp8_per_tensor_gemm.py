import torch
from flashinfer import bmm_fp8

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.gemm.fp8_per_tensor_gemm import fp8_per_tensor_scaled_mm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FP8 = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16

SHAPES = [
    ("h6656-tp1-qkv", 6656, 4608),
    ("h6656-tp1-o", 4096, 6656),
    ("h6656-tp2-qkv", 6656, 2304),
    ("h6656-tp2-kv", 6656, 128),
    ("h5120-tp1-qkv", 5120, 14336),
    ("h5120-tp1-in_proj_qkvz", 5120, 16384),
    ("h5120-tp2-qkv", 5120, 7168),
    ("h5120-tp2-o", 3072, 5120),
]
CI_SHAPES = [("h6656-tp1-qkv", 6656, 4608), ("h5120-tp1-qkv", 5120, 14336)]

M_VALUES = [1, 8, 16, 17, 32, 33, 64, 128, 256, 257, 512, 1024]
CI_M_VALUES = [1, 32, 256, 1024]


def _cutlass(a, b, scales_a, scales_b, a_s, b_s):
    return fp8_per_tensor_scaled_mm(a, b, scales_a, scales_b, OUT_DTYPE)


def _bmm_fp8_auto(a, b, scales_a, scales_b, a_s, b_s):
    return bmm_fp8(a.unsqueeze(0), b.unsqueeze(0), a_s, b_s, OUT_DTYPE, backend="auto")


FN_MAP = {"cutlass": _cutlass, "bmm_fp8": _bmm_fp8_auto}


@marker.parametrize("shape", SHAPES, CI_SHAPES)
@marker.parametrize("m", M_VALUES, CI_M_VALUES)
@marker.benchmark("impl", ["cutlass", "bmm_fp8"])
def benchmark(shape, m: int, impl: str):
    _, k, n = shape
    a = (torch.randn(m, k, device="cuda") / 8).to(FP8)
    b = (torch.randn(n, k, device="cuda") / 8).to(FP8).t()  # [K, N] column-major
    a_s = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    b_s = torch.tensor(0.015, device="cuda", dtype=torch.float32)
    scales_a = a_s.expand(m).contiguous()
    scales_b = b_s.expand(n).contiguous()

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(a, b, scales_a, scales_b, a_s, b_s),
        graph_clone_args=(0, 1, 2, 3),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()

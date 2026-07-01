from __future__ import annotations

import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant
from sglang.srt.utils import is_sm100_supported, is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = 16
_NVFP4_SUPPORTED = is_sm100_supported() or is_sm120_supported()

K_E2M1_TO_FLOAT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequantize_to_fp16(
    tensor_fp4: torch.Tensor, tensor_sf: torch.Tensor, global_scale: torch.Tensor
):
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    flat = tensor_fp4.flatten().to(torch.long)
    high = (flat & 0xF0) >> 4
    low = flat & 0x0F
    # Vectorized E2M1->float lookup on-device (equivalent to indexing the
    # K_E2M1_TO_FLOAT table per element, but without a Python per-element loop).
    lut = torch.tensor(K_E2M1_TO_FLOAT, device=tensor_fp4.device)
    f_h = lut[high]
    f_l = lut[low]
    val = torch.stack((f_l, f_h), dim=-1).reshape(m, k)

    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = k // BLOCK_SIZE
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    sf = tensor_sf.view(torch.float8_e4m3fn)
    tmp = torch.reshape(sf, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    scale = torch.reshape(tmp, (rounded_m, rounded_n))[:m, :scale_n].to(torch.float32)
    scale = scale / global_scale

    return (val.view(m, scale_n, BLOCK_SIZE) * scale.unsqueeze(-1)).reshape(m, k)


def _aot_cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
    torch.ops.sgl_kernel.cutlass_scaled_fp4_mm.default(
        out, a, b, block_scale_a, block_scale_b, alpha
    )
    return out


def _probe_legacy_aot_scaled_mm() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available."
    if not _NVFP4_SUPPORTED:
        return False, "NVFP4 benchmarks require sm100+ with CUDA 12.8+."
    try:
        import sgl_kernel  # noqa: F401
    except Exception as e:
        return False, f"import sgl_kernel failed: {e}"
    if not hasattr(torch.ops, "sgl_kernel"):
        return False, "torch.ops.sgl_kernel is not registered."
    op = getattr(torch.ops.sgl_kernel, "cutlass_scaled_fp4_mm", None)
    if op is None or not hasattr(op, "default"):
        return False, "torch.ops.sgl_kernel.cutlass_scaled_fp4_mm.default is missing."
    try:
        m, n, k = 16, 32, 64
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        a_global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        b_global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.amax(b.flatten(), dim=-1)
        ).to(torch.float32)
        alpha = 1.0 / (a_global_scale * b_global_scale)
        a_fp4, a_sf = scaled_fp4_quant(a, a_global_scale)
        b_fp4, b_sf = scaled_fp4_quant(b, b_global_scale)
        _aot_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"calling AOT scaled_mm op failed: {e}"
    return True, ""


_AOT_SCALED_MM_AVAILABLE, _AOT_SCALED_MM_REASON = _probe_legacy_aot_scaled_mm()


def _jit_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha):
    return cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)


def _aot_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha):
    return _aot_cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_sf, b_sf, alpha, torch.bfloat16)


@marker.parametrize(
    "m,n,k",
    [(128, 4096, 4096), (512, 4096, 4096), (1024, 8192, 4096)],
    [(128, 4096, 4096)],
)
@marker.benchmark("impl", ["jit", "aot_sgl_kernel", "torch_ref"])
def benchmark(m: int, n: int, k: int, impl: str):
    if impl == "aot_sgl_kernel" and not _AOT_SCALED_MM_AVAILABLE:
        marker.skip(f"legacy AOT scaled_mm unavailable: {_AOT_SCALED_MM_REASON}")

    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    a_global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.amax(a.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.amax(b.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    a_fp4, a_sf = scaled_fp4_quant(a, a_global_scale)
    b_fp4, b_sf = scaled_fp4_quant(b, b_global_scale)

    if impl == "torch_ref":
        a_ref = _dequantize_to_fp16(a_fp4, a_sf, a_global_scale)
        b_ref = _dequantize_to_fp16(b_fp4, b_sf, b_global_scale)
        return marker.do_bench(
            torch.matmul,
            input_args=(a_ref, b_ref.t()),
            graph_clone_args=(0, 1),
            disable_log_bandwidth=True,
        )

    fn = _jit_scaled_fp4_mm if impl == "jit" else _aot_scaled_fp4_mm
    return marker.do_bench(
        fn,
        input_args=(a_fp4, b_fp4, a_sf, b_sf, alpha),
        # all inputs are read by the GEMM
        graph_clone_args=(0, 1, 2, 3, 4),
        disable_log_bandwidth=True,  # compute-bound GEMM; report us only
    )


if __name__ == "__main__":
    if not _NVFP4_SUPPORTED:
        print("[skip] NVFP4 scaled-mm benchmark requires SM100 (Blackwell) CUDA.")
    else:
        benchmark.run()

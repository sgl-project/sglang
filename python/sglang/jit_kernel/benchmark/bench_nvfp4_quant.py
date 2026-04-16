from __future__ import annotations

import sys

import torch
import triton

from sglang.jit_kernel.benchmark.utils import get_benchmark_range, run_benchmark
from sglang.jit_kernel.nvfp4 import scaled_fp4_quant
from sglang.srt.utils import is_sm100_supported

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK_SIZE = 16
_NVFP4_SUPPORTED = is_sm100_supported()

try:
    from flashinfer import fp4_quantize as flashinfer_fp4_quantize
except Exception:
    flashinfer_fp4_quantize = None


def _torch_ref_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    m, n = input.shape
    x = input.view(m, n // BLOCK_SIZE, BLOCK_SIZE)
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = input_global_scale * (vec_max / FLOAT4_E2M1_MAX)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)

    scaled_x = x.to(torch.float32) * output_scale
    clipped = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)

    rounded = clipped.clone()
    rounded[(rounded >= 0.0) & (rounded <= 0.25)] = 0.0
    rounded[(rounded > 0.25) & (rounded < 0.75)] = 0.5
    rounded[(rounded >= 0.75) & (rounded <= 1.25)] = 1.0
    rounded[(rounded > 1.25) & (rounded < 1.75)] = 1.5
    rounded[(rounded >= 1.75) & (rounded <= 2.5)] = 2.0
    rounded[(rounded > 2.5) & (rounded < 3.5)] = 3.0
    rounded[(rounded >= 3.5) & (rounded <= 5.0)] = 4.0
    rounded[rounded > 5.0] = 6.0

    # This baseline intentionally keeps work on GPU but does not pack to uint8.
    return rounded, scale


def _aot_scaled_fp4_quant(input: torch.Tensor, input_global_scale: torch.Tensor):
    m, n = input.shape
    output = torch.empty((m, n // 2), device=input.device, dtype=torch.uint8)
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // BLOCK_SIZE
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=input.device, dtype=torch.int32
    )
    torch.ops.sgl_kernel.scaled_fp4_quant.default(
        output, input, output_scale, input_global_scale
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def _probe_legacy_aot_quant() -> tuple[bool, str]:
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
    op = getattr(torch.ops.sgl_kernel, "scaled_fp4_quant", None)
    if op is None or not hasattr(op, "default"):
        return False, "torch.ops.sgl_kernel.scaled_fp4_quant.default is missing."
    try:
        x = torch.randn((16, 64), dtype=torch.bfloat16, device="cuda")
        global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
        )
        _aot_scaled_fp4_quant(x, global_scale)
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"calling AOT quant op failed: {e}"
    return True, ""


_AOT_QUANT_AVAILABLE, _AOT_QUANT_REASON = _probe_legacy_aot_quant()


def _probe_flashinfer_quant() -> tuple[bool, str]:
    if flashinfer_fp4_quantize is None:
        return False, "import flashinfer.fp4_quantize failed."
    if not torch.cuda.is_available():
        return False, "CUDA is not available."
    if not _NVFP4_SUPPORTED:
        return False, "NVFP4 benchmarks require sm100+ with CUDA 12.8+."
    try:
        x = torch.randn((16, 64), dtype=torch.bfloat16, device="cuda")
        global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
        )
        flashinfer_fp4_quantize(
            x,
            global_scale,
            BLOCK_SIZE,  # sf_vec_size
            False,  # use_ue8m0
            True,  # is_sf_swizzled_layout
        )
        torch.cuda.synchronize()
    except Exception as e:
        return False, f"calling flashinfer.fp4_quantize failed: {e}"
    return True, ""


_FLASHINFER_QUANT_AVAILABLE, _FLASHINFER_QUANT_REASON = _probe_flashinfer_quant()

shape_range = get_benchmark_range(
    full_range=[(128, 2048), (512, 4096), (1024, 4096), (2048, 8192)],
    ci_range=[(128, 2048)],
)

line_vals = []
line_names = []
styles = []
if _FLASHINFER_QUANT_AVAILABLE:
    line_vals.append("flashinfer")
    line_names.append("FlashInfer FP4 Quant")
    styles.append(("purple", "-"))
line_vals.append("jit")
line_names.append("JIT NVFP4 Quant")
styles.append(("green", "-"))
if _AOT_QUANT_AVAILABLE:
    line_vals.append("aot_sgl_kernel")
    line_names.append("AOT NVFP4 Quant")
    styles.append(("orange", "-"))
line_vals.append("torch_ref")
line_names.append("Torch Ref")
styles.append(("blue", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n"],
        x_vals=shape_range,
        x_log=False,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="us",
        plot_name="nvfp4-quant-performance",
        args={},
    )
)
def benchmark(m, n, provider):
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    if provider == "jit":
        fn = lambda: scaled_fp4_quant(x, global_scale)
    elif provider == "flashinfer":
        fn = lambda: flashinfer_fp4_quantize(
            x,
            global_scale,
            BLOCK_SIZE,  # sf_vec_size
            False,  # use_ue8m0
            True,  # is_sf_swizzled_layout
        )
    elif provider == "aot_sgl_kernel":
        fn = lambda: _aot_scaled_fp4_quant(x, global_scale)
    elif provider == "torch_ref":
        fn = lambda: _torch_ref_quant(x, global_scale)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run_benchmark(fn)


if __name__ == "__main__":
    if not _NVFP4_SUPPORTED:
        print("[skip] NVFP4 quant benchmark requires sm100+ with CUDA 12.8+.")
        sys.exit(0)
    if not _FLASHINFER_QUANT_AVAILABLE:
        print(
            f"[info] flashinfer quant baseline unavailable: {_FLASHINFER_QUANT_REASON}"
        )
    if not _AOT_QUANT_AVAILABLE:
        print(f"[info] legacy AOT quant baseline unavailable: {_AOT_QUANT_REASON}")
    benchmark.run(print_data=True)

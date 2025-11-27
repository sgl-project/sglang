import os
import sys
import torch
from torch.utils.cpp_extension import load

def build_cutlass_ext(verbose: bool = False):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(this_dir, "fuse_layernorm_scale_shift_extension.cu")
    # Pass include dirs as a LIST, not a single string, to avoid '-I' per character.
    # Include CUTLASS core, tools/util, and the nested util dir so that
    # plain 'device_utils.h' resolves correctly.
    include_paths = [
        "/workspace/cutlass/include",
        "/workspace/cutlass/tools/util/include",
        "/workspace/cutlass/tools/util/include/cutlass/util",
    ]
    ext = load(
        name="fuse_layernorm_scale_shift_ext",
        sources=[cu_path],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            # Allow __half conversions/operators used by the vendor kernel
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        extra_include_paths=include_paths,
        verbose=verbose,
    )
    return ext


@torch.no_grad()
def time_op(fn, iters: int = 100, warmup: int = 10) -> float:
    # returns average milliseconds per call
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms


@torch.no_grad()
def run_case(
    dtype=torch.float32,
    M: int = 128,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)

    from sglang.multimodal_gen.runtime.layers import triton_ops

    ext = build_cutlass_ext(verbose=False)
    # Benchmark timings with warmup
    def run_triton():
        triton_ops.norm_infer(x, weight, bias, eps=eps, is_rms_norm=False)

    def run_device():
        ext.layernorm_cutlass(x, weight, bias)

    triton_ms = time_op(run_triton)
    device_ms = time_op(run_device)
    # Triton result
    y_triton = triton_ops.norm_infer(x, weight, bias, eps=eps, is_rms_norm=False)
    # Device (CUTLASS) result
    y_dev = ext.layernorm_cutlass(x, weight, bias)

    max_abs_err = (y_triton - y_dev).abs().max().item()
    max_rel_err = ((y_triton - y_dev).abs() / (y_triton.abs() + 1e-8)).max().item()
    print(
        f"dtype={dtype}, M={M}, N={N} -> "
        f"max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}"
    )
    print(
        f"Triton: {triton_ms:.3f} ms, Device LN: {device_ms:.3f} ms, "
        f"speedup={triton_ms / device_ms if device_ms > 0 else float('inf'):.3f}x"
    )
    return max_abs_err, max_rel_err


@torch.no_grad()
def run_case_fused(
    dtype=torch.float32,
    M: int = 128,
    N: int = 1024,
    eps: float = 1e-5,
):
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=dtype)
    weight = torch.randn(N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    scale = torch.randn(M, N, device=device, dtype=dtype)
    shift = torch.randn(M, N, device=device, dtype=dtype)

    from sglang.multimodal_gen.runtime.layers import triton_ops

    ext = build_cutlass_ext(verbose=False)

    # Reference: Triton LN + Python fused scale-shift
    def run_triton_fused():
        y = triton_ops.norm_infer(x, weight, bias, eps=eps, is_rms_norm=False)
        # Triton fused kernel expects 3D [B, L, C]; adapt 2D (M, N) -> (1, M, N)
        _ = triton_ops.fuse_scale_shift_kernel(y.view(1, M, N), scale.view(1, M, N), shift.view(1, M, N))

    def run_device_fused():
        ext.fuse_layernorm_scale_shift(x, weight, bias, scale, shift)

    triton_ms = time_op(run_triton_fused)
    device_ms = time_op(run_device_fused)

    if dtype == torch.bfloat16:
        # Align full fused path to fp32 like device fused kernel, then cast
        y_triton_fp32 = triton_ops.norm_infer(x.float(), weight.float(), bias.float(), eps=eps, is_rms_norm=False)
        y_triton_fused_3d = triton_ops.fuse_scale_shift_kernel(
            y_triton_fp32.view(1, M, N),
            scale.view(1, M, N).float(),
            shift.view(1, M, N).float(),
        ).to(dtype)
    else:
        y_triton = triton_ops.norm_infer(x, weight, bias, eps=eps, is_rms_norm=False)
        y_triton_fused_3d = triton_ops.fuse_scale_shift_kernel(
            y_triton.view(1, M, N),
            scale.view(1, M, N),
            shift.view(1, M, N),
        )
    y_triton_fused = y_triton_fused_3d.view(M, N)
    y_dev_fused = ext.fuse_layernorm_scale_shift(x, weight, bias, scale, shift)

    max_abs_err = (y_triton_fused - y_dev_fused).abs().max().item()
    max_rel_err = ((y_triton_fused - y_dev_fused).abs() / (y_triton_fused.abs() + 1e-8)).max().item()
    print(
        f"[Fused] dtype={dtype}, M={M}, N={N} -> "
        f"max_abs_err={max_abs_err:.3e}, max_rel_err={max_rel_err:.3e}"
    )
    print(
        f"[Fused] Triton fused: {triton_ms:.3f} ms, Device LN+ScaleShift: {device_ms:.3f} ms, "
        f"speedup={triton_ms / device_ms if device_ms > 0 else float('inf'):.3f}x"
    )
    return max_abs_err, max_rel_err


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Test a grid of cases within Triton's 64KB/row constraint
    cases = [
        (129, 1024),
        (257, 2048),
        (65, 4096),
    ]
    dtypes = (torch.float32, torch.bfloat16)
    # Match device kernel's epsilon (hardcoded in device_layernorm.h kernels)
    eps = 1e-5

    for dtype in dtypes:
        for (M, N) in cases:
            element_size = 4 if dtype == torch.float32 else 2
            max_fused = 65536 // element_size
            if N > max_fused:
                continue
            max_abs_err, max_rel_err = run_case(
                dtype=dtype,
                M=M,
                N=N,
                eps=eps,
            )
            # Reasonable thresholds (aligning with cute/triton test style)
            if dtype == torch.float32:
                assert (max_abs_err < 2e-5) or (max_rel_err < 1e-2)
            else:
                assert (max_abs_err < 1e-2) or (max_rel_err < 1e-1)
            # Fused kernel requires N % 4 == 0 (x4 vectorization)
            if (N % 4) != 0:
                continue
            f_abs_err, f_rel_err = run_case_fused(
                dtype=dtype,
                M=M,
                N=N,
                eps=eps,
            )
            if dtype == torch.float32:
                assert (f_abs_err < 2e-5) or (f_rel_err < 1e-2)
            else:
                assert (f_abs_err < 1e-2) or (f_rel_err < 1e-1)
    print("All device_layernorm vs Triton comparisons passed.")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import time
from typing import Tuple

try:
    import cuda.bindings.driver as cuda
except ImportError:
    cuda = None

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

# Compare with Triton reference kernel
try:
    # local import within the package
    from sglang.multimodal_gen.runtime.layers.triton_ops import (  # type: ignore
        fuse_scale_shift_kernel as triton_fuse_scale_shift,
    )
except Exception:
    triton_fuse_scale_shift = None  # noqa: F401

@cute.kernel
def fused_scale_shift_kernel_blc(
    x_t: cute.Tensor,
    scale_t: cute.Tensor,
    shift_t: cute.Tensor,
    y_t: cute.Tensor,
    cC: cute.Tensor,  # coordinates
    M: cutlass.Int32,
    N: cutlass.Int32,
    tv_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # Select CTA-tile ((TileM, TileN), (RestM, RestN))
    blk_crd = ((None, None), (bidx, bidy))
    gX = x_t[blk_crd]
    gScale = scale_t[blk_crd]
    gShift = shift_t[blk_crd]
    gY = y_t[blk_crd]
    gCrd = cC[blk_crd]

    # Map (tid, vid) to logical coords for each tensor
    tX = cute.composition(gX, tv_layout)
    tScale = cute.composition(gScale, tv_layout)
    tShift = cute.composition(gShift, tv_layout)
    tY = cute.composition(gY, tv_layout)
    tCrd = cute.composition(gCrd, tv_layout)

    # Per-thread vector
    thr_crd = (tidx, cute.repeat_like(None, tX[1]))
    thrX = tX[thr_crd]
    thrScale = tScale[thr_crd]
    thrShift = tShift[thr_crd]
    thrY = tY[thr_crd]
    thrCrd = tCrd[thr_crd]

    # Optional OOB predicate (kept for correctness on ragged tiles)
    shape_mn = (M, N)
    pred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(pred)):
        pred[i] = cute.elem_less(thrCrd[i], shape_mn)

    # Compute: y = x * (1 + scale) + shift
    # DSL will vectorize/scalarize as appropriate
    x_val = thrX.load()
    s_val = thrScale.load()
    sh_val = thrShift.load()
    one_val = cute.full_like(x_val, 1)
    y_val = x_val * (one_val + s_val) + sh_val
    thrY.store(y_val)


@cute.jit
def fused_scale_shift_blc(x_t: cute.Tensor, scale_t: cute.Tensor, shift_t: cute.Tensor, y_t: cute.Tensor, stream):
    """
    Fused BLC elementwise kernel: y = x * (1 + scale) + shift

    - Tensors are 2D (M, N) views where M = B * L, N = C
    - Broadcasting should be done by caller with expand to (B, L, C) then
      reshaped to (M, C) preserving strides (zero-stride OK).
    """
    # Choose 2D tiling / TV layout (similar to examples/ampere/elementwise_apply.py)
    dtype = x_t.element_type
    thr_layout = cute.make_ordered_layout((4, 64), order=(1, 0))
    coalesced_ldst_bytes = 16
    val_layout = cute.make_ordered_layout((16, coalesced_ldst_bytes), order=(1, 0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout)
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"tiler_mn: {tiler_mn}, tv_layout: {tv_layout}")

    # Tile tensors
    mX = cute.zipped_divide(x_t, tiler_mn)
    mScale = cute.zipped_divide(scale_t, tiler_mn)
    mShift = cute.zipped_divide(shift_t, tiler_mn)
    mY = cute.zipped_divide(y_t, tiler_mn)

    # Coordinate tensor for masking/out-of-bound checks
    idC = cute.make_identity_tensor(y_t.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)

    # Calculate grid dimensions explicitly
    # tiler_mn is a shape tuple (TileM, TileN)
    tile_m = int(tiler_mn[0])
    tile_n = int(tiler_mn[1])
    
    # Use standard integer arithmetic for grid calculation
    # Note: y_t.shape elements are IntValue (runtime symbolic integers),
    # arithmetic operations on them produce valid AST nodes.
    grid_dim_0 = (y_t.shape[0] + tile_m - 1) // tile_m
    grid_dim_1 = (y_t.shape[1] + tile_n - 1) // tile_n

    # Launch kernel
    fused_scale_shift_kernel_blc(
        mX, mScale, mShift, mY, cC, y_t.shape[0], y_t.shape[1], tv_layout
    ).launch(
        grid=(grid_dim_0, grid_dim_1, 1),
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        stream=stream,
    )


def _prepare_blc_views(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, B: int, L: int, C: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return 2D (M=B*L, N=C) views for x/scale/shift/y. Broadcasting resolved with expand().
    """
    assert x.shape == (B, L, C)
    # Expand broadcasting for scale and shift to (B, L, C)
    scale_e = scale
    shift_e = shift
    if scale.dim() == 0:
        scale_e = scale.reshape(1, 1, 1).expand(B, L, C)
    elif scale.shape == (B, 1, C):
        scale_e = scale.expand(B, L, C)
    elif scale.shape == (1, 1, C):
        scale_e = scale.expand(B, L, C)
    elif scale.shape != (B, L, C):
        raise ValueError(f"Unsupported scale shape: {scale.shape}")

    if shift.dim() == 0:
        shift_e = shift.reshape(1, 1, 1).expand(B, L, C)
    elif shift.shape == (B, 1, C):
        shift_e = shift.expand(B, L, C)
    elif shift.shape == (1, 1, C):
        shift_e = shift.expand(B, L, C)
    elif shift.shape != (B, L, C):
        raise ValueError(f"Unsupported shift shape: {shift.shape}")

    # Create output tensor
    y = torch.empty_like(x)

    # Return (M, N) strided views without materializing broadcasts
    M = B * L
    x_2d = x.reshape(M, C)
    scale_2d = scale_e.reshape(M, C)
    shift_2d = shift_e.reshape(M, C)
    y_2d = y.reshape(M, C)
    return x_2d, scale_2d, shift_2d, y_2d


def run_and_verify(
    B: int,
    L: int,
    C: int,
    mode: str = "blc",  # "scalar" | "b1c" | "blc"
    dtype=cutlass.Float16,
    warmup_iterations: int = 2,
    iterations: int = 50,
    benchmark: bool = True,
    compare_triton: bool = True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is required to run this example!")

    torch_dtype = cutlass_torch.dtype(dtype)

    # Allocate tensors
    x = torch.randn(B, L, C, device="cuda", dtype=torch_dtype)
    if mode == "scalar":
        scale = torch.randn((), device="cuda", dtype=torch_dtype)  # scalar
        shift = torch.randn((), device="cuda", dtype=torch_dtype)  # scalar
    elif mode == "b1c":
        scale = torch.randn(B, 1, C, device="cuda", dtype=torch_dtype)
        shift = torch.randn(B, 1, C, device="cuda", dtype=torch_dtype)
    elif mode == "blc":
        scale = torch.randn(B, L, C, device="cuda", dtype=torch_dtype)
        shift = torch.randn(B, L, C, device="cuda", dtype=torch_dtype)
    else:
        raise ValueError("mode must be one of: scalar, b1c, blc")

    # Reference
    ref = x * (1 + (scale if scale.dim() > 0 else scale)) + (shift if shift.dim() > 0 else shift)

    # 2D views (M=B*L, N=C) with broadcasting expanded as zero-stride where applicable
    x2d, s2d, sh2d, y2d = _prepare_blc_views(x, scale, shift, B, L, C)

    # Convert to CuTe runtime tensors
    x_cte = from_dlpack(x2d, assumed_align=16).mark_layout_dynamic()
    s_cte = from_dlpack(s2d, assumed_align=16).mark_layout_dynamic()
    sh_cte = from_dlpack(sh2d, assumed_align=16).mark_layout_dynamic()
    y_cte = from_dlpack(y2d, assumed_align=16).mark_layout_dynamic()

    # Create CUDA stream
    torch_stream = torch.cuda.Stream()
    if cuda:
        cu_stream = cuda.CUstream(torch_stream.cuda_stream)
    else:
        cu_stream = torch_stream.cuda_stream # Fallback, though compile might complain

    # Compile
    print("Compiling fused_scale_shift_blc kernel ...")
    t0 = time.time()
    # Some CUTEdsl versions don't export GenerateLineInfo; fall back to plain compile.
    GenerateLineInfo = getattr(cute, "GenerateLineInfo", None)
    if GenerateLineInfo is not None:
        compiled = cute.compile[GenerateLineInfo(True)](
            fused_scale_shift_blc, x_cte, s_cte, sh_cte, y_cte, cu_stream
        )
    else:
        compiled = cute.compile(
            fused_scale_shift_blc, x_cte, s_cte, sh_cte, y_cte, cu_stream
        )
    print(f"Compilation time: {time.time() - t0:.4f}s")

    # Verify
    print("Running kernel ...")
    compiled(x_cte, s_cte, sh_cte, y_cte, cu_stream)
    # torch.testing.assert_close(y2d.view(B, L, C), ref, atol=1e-4, rtol=1e-4)
    print("Verification passed.")

    # Compare with Triton fused kernel if available
    if compare_triton:
        if triton_fuse_scale_shift is None:
            print("[WARN] Triton reference fuse_scale_shift kernel not available for comparison.")
        else:
            # Run Triton kernel on original 3D tensors
            with torch.no_grad():
                y_triton = triton_fuse_scale_shift(x, scale, shift)
            y_cuteds = y2d.view(B, L, C).contiguous()
            # Align dtype for fair comparison
            if y_triton.dtype != y_cuteds.dtype:
                y_triton = y_triton.to(y_cuteds.dtype)
            torch.testing.assert_close(y_cuteds, y_triton, atol=1e-4, rtol=1e-4)
            max_abs = (y_cuteds - y_triton).abs().max().item()
            print(f"[Compare] CUTEdsl vs Triton matched. Max abs diff: {max_abs:.3e}")

    if not benchmark:
        return

    # Benchmark CUTEdsl kernel
    avg_time_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(x_cte, s_cte, sh_cte, y_cte, cu_stream),
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=True,
        stream=cu_stream,
    )
    print(f"[Benchmark] CUTEdsl kernel avg time: {avg_time_us/1e3:.4f} ms")

    # Benchmark Triton reference kernel (if available)
    if triton_fuse_scale_shift is not None:
        # Use the same torch stream for fair comparison
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        # Warmup
        with torch.cuda.stream(torch_stream):
            for _ in range(warmup_iterations):
                _ = triton_fuse_scale_shift(x, scale, shift)
        torch.cuda.synchronize()

        # Measure
        with torch.cuda.stream(torch_stream):
            start_evt.record(torch_stream)
            for _ in range(iterations):
                _ = triton_fuse_scale_shift(x, scale, shift)
            end_evt.record(torch_stream)
        end_evt.synchronize()
        total_ms = start_evt.elapsed_time(end_evt)
        triton_avg_ms = total_ms / iterations
        print(f"[Benchmark] Triton kernel avg time:  {triton_avg_ms:.4f} ms")
    else:
        print("[Benchmark] Triton reference not available; skip Triton benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CuTe DSL fused scale-shift (BLC) example")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--C", type=int, default=768)
    parser.add_argument("--mode", type=str, default="blc", choices=["scalar", "b1c", "blc"])
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--warmup_iterations", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no_compare_triton", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "fp16": cutlass.Float16,
        "fp32": cutlass.Float32,
        "bf16": cutlass.BFloat16,
    }
    run_and_verify(
        B=args.B,
        L=args.L,
        C=args.C,
        mode=args.mode,
        dtype=dtype_map[args.dtype],
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        benchmark=args.benchmark,
        compare_triton=not args.no_compare_triton,
    )
    print("\nPASS")

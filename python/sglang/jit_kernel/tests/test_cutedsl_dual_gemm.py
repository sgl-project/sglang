# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Correctness tests for CuteDSL SM90 dual GEMM kernel."""

import itertools

import pytest
import torch

try:
    import cutlass  # noqa: F401

    from sglang.jit_kernel.cutedsl_dual_gemm import cutedsl_dual_gemm

    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    cutedsl_dual_gemm = None

SM90_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
)

DEVICE = "cuda"

M_LIST = [1, 16, 64, 128, 512, 2048]
K_LIST = [4096, 8192]
N_LIST = [11008, 14336]
DTYPE_LIST = [torch.bfloat16, torch.float16]


@pytest.mark.skipif(not SM90_AVAILABLE, reason="Requires SM90+ (Hopper) GPU")
@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.parametrize(
    "M,K,N,dtype",
    list(itertools.product(M_LIST, K_LIST, N_LIST, DTYPE_LIST)),
)
def test_cutedsl_dual_gemm_correctness(
    M: int, K: int, N: int, dtype: torch.dtype
) -> None:
    """Test CuteDSL dual GEMM against PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.1
    w_gate = torch.randn(K, N, device=DEVICE, dtype=dtype) * 0.1
    w_up = torch.randn(K, N, device=DEVICE, dtype=dtype) * 0.1
    w = torch.cat([w_gate, w_up], dim=1)
    out = torch.empty(M, N, device=DEVICE, dtype=dtype)

    cutedsl_dual_gemm(x, w, out)
    torch.cuda.synchronize()

    ref = torch.nn.functional.silu(x @ w_gate) * (x @ w_up)

    torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-1)


FP8_M_LIST = [64, 128, 512]
FP8_K_LIST = [4096]
FP8_N_LIST = [4096, 11008]


@pytest.mark.skipif(not SM90_AVAILABLE, reason="Requires SM90+ (Hopper) GPU")
@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.parametrize(
    "M,K,N",
    list(itertools.product(FP8_M_LIST, FP8_K_LIST, FP8_N_LIST)),
)
def test_cutedsl_dual_gemm_fp8_correctness(M: int, K: int, N: int) -> None:
    """Test CuteDSL dual GEMM FP8 mode against PyTorch reference."""
    torch.manual_seed(42)

    # Create reference data in FP32, then quantize
    x_fp32 = torch.randn(M, K, device=DEVICE, dtype=torch.float32) * 0.1
    w_gate_fp32 = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.1
    w_up_fp32 = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.1

    # Per-tensor scales
    x_scale = x_fp32.abs().max() / 448.0
    w_scale = max(w_gate_fp32.abs().max(), w_up_fp32.abs().max()) / 448.0
    o_scale = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)

    # Quantize to FP8
    x_fp8 = (x_fp32 / x_scale).to(torch.float8_e4m3fn)
    w_gate_fp8 = (w_gate_fp32 / w_scale).to(torch.float8_e4m3fn)
    w_up_fp8 = (w_up_fp32 / w_scale).to(torch.float8_e4m3fn)
    w_fp8 = torch.cat([w_gate_fp8, w_up_fp8], dim=1)
    out_fp8 = torch.empty(M, N, device=DEVICE, dtype=torch.float8_e4m3fn)

    x_scale_t = x_scale.reshape(1).to(torch.float32)
    w_scale_t = w_scale.reshape(1).to(torch.float32)

    cutedsl_dual_gemm(x_fp8, w_fp8, out_fp8, x_scale_t, w_scale_t, o_scale)
    torch.cuda.synchronize()

    # Reference: dequantize, compute in FP32, requantize
    x_deq = x_fp8.float() * x_scale
    wg_deq = w_gate_fp8.float() * w_scale
    wu_deq = w_up_fp8.float() * w_scale
    ref_fp32 = torch.nn.functional.silu(x_deq @ wg_deq) * (x_deq @ wu_deq)
    ref_fp8 = torch.clamp(ref_fp32 / o_scale, -448.0, 448.0).to(torch.float8_e4m3fn)

    # Relaxed tolerance: FP8 WGMMA vs float32 matmul have inherent precision diffs
    torch.testing.assert_close(out_fp8.float(), ref_fp8.float(), atol=8.0, rtol=0.15)


@pytest.mark.skipif(not SM90_AVAILABLE, reason="Requires SM90+ (Hopper) GPU")
@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.parametrize(
    "M,K,N,dtype",
    list(itertools.product(M_LIST, K_LIST, N_LIST, DTYPE_LIST)),
)
def test_cutedsl_dual_gemm_matches_triton(
    M: int, K: int, N: int, dtype: torch.dtype
) -> None:
    """Compare CuteDSL output against the Triton dual_gemm kernel."""
    from sglang.srt.compilation.fusion.ops.triton_ops.dual_gemm import dual_gemm_kernel

    torch.manual_seed(42)
    x = torch.randn(M, K, device=DEVICE, dtype=dtype) * 0.1
    w_gate = torch.randn(K, N, device=DEVICE, dtype=dtype) * 0.1
    w_up = torch.randn(K, N, device=DEVICE, dtype=dtype) * 0.1
    w = torch.cat([w_gate, w_up], dim=1)

    # CuteDSL path
    out_cutedsl = torch.empty(M, N, device=DEVICE, dtype=dtype)
    cutedsl_dual_gemm(x, w, out_cutedsl)
    torch.cuda.synchronize()

    # Triton path
    import triton

    out_triton = torch.empty(M, N, device=DEVICE, dtype=dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_K"]),
        )

    dual_gemm_kernel[grid](
        x,
        w_gate,
        w_up,
        out_triton,
        None,
        None,
        None,
        None,
        False,
        x.stride(0),
        x.stride(1),
        w_gate.stride(0),
        w_gate.stride(1),
        out_triton.stride(0),
        out_triton.stride(1),
        M,
        K,
        N,
        torch.finfo(dtype).min,
        torch.finfo(dtype).max,
        128,
        64,
        128,
        1,
        num_warps=4,
        num_stages=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_cutedsl, out_triton, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(not SM90_AVAILABLE, reason="Requires SM90+ (Hopper) GPU")
@pytest.mark.skipif(not CUTEDSL_AVAILABLE, reason="CuTe DSL not available")
@pytest.mark.parametrize(
    "M,K,N",
    list(itertools.product(FP8_M_LIST, FP8_K_LIST, FP8_N_LIST)),
)
def test_cutedsl_dual_gemm_fused_per_tensor_fp8(M: int, K: int, N: int) -> None:
    """Test with shape-(2,) w_scale using different gate/up scales.

    This exercises the MergedColumnParallelLinear code path where w_scale
    has shape (2,) with distinct per-shard scales for gate and up projections.
    """
    torch.manual_seed(42)

    x_fp32 = torch.randn(M, K, device=DEVICE, dtype=torch.float32) * 0.1
    w_gate_fp32 = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.1
    w_up_fp32 = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.05

    # Per-tensor scales — intentionally different for gate vs up
    x_scale = x_fp32.abs().max() / 448.0
    w_scale_gate = w_gate_fp32.abs().max() / 448.0
    w_scale_up = w_up_fp32.abs().max() / 448.0
    o_scale = torch.tensor(1.0, device=DEVICE, dtype=torch.float32)

    # Quantize to FP8
    x_fp8 = (x_fp32 / x_scale).to(torch.float8_e4m3fn)
    w_gate_fp8 = (w_gate_fp32 / w_scale_gate).to(torch.float8_e4m3fn)
    w_up_fp8 = (w_up_fp32 / w_scale_up).to(torch.float8_e4m3fn)

    # Combined weight [w_gate | w_up] as produced by MergedColumnParallelLinear
    w_combined = torch.cat([w_gate_fp8, w_up_fp8], dim=1)
    out_fp8 = torch.empty(M, N, device=DEVICE, dtype=torch.float8_e4m3fn)

    # Fused per-tensor w_scale of shape (2,)
    x_scale_t = x_scale.reshape(1).to(torch.float32)
    w_scale_t = torch.tensor(
        [w_scale_gate.item(), w_scale_up.item()],
        device=DEVICE,
        dtype=torch.float32,
    )

    cutedsl_dual_gemm(x_fp8, w_combined, out_fp8, x_scale_t, w_scale_t, o_scale)
    torch.cuda.synchronize()

    # Reference: dequantize with correct per-shard scales, compute in FP32
    x_deq = x_fp8.float() * x_scale
    wg_deq = w_gate_fp8.float() * w_scale_gate
    wu_deq = w_up_fp8.float() * w_scale_up
    ref_fp32 = torch.nn.functional.silu(x_deq @ wg_deq) * (x_deq @ wu_deq)
    ref_fp8 = torch.clamp(ref_fp32 / o_scale, -448.0, 448.0).to(torch.float8_e4m3fn)

    torch.testing.assert_close(out_fp8.float(), ref_fp8.float(), atol=8.0, rtol=0.15)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
    pytest sgl_diffusion/sgl_diffusion/tests/inference/triton_ops/test_layernorm_perf.py
"""

import pytest
import torch
import triton
import triton.language as tl


# Implementation 1: torch.compile
# To ensure torch.compile is active and using an optimized backend,
# we can explicitly set the backend. "inductor" is the recommended
# backend for modern NVIDIA GPUs.
@torch.compile(backend="inductor")
def layer_norm_torch_compiled(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states
    mean = hidden_states.mean(-1, keepdim=True)
    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states - mean) * torch.rsqrt(variance + variance_epsilon)
    hidden_states = weight * hidden_states
    return hidden_states.to(input_dtype)


# Implementation 2: Triton
@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"HAS_WEIGHT": lambda args: args["W"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    if HAS_WEIGHT:
        W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        y = x_hat * w
    else:
        y = x_hat

    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
        y += b

    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def layer_norm_triton(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    _layer_norm_fwd_1pass_kernel[grid](
        x,
        out,
        weight,
        bias,
        z,
        mean,
        rstd,
        x.stride(0),
        out.stride(0),
        z.stride(0) if z is not None else 0,
        M,
        group_size,
        eps,
        BLOCK_N=BLOCK_N,
        NORM_BEFORE_GATE=norm_before_gate,
        IS_RMS_NORM=is_rms_norm,
        num_warps=num_warps,
    )
    return out


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1024, 4096])
@pytest.mark.parametrize("inner_dim", [768, 1152, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layernorm_accuracy(batch_size, seq_len, inner_dim, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA.")

    device = "cuda"
    eps = 1e-5

    # Create input tensors
    x_3d = torch.randn(batch_size, seq_len, inner_dim, device=device, dtype=dtype)
    weight = torch.randn(inner_dim, device=device, dtype=dtype)
    bias = torch.randn(inner_dim, device=device, dtype=dtype)

    # Reshape for Triton which expects 2D input
    x_2d = x_3d.view(-1, inner_dim).contiguous()

    # The tolerance needs to be adjusted for lower precision dtypes
    rtol, atol = {
        torch.float16: (1e-2, 1e-2),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]

    # --- Correctness Check ---
    # Reference PyTorch implementation (with bias)
    ref_output_bias = torch.nn.functional.layer_norm(
        x_3d, (inner_dim,), weight, bias, eps
    )
    # Triton implementation (with bias)
    triton_output_bias = layer_norm_triton(x_2d, weight, bias, eps)
    triton_output_bias = triton_output_bias.view(x_3d.shape)
    assert torch.allclose(
        ref_output_bias, triton_output_bias, rtol=rtol, atol=atol
    ), "Triton (with bias) output mismatch"

    # Reference PyTorch implementation (no bias)
    ref_output_no_bias = torch.nn.functional.layer_norm(
        x_3d, (inner_dim,), weight, None, eps
    )
    # torch.compile implementation (no bias)
    compiled_output_no_bias = layer_norm_torch_compiled(x_3d, weight, eps)
    assert torch.allclose(
        ref_output_no_bias, compiled_output_no_bias, rtol=rtol, atol=atol
    ), "torch.compile output mismatch"
    # Triton implementation (no bias)
    triton_output_no_bias = layer_norm_triton(x_2d, weight, None, eps)
    triton_output_no_bias = triton_output_no_bias.view(x_3d.shape)
    assert torch.allclose(
        ref_output_no_bias, triton_output_no_bias, rtol=rtol, atol=atol
    ), "Triton (no bias) output mismatch"

    # Reference PyTorch implementation (no weight, no bias)
    # torch.nn.functional.layer_norm with weight=None defaults to ones.
    # Our kernel with weight=None skips the multiplication, which is equivalent.
    ref_output_no_affine = torch.nn.functional.layer_norm(
        x_3d, (inner_dim,), None, None, eps
    )
    # Triton implementation (no weight, no bias)
    triton_output_no_affine = layer_norm_triton(x_2d, None, None, eps)
    triton_output_no_affine = triton_output_no_affine.view(x_3d.shape)
    assert torch.allclose(
        ref_output_no_affine, triton_output_no_affine, rtol=rtol, atol=atol
    ), "Triton (no affine) output mismatch"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1024, 4096])
@pytest.mark.parametrize("inner_dim", [768, 1152, 1536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layernorm_performance(batch_size, seq_len, inner_dim, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA.")

    device = "cuda"
    eps = 1e-5

    # Create input tensors
    x_3d = torch.randn(batch_size, seq_len, inner_dim, device=device, dtype=dtype)
    weight = torch.randn(inner_dim, device=device, dtype=dtype)
    bias = torch.randn(inner_dim, device=device, dtype=dtype)

    # Reshape for Triton which expects 2D input
    x_2d = x_3d.view(-1, inner_dim).contiguous()

    # --- Performance Benchmark ---
    print(f"\nBenchmarking for shape={(batch_size, seq_len, inner_dim)}, dtype={dtype}")

    # Benchmark case: with bias
    pytorch_ms = triton.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(x_3d, (inner_dim,), weight, bias, eps)
    )
    triton_ms = triton.testing.do_bench(
        lambda: layer_norm_triton(x_2d, weight, bias, eps)
    )
    print(f"--- With Bias ---")
    print(f"PyTorch implementation: {pytorch_ms:.4f} ms")
    print(f"Triton implementation:  {triton_ms:.4f} ms")
    print(f"Speedup (Triton vs PyTorch): {pytorch_ms / triton_ms:.2f}x")

    # Benchmark case: no bias
    pytorch_no_bias_ms = triton.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(x_3d, (inner_dim,), weight, None, eps)
    )
    # Warm up for torch.compile
    for _ in range(3):
        layer_norm_torch_compiled(x_3d, weight, eps)
    torch.cuda.synchronize()
    compiled_ms = triton.testing.do_bench(
        lambda: layer_norm_torch_compiled(x_3d, weight, eps)
    )
    triton_no_bias_ms = triton.testing.do_bench(
        lambda: layer_norm_triton(x_2d, weight, None, eps)
    )
    print(f"--- No Bias ---")
    print(f"PyTorch implementation:    {pytorch_no_bias_ms:.4f} ms")
    print(f"torch.compile implementation: {compiled_ms:.4f} ms")
    print(f"Triton implementation:     {triton_no_bias_ms:.4f} ms")
    print(f"Speedup (Triton vs PyTorch): {pytorch_no_bias_ms / triton_no_bias_ms:.2f}x")
    print(f"Speedup (Triton vs torch.compile): {compiled_ms / triton_no_bias_ms:.2f}x")

    # Benchmark case: no weight, no bias
    pytorch_no_affine_ms = triton.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(x_3d, (inner_dim,), None, None, eps)
    )
    triton_no_affine_ms = triton.testing.do_bench(
        lambda: layer_norm_triton(x_2d, None, None, eps)
    )
    print(f"--- No Weight, No Bias ---")
    print(f"PyTorch implementation:    {pytorch_no_affine_ms:.4f} ms")
    print(f"Triton implementation:     {triton_no_affine_ms:.4f} ms")
    print(
        f"Speedup (Triton vs PyTorch): {pytorch_no_affine_ms / triton_no_affine_ms:.2f}x"
    )

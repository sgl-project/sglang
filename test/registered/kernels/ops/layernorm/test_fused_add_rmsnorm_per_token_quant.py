import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")

FP8_MAX = 448.0


def reference_norm_quant(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """fp32 reference: returns (normed_bf16, fp8, per_token_scale, residual_out)."""
    acc = x.to(torch.float32)
    residual_out = None
    if residual is not None:
        acc = acc + residual.to(torch.float32)
        residual_out = acc.to(x.dtype)
    variance = acc.pow(2).mean(-1, keepdim=True)
    normed = acc * torch.rsqrt(variance + eps)
    normed = normed * w.to(torch.float32)

    absmax = normed.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(absmax, min=1e-10) / FP8_MAX
    fp8 = (normed / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return normed.to(x.dtype), fp8, scale, residual_out


@pytest.mark.parametrize(
    "num_tokens,hidden_size,dtype,with_residual",
    list(
        itertools.product(
            # get_ci_test_range(full_range, ci_range): full sweep first, the
            # reduced CI list second.
            get_ci_test_range([1, 2, 7, 64, 128, 1024], [1, 7, 128]),
            # Sizes whose vector width (hidden/8) is NOT a multiple of 32, so
            # blockDim is padded up to a whole warp and the tail warp is only
            # partly active: 264->33, 520->65, 1032->129, 3080->385, 5128->641.
            # (Every multiple of 256 -- 512/4096/8192 -- has an exact warp
            # count and does not exercise that path.)
            get_ci_test_range(
                [264, 512, 520, 1032, 1536, 3080, 4096, 5128, 8192],
                [264, 1032, 4096, 8192],
            ),
            [torch.bfloat16],
            [True, False],
        )
    ),
)
def test_matches_reference(num_tokens, hidden_size, dtype, with_residual):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.kernels.ops.layernorm.fused_norm_quant import (
        can_use_fused_norm_per_token_quant,
        fused_add_rmsnorm_per_token_quant,
        fused_rmsnorm_per_token_quant,
    )

    if not can_use_fused_norm_per_token_quant(hidden_size, dtype):
        pytest.skip(f"kernel unavailable for hidden_size={hidden_size}")

    torch.manual_seed(0)
    eps = 1e-6
    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    w = torch.randn((hidden_size,), dtype=dtype, device="cuda")
    residual = (
        torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
        if with_residual
        else None
    )

    ref_bf16, ref_fp8, ref_scale, ref_residual = reference_norm_quant(
        x, residual, w, eps
    )

    if with_residual:
        residual_actual = residual.clone()
        fp8, scale = fused_add_rmsnorm_per_token_quant(x, residual_actual, w, eps)
        torch.testing.assert_close(
            residual_actual.to(torch.float32),
            ref_residual.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    else:
        fp8, scale = fused_rmsnorm_per_token_quant(x, w, eps)

    torch.testing.assert_close(scale, ref_scale, rtol=1e-3, atol=1e-8)

    # Dequantized fp8 must track the fp32 reference within one fp8 ulp-ish band.
    deq = fp8.to(torch.float32) * scale
    torch.testing.assert_close(deq, ref_bf16.to(torch.float32), rtol=8e-2, atol=8e-2)


def test_scale_shape_and_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.kernels.ops.layernorm.fused_norm_quant import (
        can_use_fused_norm_per_token_quant,
        fused_add_rmsnorm_per_token_quant,
    )

    hidden_size, num_tokens, dtype = 4096, 17, torch.bfloat16
    if not can_use_fused_norm_per_token_quant(hidden_size, dtype):
        pytest.skip("kernel unavailable")

    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    w = torch.randn((hidden_size,), dtype=dtype, device="cuda")

    fp8, scale = fused_add_rmsnorm_per_token_quant(x, residual, w, 1e-6)
    assert scale.shape == (num_tokens, 1)
    assert scale.dtype == torch.float32
    assert fp8.dtype == torch.float8_e4m3fn
    assert fp8.shape == (num_tokens, hidden_size)


def test_all_zero_row_does_not_produce_nan():
    """An all-zero row must yield a finite scale, not 1/0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.kernels.ops.layernorm.fused_norm_quant import (
        can_use_fused_norm_per_token_quant,
        fused_rmsnorm_per_token_quant,
    )

    hidden_size, dtype = 4096, torch.bfloat16
    if not can_use_fused_norm_per_token_quant(hidden_size, dtype):
        pytest.skip("kernel unavailable")

    x = torch.zeros((4, hidden_size), dtype=dtype, device="cuda")
    w = torch.ones((hidden_size,), dtype=dtype, device="cuda")
    fp8, scale = fused_rmsnorm_per_token_quant(x, w, 1e-6)
    assert torch.isfinite(scale).all()
    assert torch.isfinite(fp8.to(torch.float32)).all()


def test_cuda_graph_capture_replay():
    """The fused path must be capturable and replay with fresh data.

    This is the regression guard for the reason the original attribute-stamping
    design was rejected: the (fp8, scale) pair must be real tensors owned by the
    caller, so a replay writes into the same buffers the consumer reads.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.kernels.ops.layernorm.fused_norm_quant import (
        can_use_fused_norm_per_token_quant,
        fused_add_rmsnorm_per_token_quant,
    )

    hidden_size, num_tokens, dtype = 4096, 8, torch.bfloat16
    if not can_use_fused_norm_per_token_quant(hidden_size, dtype):
        pytest.skip("kernel unavailable")

    x = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    w = torch.randn((hidden_size,), dtype=dtype, device="cuda")

    # Warm up the JIT load and the allocator outside the capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fused_add_rmsnorm_per_token_quant(x, residual.clone(), w, 1e-6)
    torch.cuda.current_stream().wait_stream(s)

    static_res = residual.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        g_fp8, g_scale = fused_add_rmsnorm_per_token_quant(x, static_res, w, 1e-6)

    x2 = torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda")
    x.copy_(x2)
    static_res.copy_(torch.randn_like(static_res))
    expected_res = static_res.clone()
    g.replay()
    torch.cuda.synchronize()

    _, ref_fp8, ref_scale, _ = reference_norm_quant(x, expected_res, w, 1e-6)
    torch.testing.assert_close(g_scale, ref_scale, rtol=1e-3, atol=1e-8)
    torch.testing.assert_close(
        g_fp8.to(torch.float32) * g_scale,
        ref_fp8.to(torch.float32) * ref_scale,
        rtol=8e-2,
        atol=8e-2,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

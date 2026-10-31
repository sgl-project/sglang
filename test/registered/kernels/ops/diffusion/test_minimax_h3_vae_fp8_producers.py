"""FP8 producers of the MiniMax-H3 VAE decoder: fp32 residual + RMSNorm /
LayerNorm with a dynamic per-token FP8 epilogue, and fp32 silu(gate) * up + FP8.

References are public PyTorch operators (``F.rms_norm``, ``F.layer_norm``,
``F.silu``) followed by the eager dynamic per-token quantizer. The kernels use
a different fp32 reduction tree, so a value can land in the neighbouring E4M3
bin; every FP8 check asserts exact scales, >= 99.9% identical bins and a
pointwise half-ulp bound on the dequantized values.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_residual_layernorm,
    can_use_minimax_h3_vae_residual_rmsnorm_fp8,
    can_use_minimax_h3_vae_rmsnorm_fp8,
    can_use_silu_mul_quant_fp8,
    minimax_h3_vae_residual_layernorm,
    minimax_h3_vae_residual_rmsnorm_fp8,
    minimax_h3_vae_rmsnorm_fp8,
    silu_mul_quant_fp8,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen.test.quality_tier_admission import discrete_flip_rate
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE FP8 producers are gated to SM100+",
)

WIDTH = 2048
FFN_WIDTH = 8192
EPS = 1e-5
FP8_MAX = 448.0


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260920)
    with torch.inference_mode():
        yield


def record_error(record_property, actual, expected, prefix="output"):
    difference = (actual.float() - expected.float()).abs()
    finite = difference[torch.isfinite(difference)]
    if finite.numel():
        record_property(f"{prefix}_max_abs", finite.max().item())
        record_property(f"{prefix}_rms_abs", finite.square().mean().sqrt().item())


def check_fp8(q, scales, reference, record_property):
    reference = reference.reshape(q.shape)
    expected_scale = reference.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    torch.testing.assert_close(scales, expected_scale, atol=1e-8, rtol=3e-6)
    expected_q = (
        (reference / expected_scale.clamp_min(torch.finfo(torch.float32).tiny))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    mismatch = discrete_flip_rate(expected_q.view(torch.uint8), q.view(torch.uint8))
    record_property("fp8_boundary_mismatch_fraction", mismatch)
    # A different reduction tree can move values across an E4M3 midpoint.
    # Require >99.9% identical bins as well as a pointwise half-ULP error bound.
    assert mismatch <= 1e-3
    dequantized = q.float() * scales
    error = (dequantized - reference).abs()
    bound = 0.063 * reference.abs() + scales / 1024 + 1e-6
    assert torch.all(error <= bound).item()
    record_error(record_property, dequantized, reference, "dequantized")
    zero_rows = expected_scale.squeeze(-1) == 0
    if zero_rows.any():
        assert torch.count_nonzero(q.float()[zero_rows]) == 0
        assert torch.count_nonzero(scales[zero_rows]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, WIDTH), (2, 7, WIDTH)])
def test_rmsnorm_fp8(dtype, shape, record_property):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    if x.numel() > WIDTH:
        x.view(-1, WIDTH)[0].zero_()
    weight = torch.randn(WIDTH, device="cuda")
    assert can_use_minimax_h3_vae_rmsnorm_fp8(x, weight, eps=EPS)
    q, scales = minimax_h3_vae_rmsnorm_fp8(x, weight, eps=EPS)
    rows = x.numel() // WIDTH
    assert q.shape == (rows, WIDTH) and q.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, 1) and scales.dtype == torch.float32
    reference = F.rms_norm(x.float(), (WIDTH,), weight, eps=EPS)
    check_fp8(q, scales, reference, record_property)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("projected_dtype", [torch.float16, torch.float32])
def test_residual_rmsnorm_fp8(dtype, projected_dtype, record_property):
    x = torch.randn(2, 7, WIDTH, dtype=dtype, device="cuda")
    projected = torch.randn(2, 7, WIDTH, dtype=projected_dtype, device="cuda")
    x[0, 0].zero_()
    projected[0, 0].zero_()
    layer_scale = torch.randn(WIDTH, device="cuda") * 0.3
    weight = torch.randn(WIDTH, device="cuda")
    assert can_use_minimax_h3_vae_residual_rmsnorm_fp8(
        x, projected, layer_scale, weight, eps=EPS
    )
    residual, q, scales = minimax_h3_vae_residual_rmsnorm_fp8(
        x, projected, layer_scale, weight, eps=EPS
    )
    expected_residual = x.float() + projected.float() * layer_scale
    assert residual.shape == x.shape and residual.dtype == torch.float32
    torch.testing.assert_close(residual, expected_residual, atol=1e-6, rtol=2e-6)
    reference = F.rms_norm(expected_residual, (WIDTH,), weight, eps=EPS)
    check_fp8(q, scales, reference, record_property)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("projected_dtype", [torch.float16, torch.float32])
def test_residual_layernorm(dtype, projected_dtype, record_property):
    x = torch.randn(2, 7, WIDTH, device="cuda", dtype=dtype)
    projected = torch.randn(2, 7, WIDTH, device="cuda", dtype=projected_dtype)
    layer_scale = torch.randn(WIDTH, device="cuda") * 0.3
    weight, bias = torch.randn(2, WIDTH, device="cuda")
    reference = F.layer_norm(
        x.float() + projected.float() * layer_scale, (WIDTH,), weight, bias, EPS
    )
    assert can_use_minimax_h3_vae_residual_layernorm(
        x, projected, layer_scale, weight, bias, eps=EPS
    )
    out = minimax_h3_vae_residual_layernorm(
        x, projected, layer_scale, weight, bias, eps=EPS
    )
    assert out.shape == x.shape and out.dtype == torch.float32
    record_error(record_property, out, reference)
    torch.testing.assert_close(out, reference, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 257])
def test_silu_mul_quant_fp8(dtype, rows, record_property):
    x = torch.randn(rows, 2 * FFN_WIDTH, device="cuda", dtype=dtype)
    if rows > 1:
        x[0].zero_()
    gate, up = x.float().chunk(2, dim=-1)
    reference = F.silu(gate) * up
    assert can_use_silu_mul_quant_fp8(x)
    q, scales = silu_mul_quant_fp8(x)
    assert q.shape == (rows, FFN_WIDTH) and q.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, 1) and scales.dtype == torch.float32
    check_fp8(q, scales, reference, record_property)


def test_silu_mul_quant_fp8_flattens_leading_dims(record_property):
    x = torch.randn(2, 5, 2 * FFN_WIDTH, device="cuda", dtype=torch.float16)
    gate, up = x.float().chunk(2, dim=-1)
    q, scales = silu_mul_quant_fp8(x)
    assert q.shape == (10, FFN_WIDTH) and scales.shape == (10, 1)
    check_fp8(q, scales, F.silu(gate) * up, record_property)


def test_reject_unsupported_inputs():
    x = torch.randn(2, 7, WIDTH, device="cuda")
    projected = torch.randn_like(x, dtype=torch.float16)
    weight, bias = torch.randn(2, WIDTH, device="cuda")
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(x.cpu(), weight.cpu(), eps=EPS)
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(x, weight.half(), eps=EPS)
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(x, weight, eps=0.0)
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(x.transpose(0, 1), weight, eps=EPS)
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(
        x[..., : WIDTH // 2], weight[: WIDTH // 2], eps=EPS
    )
    assert not can_use_minimax_h3_vae_residual_rmsnorm_fp8(
        x, projected[0], weight, weight, eps=EPS
    )
    assert not can_use_minimax_h3_vae_residual_layernorm(
        x, projected, weight, weight, bias.half(), eps=EPS
    )
    with pytest.raises(ValueError, match="at least two dimensions"):
        minimax_h3_vae_rmsnorm_fp8(x[0, 0], weight, eps=EPS)
    # Equal element counts must not hide different original tensor shapes.
    for residual, update in (
        (x, projected.view(7, 2, WIDTH)),
        (x[0, 0], projected[0, 0]),
    ):
        with pytest.raises(ValueError, match="same shape and at least two dimensions"):
            minimax_h3_vae_residual_rmsnorm_fp8(
                residual, update, weight, weight, eps=EPS
            )
        with pytest.raises(ValueError, match="same shape and at least two dimensions"):
            minimax_h3_vae_residual_layernorm(
                residual, update, weight, weight, bias, eps=EPS
            )
    # A contiguous slice can still be misaligned for the vectorized loads.
    offset_rows = torch.empty(2 * WIDTH + 4, device="cuda")[1 : 2 * WIDTH + 1]
    offset_rows = offset_rows.view(2, WIDTH)
    assert offset_rows.is_contiguous() and offset_rows.data_ptr() % 16 != 0
    assert not can_use_minimax_h3_vae_rmsnorm_fp8(offset_rows, weight, eps=EPS)
    with pytest.raises(RuntimeError, match="aligned"):
        minimax_h3_vae_rmsnorm_fp8(offset_rows, weight, eps=EPS)
    with pytest.raises(RuntimeError):
        minimax_h3_vae_rmsnorm_fp8(x, weight.half(), eps=EPS)
    offset_ffn = torch.empty(2 * FFN_WIDTH + 1, dtype=torch.float16, device="cuda")
    offset_ffn = offset_ffn[1:].view(1, 2 * FFN_WIDTH)
    assert offset_ffn.is_contiguous() and offset_ffn.data_ptr() % 16 != 0
    assert not can_use_silu_mul_quant_fp8(offset_ffn)
    with pytest.raises(RuntimeError, match="aligned"):
        silu_mul_quant_fp8(offset_ffn)
    assert not can_use_silu_mul_quant_fp8(
        torch.randn(3, 2 * 1536, device="cuda").half()
    )
    assert not can_use_silu_mul_quant_fp8(torch.randn(3, 2 * FFN_WIDTH, device="cuda"))
    with pytest.raises(RuntimeError, match="width"):
        silu_mul_quant_fp8(torch.randn(3, 2 * 1536, device="cuda").half())
    with pytest.raises(ValueError, match="at least two dimensions"):
        silu_mul_quant_fp8(
            torch.empty(2 * FFN_WIDTH, device="cuda", dtype=torch.float16)
        )
    # Flattening this odd width used to reinterpret it as 4097 valid 4096-wide rows.
    with pytest.raises(ValueError, match="even gate/up width"):
        silu_mul_quant_fp8(torch.empty(4096, 4097, device="cuda", dtype=torch.float16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))

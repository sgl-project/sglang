# SPDX-License-Identifier: Apache-2.0
"""Reference tests for MiniMax-M3 ROCm Gemma RMSNorm Triton kernels."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

if not is_hip():
    pytest.skip(
        "MiniMax-M3 Gemma RMSNorm Triton kernels are ROCm-only.",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (  # noqa: E402
    _FOUR_ROWS_MIN_ROWS,
    gemma_add_rmsnorm_fp8,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = "cuda"
_GFX95_ONLY = pytest.mark.skipif(
    not is_gfx95_supported(), reason="the fp8 producers feed gfx950-only GEMMs"
)
EPS = 1e-6
# four rows per program, the last one partial
FOUR_ROW_TOKENS = _FOUR_ROWS_MIN_ROWS + 3


def _gemma_rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    out = x_f * torch.rsqrt(variance + EPS)
    out = out * (1.0 + weight.float())
    return out.to(orig_dtype)


@pytest.mark.parametrize("shape", [(1, 512), (64, 6144), (257, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_rmsnorm_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(shape[-1], device=DEVICE, dtype=torch.float32)

    got = gemma_rmsnorm(x, weight, EPS)
    ref = _gemma_rmsnorm_ref(x, weight)

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_rmsnorm_accepts_strided_2d_input(dtype):
    torch.manual_seed(0)
    base = torch.randn(128, 1024, device=DEVICE, dtype=dtype)
    x = base[:, ::2]
    weight = torch.randn(x.shape[-1], device=DEVICE, dtype=torch.float32)

    assert not x.is_contiguous()
    got = gemma_rmsnorm(x, weight, EPS)
    ref = _gemma_rmsnorm_ref(x, weight)

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("shape", [(1, 512), (64, 6144), (257, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_fused_add_rmsnorm_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    residual = torch.randn(*shape, device=DEVICE, dtype=dtype)
    weight = torch.randn(shape[-1], device=DEVICE, dtype=torch.float32)

    got, residual_out = gemma_fused_add_rmsnorm(x, residual, weight, EPS)
    ref_residual = x + residual
    ref = _gemma_rmsnorm_ref(ref_residual, weight)

    torch.testing.assert_close(residual_out, ref_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _norm_then_quant(x, residual, weight, quant_format="fp8_per_token"):
    import aiter

    if residual is None:
        normed, residual_out = gemma_rmsnorm(x, weight, EPS), x
    else:
        normed, residual_out = gemma_fused_add_rmsnorm(x, residual, weight, EPS)
    if quant_format == "fp8_per_token":
        q, scale = aiter.per_token_quant_hip(normed, quant_dtype=torch.float8_e4m3fn)
    else:
        q, scale = aiter.get_hip_quant(aiter.QuantType.per_1x128)(
            normed, quant_dtype=torch.float8_e4m3fn, transpose_scale=False
        )
    return q, scale, normed, residual_out


def _decode_inputs(num_tokens: int):
    torch.manual_seed(num_tokens)
    # per-token magnitudes spread over orders, like a residual stream
    scale = torch.exp(torch.randn(num_tokens, 1, device=DEVICE) * 1.5)
    x = (torch.randn(num_tokens, 6144, device=DEVICE) * scale).to(torch.bfloat16)
    residual = (torch.randn(num_tokens, 6144, device=DEVICE) * 20 * scale).to(
        torch.bfloat16
    )
    weight = (torch.randn(6144, device=DEVICE) * 0.3).to(torch.bfloat16)
    return x, residual, weight


@_GFX95_ONLY
@pytest.mark.parametrize("num_tokens", [1024, FOUR_ROW_TOKENS])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("keep_bf16", [True, False])
@torch.inference_mode()
def test_gemma_add_rmsnorm_fp8_per_token_writes_the_separate_kernels_bytes(
    num_tokens, with_residual, keep_bf16
):
    """The fused norm + quant hands the ptpc GEMM what the norm kernels followed by
    aiter's per_token_quant_hip write: same norm code and launch, and the norm
    rounded to bf16 before the quant."""
    x, residual, weight = _decode_inputs(num_tokens)
    residual = residual if with_residual else None
    q_ref, scale_ref, normed_ref, residual_ref = _norm_then_quant(x, residual, weight)

    quantized, residual_out = gemma_add_rmsnorm_fp8(
        x, residual, weight, EPS, keep_bf16=keep_bf16
    )

    assert torch.equal(_bits(quantized[0]), _bits(q_ref))
    assert torch.equal(_bits(quantized[1]), _bits(scale_ref))
    assert torch.equal(_bits(residual_out), _bits(residual_ref))
    if keep_bf16:
        assert torch.equal(_bits(quantized[2]), _bits(normed_ref))


@_GFX95_ONLY
@pytest.mark.parametrize("num_tokens", [1000, FOUR_ROW_TOKENS])
@pytest.mark.parametrize("with_residual", [True, False])
@pytest.mark.parametrize("transpose_scale", [True, False])
@torch.inference_mode()
def test_gemma_add_rmsnorm_fp8_group_writes_the_separate_kernels_bytes(
    num_tokens, with_residual, transpose_scale
):
    """Per 1 x 128 the fused norm + quant writes what the norm kernels followed by
    aiter's per_1x128 quant write, an all-zero row's 1e-10 amax floor included;
    transpose_scale stores the scale column-major."""
    x, residual, weight = _decode_inputs(num_tokens)
    x[0] = 0
    residual[0] = 0
    residual = residual if with_residual else None
    q_ref, scale_ref, normed_ref, residual_ref = _norm_then_quant(
        x, residual, weight, quant_format="fp8"
    )

    quantized, residual_out = gemma_add_rmsnorm_fp8(
        x,
        residual,
        weight,
        EPS,
        group_size=128,
        transpose_scale=transpose_scale,
        keep_bf16=True,
    )

    assert torch.equal(_bits(quantized[0]), _bits(q_ref))
    assert torch.equal(_bits(quantized[1]), _bits(scale_ref))
    assert quantized[1].stride() == ((1, num_tokens) if transpose_scale else (48, 1))
    assert torch.equal(_bits(residual_out), _bits(residual_ref))
    assert torch.equal(_bits(quantized[2]), _bits(normed_ref))


@_GFX95_ONLY
@torch.inference_mode()
def test_four_rows_without_bf16_write_the_same_fp8_and_zero_fill():
    """Four rows per program without the bf16 norm (the prefill attention input) write
    the fp8, scales and residual of the launch that keeps it, and zero-fill the GEMM
    output they are handed."""
    num_tokens = FOUR_ROW_TOKENS
    x, residual, weight = _decode_inputs(num_tokens)
    (q_ref, scale_ref, _), residual_ref = gemma_add_rmsnorm_fp8(
        x, residual, weight, EPS, group_size=128, transpose_scale=True, keep_bf16=True
    )
    zeros = torch.full((num_tokens, 100), 7.0, device=DEVICE, dtype=torch.bfloat16)

    (q, scale), residual_out = gemma_add_rmsnorm_fp8(
        x, residual, weight, EPS, group_size=128, transpose_scale=True, zeros=zeros
    )

    assert torch.equal(_bits(q), _bits(q_ref))
    assert torch.equal(_bits(scale), _bits(scale_ref))
    assert torch.equal(_bits(residual_out), _bits(residual_ref))
    assert not zeros.any()


@_GFX95_ONLY
@pytest.mark.parametrize(
    "quant_format, num_tokens", [("fp8_per_token", 64), ("fp8", 256)]
)
@torch.inference_mode()
def test_fp8_reads_of_a_gemma_norm_apply_one_plus_weight(quant_format, num_tokens):
    """The attention and FFN reads that quantize for a dense fp8 GEMM must normalize
    as GemmaRMSNorm does (1 + weight), not with the raw weight that aiter's
    plain-RMSNorm kernels take."""
    from sglang.srt.layers.layer_boundary.residual.add_norm import (
        NORM_QUANT_READOUT,
        PLAIN_ADD,
    )
    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.models.minimax_common.amd.minimax_m3_hip import NormFp8Readout

    x, residual, weight = _decode_inputs(num_tokens)
    norm = GemmaRMSNorm(6144, eps=EPS).to(DEVICE, torch.bfloat16)
    norm.weight.data.copy_(weight)
    q_ref, scale_ref, normed_ref, residual_ref = _norm_then_quant(
        x, residual, weight, quant_format
    )

    attn_input, attn_residual = NORM_QUANT_READOUT.update_and_read(
        PLAIN_ADD, x, residual, norm, quant_format=quant_format
    )
    ffn_input, ffn_residual = NormFp8Readout(
        quant_format_for=lambda num_tokens: quant_format
    ).update_and_read(PLAIN_ADD, x, residual, norm)

    for quantized, residual_out in (
        (attn_input, attn_residual),
        (ffn_input, ffn_residual),
    ):
        assert torch.equal(_bits(quantized[0]), _bits(q_ref))
        assert torch.equal(_bits(quantized[1]), _bits(scale_ref))
        assert torch.equal(_bits(residual_out), _bits(residual_ref))
    assert torch.equal(_bits(ffn_input[2]), _bits(normed_ref))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

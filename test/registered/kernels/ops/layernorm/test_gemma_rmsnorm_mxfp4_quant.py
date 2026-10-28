# SPDX-License-Identifier: Apache-2.0
"""gfx950 byte tests for the fused Gemma add+RMSNorm+MXFP4 quant producer:
gemma_fused_add_rmsnorm_mxfp4_quant, and the Qwen3.5 norm that runs it, must
write the fp4x2 codes and shuffled E8M0 scales that the unfused chain writes
(the ROCm Triton Gemma norm, then AITER per_1x32_f4_quant(shuffle=True)),
byte for byte, and the same residual."""

import pytest
import torch

from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip

if not (is_hip() and torch.cuda.is_available() and is_gfx95_supported()):
    pytest.skip("The MXFP4 quant epilogue needs gfx950.", allow_module_level=True)

from aiter import per_1x32_f4_quant_hip  # noqa: E402

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_fused_add_rmsnorm_mxfp4_quant,
)
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

DEVICE = "cuda"
EPS = 1e-6
# Qwen3.8-27B hidden size; decode, the 32-row scale tiles, the 256-row scale
# padding and prefill chunks.
HIDDEN = 5120
M_VALUES = [1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 64, 255, 256, 257, 870]
M_VALUES += [2048, 4377, 6553, 16384]


def _unshuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 scales in the AITER ASM FP4 GEMM tile order to row-major."""
    rows, cols = scales.shape
    return (
        scales.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)
    )


def _assert_same_mxfp4(q, scales, ref_q, ref_scales, m):
    """Same fp4 bytes and same scales for the m valid rows; no NaN (0xFF) scale."""
    assert scales.shape == ref_scales.shape
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)[:m])
    got = _unshuffle_scales(scales)[:m]
    assert torch.equal(got, _unshuffle_scales(ref_scales)[:m])
    assert not bool((got == 0xFF).any())


def _inputs(m, n, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(m, n, device=DEVICE, generator=g).to(torch.bfloat16)
    residual = torch.randn(m, n, device=DEVICE, generator=g).to(torch.bfloat16)
    weight = (0.1 * torch.randn(n, device=DEVICE, generator=g)).to(torch.bfloat16)
    return x, residual, weight


def _check_kernel(x, residual, weight):
    m = x.shape[0]
    out, residual_ref = gemma_fused_add_rmsnorm(x, residual, weight, EPS)
    ref_q, ref_scales = per_1x32_f4_quant_hip(out, shuffle=True)
    q, scales, residual_out = gemma_fused_add_rmsnorm_mxfp4_quant(
        x, residual, weight, EPS
    )
    assert q.shape == (m, x.shape[1] // 2)
    _assert_same_mxfp4(q, scales, ref_q, ref_scales, m)
    assert torch.equal(residual_out, residual_ref)


@pytest.mark.parametrize("m", M_VALUES)
@torch.inference_mode()
def test_matches_norm_then_quant(m):
    _check_kernel(*_inputs(m, HIDDEN, seed=m))


@pytest.mark.parametrize("n", [1024, 2560, 4096, 8192])
@pytest.mark.parametrize("m", [1, 33, 257, 2048])
@torch.inference_mode()
def test_matches_norm_then_quant_other_hidden_sizes(m, n):
    _check_kernel(*_inputs(m, n, seed=m + n))


@pytest.mark.parametrize("m", [1, 33, 2048])
@torch.inference_mode()
def test_edge_blocks(m):
    x, residual, weight = _inputs(m, HIDDEN, seed=7)
    residual = torch.zeros_like(residual)
    x[0, :64] = 0  # all-zero blocks (amax floor)
    x[-1, 32:64] = 1e-12  # blocks below the amax floor
    x[-1, 64:96] *= 1e4  # large blocks
    weight[96:128] = -1.0  # (1 + weight) == 0: all-zero output blocks
    _check_kernel(x, residual, weight)


@pytest.mark.skipif(
    not get_bool_env_var("SGLANG_USE_AITER"),
    reason="GemmaRMSNorm runs the ROCm Triton norm with SGLANG_USE_AITER=1",
)
@pytest.mark.parametrize(
    "m", [1, 7, 8, 9, 64, 255, 256, 257, 870, 2048, 4377, 6553, 16384]
)
@torch.inference_mode()
def test_qwen3_5_norm_matches_unfused(m):
    """The Qwen3.5 norm in front of an MXFP4 projection hands it what the
    projection would quantize from GemmaRMSNorm, in the layout its AITER ASM
    FP4 GEMM takes as is."""
    from types import SimpleNamespace

    from sglang.srt.layers.layernorm import GemmaRMSNorm
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
        _AsmShuffledMxfp4Input,
        _validate_asm_prequantized_input,
    )
    from sglang.srt.models.qwen3_5 import _GemmaRMSNormMXFP4Quant

    x, residual, weight = _inputs(m, HIDDEN, seed=m)
    fused = _GemmaRMSNormMXFP4Quant(HIDDEN, EPS, lambda num_tokens: True)
    ref = GemmaRMSNorm(HIDDEN, eps=EPS)
    for norm in (fused, ref):
        norm.to(DEVICE, torch.bfloat16)
        norm.weight.data.copy_(weight)

    out, residual_out = fused.forward_hip(x, residual)
    y, residual_ref = ref.forward_hip(x, residual)
    assert isinstance(out, _AsmShuffledMxfp4Input)
    ref_q, ref_scales = per_1x32_f4_quant_hip(y, shuffle=True)
    _assert_same_mxfp4(*out, ref_q, ref_scales, m)
    assert torch.equal(residual_out, residual_ref)
    layer = SimpleNamespace(
        weight=torch.empty(256, HIDDEN // 2, dtype=torch.uint8, device=DEVICE)
    )
    _validate_asm_prequantized_input(layer, out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

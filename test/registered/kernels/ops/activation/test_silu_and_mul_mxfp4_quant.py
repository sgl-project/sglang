"""gfx950 byte tests for the Qwen3.5 dense-MLP SiLU+mul+MXFP4 quant producer:
its fused path (AITER act_mul_and_mxfp4_quant) must hand down_proj the fp4x2
codes and shuffled E8M0 scales that the unfused path gives it (SiluAndMul, then
AITER per_1x32_f4_quant(shuffle=True)), byte for byte."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

if not (is_hip() and torch.cuda.is_available() and is_gfx95_supported()):
    pytest.skip("The AITER ASM FP4 GEMM input needs gfx950.", allow_module_level=True)

from aiter import dtypes as aiter_dtypes  # noqa: E402
from aiter import per_1x32_f4_quant_hip  # noqa: E402
from aiter.ops.triton.fusions.activation import act_mul_and_mxfp4_quant  # noqa: E402

from sglang.srt.layers.activation import SiluAndMul  # noqa: E402
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (  # noqa: E402
    _AsmShuffledMxfp4Input,
    _validate_asm_prequantized_input,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

if "round_to_input_dtype" not in inspect.signature(act_mul_and_mxfp4_quant).parameters:
    pytest.skip(
        "This AITER cannot round like SiluAndMul; Qwen3.5 keeps the fusion off.",
        allow_module_level=True,
    )

DEVICE = "cuda"
HIDDEN = 5120
# Qwen3.8-27B down_proj K per rank: TP2 and TP1.
INTERMEDIATE_SIZES = [8704, 17408]
M_VALUES = [1, 7, 8, 9, 64, 255, 256, 257, 870, 2048, 4377, 6553, 16384]
# What qwen3_5 sets on an MLP whose fusion it enables.
_MLP = SimpleNamespace(
    _mxfp4_x_dtype=aiter_dtypes.fp4x2, _mxfp4_scale_dtype=aiter_dtypes.fp8_e8m0
)


def _unshuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 scales in the AITER ASM FP4 GEMM tile order to row-major."""
    rows, cols = scales.shape
    return (
        scales.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)
    )


def _check(gate_up):
    m, n = gate_up.shape[0], gate_up.shape[1] // 2
    out = Qwen2MoeMLP._silu_mxfp4_quant_fused(_MLP, gate_up)
    assert isinstance(out, _AsmShuffledMxfp4Input)
    ref_q, ref_scales = per_1x32_f4_quant_hip(SiluAndMul()(gate_up), shuffle=True)
    q, scales = out
    assert scales.shape == ref_scales.shape
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)[:m])
    got = _unshuffle_scales(scales)[:m]
    assert torch.equal(got, _unshuffle_scales(ref_scales)[:m])
    assert not bool((got == 0xFF).any())
    # The layout down_proj's AITER ASM FP4 GEMM takes as is.
    down_proj = SimpleNamespace(
        weight=torch.empty(HIDDEN, n // 2, dtype=torch.uint8, device=DEVICE)
    )
    _validate_asm_prequantized_input(down_proj, out)


@pytest.mark.parametrize("n", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("m", M_VALUES)
@torch.inference_mode()
def test_matches_silu_and_mul_then_quant(m, n):
    g = torch.Generator(device=DEVICE).manual_seed(m + n)
    gate_up = torch.randn(m, 2 * n, device=DEVICE, generator=g).to(torch.bfloat16)
    _check(gate_up)


@pytest.mark.parametrize("n", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("m", [1, 33, 2048])
@torch.inference_mode()
def test_edge_blocks(m, n):
    g = torch.Generator(device=DEVICE).manual_seed(7)
    gate_up = torch.randn(m, 2 * n, device=DEVICE, generator=g).to(torch.bfloat16)
    gate_up[0, :64] = 0  # all-zero blocks (amax floor)
    gate_up[-1, 32:64] = 1e-12  # blocks below the amax floor
    gate_up[-1, n + 64 : n + 96] *= 64  # large blocks
    _check(gate_up)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

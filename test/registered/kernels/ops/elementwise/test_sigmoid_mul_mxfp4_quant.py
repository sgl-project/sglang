# SPDX-License-Identifier: Apache-2.0
"""gfx950 byte tests for fused_sigmoid_mul_mxfp4_quant: it must write the fp4x2
codes and shuffled E8M0 scales that fused_sigmoid_mul followed by AITER
per_1x32_f4_quant(shuffle=True) writes, byte for byte."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip

if not (is_hip() and torch.cuda.is_available() and is_gfx95_supported()):
    pytest.skip("The MXFP4 quant epilogue needs gfx950.", allow_module_level=True)

from aiter import per_1x32_f4_quant_hip  # noqa: E402

from sglang.kernels.ops.elementwise.elementwise import (  # noqa: E402
    fused_sigmoid_mul,
    fused_sigmoid_mul_mxfp4_quant,
)
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

DEVICE = "cuda"
# Decode, the 32-row scale tiles, the 256-row scale padding and prefill chunks.
M_VALUES = [1, 2, 7, 16, 31, 32, 33, 255, 256, 257, 870, 2048, 3579, 16384]


def _unshuffle_scales(scales: torch.Tensor) -> torch.Tensor:
    """E8M0 scales in the AITER ASM FP4 GEMM tile order to row-major."""
    rows, cols = scales.shape
    return (
        scales.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)
    )


def _inputs(m, num_heads, head_dim, seed, strided_gate):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    attn = torch.randn(m, num_heads * head_dim, device=DEVICE, generator=g)
    if strided_gate:  # gate heads between the q heads, as in the q/gate projection
        q_gate = torch.randn(m, num_heads, 2 * head_dim, device=DEVICE, generator=g)
        gate = q_gate.to(torch.bfloat16)[..., head_dim:]
    else:
        gate = torch.randn(m, num_heads, head_dim, device=DEVICE, generator=g)
        gate = gate.to(torch.bfloat16)
    return attn.to(torch.bfloat16), gate


def _check(attn, gate):
    m = attn.shape[0]
    gated = fused_sigmoid_mul(attn.clone(), gate, inplace=True)
    ref_q, ref_scales = per_1x32_f4_quant_hip(gated, shuffle=True)
    q, scales = fused_sigmoid_mul_mxfp4_quant(attn, gate)
    assert q.shape == (m, attn.shape[1] // 2)
    assert scales.shape == ref_scales.shape
    assert torch.equal(q, ref_q.view(torch.uint8)[:m])
    got = _unshuffle_scales(scales)[:m]
    assert torch.equal(got, _unshuffle_scales(ref_scales)[:m])
    assert not bool((got == 0xFF).any())


@pytest.mark.parametrize("strided_gate", [False, True])
@pytest.mark.parametrize("m", M_VALUES)
@torch.inference_mode()
def test_matches_sigmoid_mul_then_quant(m, strided_gate):
    # Qwen3.8-27B full attention: 24 heads of 256.
    _check(*_inputs(m, 24, 256, seed=m, strided_gate=strided_gate))


@pytest.mark.parametrize(
    "num_heads,head_dim", [(6, 256), (2, 128), (16, 128), (32, 128)]
)
@pytest.mark.parametrize("m", [1, 33, 257, 2048])
@torch.inference_mode()
def test_other_head_layouts(m, num_heads, head_dim):
    _check(*_inputs(m, num_heads, head_dim, seed=m + num_heads, strided_gate=True))


@torch.inference_mode()
def test_edge_blocks():
    attn, gate = _inputs(33, 24, 256, seed=7, strided_gate=False)
    attn[0, :64] = 0  # all-zero blocks (amax floor)
    attn[-1, 32:64] = 1e-12  # blocks below the amax floor
    attn[-1, 64:96] *= 1e4  # large blocks
    gate[1, 0, :32] = -100.0  # sigmoid underflow
    _check(attn, gate)

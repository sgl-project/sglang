# SPDX-License-Identifier: Apache-2.0
"""Fused producers that quantize like aiter's HIP quant kernels for the gfx95 GEMMs."""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported

if not is_gfx95_supported():
    pytest.skip("the aiter fp8 quant epilogues are gfx95-only", allow_module_level=True)

from sglang.kernels.ops.quantization.aiter_fp8_quant_triton import (  # noqa: E402
    swiglu_oai_fp8,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (  # noqa: E402
    swiglu_no_interleaved_with_alpha_and_limit,
)
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gate_up(num_tokens: int) -> torch.Tensor:
    torch.manual_seed(num_tokens)
    scale = torch.exp(torch.randn(num_tokens, 1, device="cuda"))
    # beyond +-7 in places, so both clamps act
    return (torch.randn(num_tokens, 1536, device="cuda") * 3 * scale).to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 128, 1000])
@torch.inference_mode()
def test_swiglu_oai_fp8_per_token_writes_the_separate_kernels_bytes(num_tokens):
    """The fused SwiGLU-OAI + quant hands the down projection what the compiled
    activation followed by per_token_quant_hip writes: Inductor's fp32 op order,
    the bf16 rounding between them, and the scale's v_rcp_f32."""
    import aiter

    hidden = _gate_up(num_tokens)
    act = swiglu_no_interleaved_with_alpha_and_limit(hidden, 1.702, 7.0)
    q_ref, scale_ref = aiter.per_token_quant_hip(act, quant_dtype=torch.float8_e4m3fn)

    q, scale = swiglu_oai_fp8(hidden, 1.702, 7.0)

    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), scale_ref.view(torch.int32))


@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("num_tokens", [129, 1000])
@torch.inference_mode()
def test_swiglu_oai_fp8_group_writes_the_separate_kernels_bytes(
    num_tokens, transpose_scale
):
    """Per 1 x 128 the fused SwiGLU-OAI + quant writes what aiter's per_1x128 quant
    of the compiled activation does: its 1e-10 amax floor (an all-zero group) and its
    exactly rounded 1 / scale; transpose_scale stores the scale column-major."""
    import aiter

    hidden = _gate_up(num_tokens)
    hidden[1, :128] = 0
    act = swiglu_no_interleaved_with_alpha_and_limit(hidden, 1.702, 7.0)
    q_ref, scale_ref = aiter.get_hip_quant(aiter.QuantType.per_1x128)(
        act, quant_dtype=torch.float8_e4m3fn, transpose_scale=False
    )

    q, scale = swiglu_oai_fp8(
        hidden, 1.702, 7.0, group_size=128, transpose_scale=transpose_scale
    )

    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), scale_ref.view(torch.int32))
    assert scale.stride() == ((1, num_tokens) if transpose_scale else (6, 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

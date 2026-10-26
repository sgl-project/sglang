"""The indexed scale-shift MXFP8 producer against ``flashinfer.mxfp8_quantize`` of
the bf16 tensor the unfused kernel stores: payload and the swizzled E8M0 scale
buffer, padding included, byte for byte."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import indexed_scale_shift_bf16_
from sglang.kernels.ops.diffusion.modulate.indexed_scale_shift_mxfp8_triton import (
    indexed_scale_shift_mxfp8_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

ROWS, HIDDEN = 333, 5376  # rows and scale columns both need padding


def _assert_matches_flashinfer(
    got: tuple[torch.Tensor, torch.Tensor], bf16: torch.Tensor
) -> None:
    import flashinfer

    q, s = flashinfer.mxfp8_quantize(bf16.contiguous(), True)
    assert torch.equal(got[0].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(got[1].view(torch.uint8), s.view(torch.uint8))


@pytest.mark.parametrize("keep_bf16", [True, False])
def test_indexed_scale_shift_mxfp8_is_byte_exact(keep_bf16: bool) -> None:
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn((ROWS, HIDDEN), device="cuda", generator=g).to(torch.bfloat16)
    shift = torch.randn((3, HIDDEN), device="cuda", generator=g).to(torch.bfloat16)
    scale = torch.randn((3, HIDDEN), device="cuda", generator=g).to(torch.bfloat16)
    indices = torch.randint(0, 3, (ROWS,), device="cuda", generator=g)
    ref = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    kept, q, s = indexed_scale_shift_mxfp8_(
        x, shift, scale, indices, keep_bf16=keep_bf16
    )
    _assert_matches_flashinfer((q, s), ref)
    assert (kept is x and torch.equal(x, ref)) if keep_bf16 else kept is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

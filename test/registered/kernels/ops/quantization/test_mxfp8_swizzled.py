"""MXFP8 producers against ``flashinfer.mxfp8_quantize`` of the bf16 tensor the
unfused kernel stores: payload and the swizzled E8M0 scale buffer, padding
included, byte for byte."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_mxfp8_swizzled,
    can_use_silu_mul_mxfp8,
    indexed_scale_shift_bf16_,
    indexed_scale_shift_mxfp8_,
    mxfp8_quantize_swizzled,
    silu_mul_mxfp8,
)
from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import swiglu_oai_mxfp8
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    swiglu_no_interleaved_with_alpha_and_limit,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

ROWS, HIDDEN = 333, 5376  # rows and scale columns both need padding


def _assert_matches_flashinfer(
    got: tuple[torch.Tensor, torch.Tensor], bf16: torch.Tensor
) -> None:
    import flashinfer

    q, s = flashinfer.mxfp8_quantize(bf16.contiguous(), True)
    assert torch.equal(got[0].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(got[1].view(torch.uint8), s.view(torch.uint8))


def test_quantize_swizzled_is_byte_exact() -> None:
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn((ROWS, HIDDEN), device="cuda", generator=g) * 3).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block takes the minimum exponent
    assert can_use_mxfp8_swizzled(x)
    _assert_matches_flashinfer(mxfp8_quantize_swizzled(x), x)


def test_silu_mul_mxfp8_is_byte_exact() -> None:
    g = torch.Generator(device="cuda").manual_seed(2)
    x = (torch.randn((ROWS, 2 * HIDDEN), device="cuda", generator=g) * 2).to(
        torch.bfloat16
    )
    assert can_use_silu_mul_mxfp8(x)
    ref = torch.nn.functional.silu(x[:, :HIDDEN]) * x[:, HIDDEN:]
    _assert_matches_flashinfer(silu_mul_mxfp8(x), ref)


def _dirty_allocator() -> None:
    torch.full((1 << 19,), 0xFF, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("rows", [1, 7, 128, 333])
@pytest.mark.parametrize("n", [768, 1056])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 30.0])
def test_swiglu_oai_mxfp8_matches_eager_chain(rows: int, n: int, scale: float) -> None:
    g = torch.Generator(device="cuda").manual_seed(rows * 31 + n)
    x = (torch.randn((rows, 2 * n + 64), device="cuda", generator=g) * scale).to(
        torch.bfloat16
    )[:, : 2 * n]
    ref = swiglu_no_interleaved_with_alpha_and_limit(x, 1.702, 7.0)
    _dirty_allocator()
    _assert_matches_flashinfer(swiglu_oai_mxfp8(x, 1.702, 7.0), ref)


def test_swiglu_oai_mxfp8_cuda_graph_replay() -> None:
    rows, n = 40, 768
    x = torch.randn((rows, 2 * n), device="cuda").to(torch.bfloat16)
    swiglu_oai_mxfp8(x, 1.702, 7.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = swiglu_oai_mxfp8(x, 1.702, 7.0)
    for scale in [0.0, 1e-4, 1.0, 100.0]:
        x.copy_(torch.randn_like(x, dtype=torch.float32) * scale)
        snapshot = x.clone()
        graph.replay()
        assert torch.equal(x, snapshot)
        ref = swiglu_no_interleaved_with_alpha_and_limit(x, 1.702, 7.0)
        _assert_matches_flashinfer(got, ref)


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


def test_predicates_reject_unsupported_input() -> None:
    assert not can_use_mxfp8_swizzled(
        torch.randn(4, 64, device="cuda", dtype=torch.float16)
    )
    assert not can_use_silu_mul_mxfp8(
        torch.randn(4, 96, device="cuda", dtype=torch.bfloat16)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

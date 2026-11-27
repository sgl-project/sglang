"""MXFP4 (packed e2m1 + E8M0 scale) -> bf16 dequantization JIT kernel."""

import sys

import pytest
import torch

from sglang.kernels.ops.quantization.mxfp4_dequant import mxfp4_dequant
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-v for v in E2M1]
SHAPES = [(1, 32), (5, 64), (2880, 2880), (4, 5760, 2880)]


def _reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    *lead, packed_k = packed.shape
    k = packed_k * 2
    lut = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    p = packed.long()
    values = torch.stack((lut[p & 0xF], lut[p >> 4]), dim=-1).reshape(
        *lead, k // 32, 32
    )
    block_scale = (scale.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    bf16_max = torch.finfo(torch.bfloat16).max
    out = (values * block_scale).clamp(-bf16_max, bf16_max)
    out = torch.where((scale == 0xFF).unsqueeze(-1), float("nan"), out)
    return out.reshape(*lead, k).to(torch.bfloat16)


def _make_inputs(shape, seed=0):
    *lead, k = shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(
        0, 256, (*lead, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    scale = torch.randint(
        110, 135, (*lead, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    return packed, scale


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor):
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape):
    packed, scale = _make_inputs(shape)
    out = mxfp4_dequant(packed, scale)
    assert out.shape == shape and out.dtype == torch.bfloat16
    _assert_same_bits(out, _reference(packed, scale))


def test_every_code_and_edge_scales():
    # Each row repeats one scale over all 256 byte values (every e2m1 pair).
    scales = [0, 1, 2, 126, 127, 128, 200, 253, 254, 255]
    packed = (
        torch.arange(256, dtype=torch.int32, device="cuda")
        .to(torch.uint8)
        .repeat(len(scales), 1)
    )
    scale = torch.tensor(scales, dtype=torch.uint8, device="cuda")
    scale = scale.view(-1, 1).repeat(1, 512 // 32)
    out = mxfp4_dequant(packed, scale)
    _assert_same_bits(out, _reference(packed, scale))
    assert torch.isnan(out[-1]).all()  # scale byte 0xFF
    assert torch.isfinite(out[:-1]).all()  # overflow clamps instead of inf


def test_bitwise_matches_triton_kernels():
    mxfp = pytest.importorskip("triton_kernels.numerics_details.mxfp")
    for shape in SHAPES:
        packed, scale = _make_inputs(shape, seed=1)
        expected = mxfp.upcast_from_mxfp(
            packed, scale, target_dtype=torch.bfloat16, axis=-1
        )
        out = mxfp4_dequant(packed, scale)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

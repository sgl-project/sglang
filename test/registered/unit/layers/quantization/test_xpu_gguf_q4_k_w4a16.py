"""CPU reference tests for the standalone GGUF Q4_K W4A16 repacker."""

import importlib.util
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# This helper intentionally has no SGLang runtime dependency.  Load the module
# by path so this focused CPU test stays runnable in a minimal torch+gguf env.
_HELPER_PATH = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/quantization/xpu_gguf_q4_k.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "xpu_gguf_q4_k_test_helper", _HELPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)

Q4_K_BLOCK_BYTES = _HELPER.Q4_K_BLOCK_BYTES
W4A16_GROUP_SIZE = _HELPER.W4A16_GROUP_SIZE
dequantize_w4a16 = _HELPER.dequantize_w4a16
repack_q4_k_to_w4a16 = _HELPER.repack_q4_k_to_w4a16


def _set_bases(
    raw: torch.Tensor, scale: float = 0.125, minimum: float = 0.0625
) -> None:
    raw[..., :4] = torch.tensor([scale, minimum], dtype=torch.float16).view(torch.uint8)


def _finite_q4_k(rows: int = 3, blocks: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260918)
    raw = torch.randint(
        0,
        256,
        (rows, blocks, Q4_K_BLOCK_BYTES),
        dtype=torch.uint8,
        generator=generator,
    )
    _set_bases(raw)
    # Avoid zero scales in this random-reference fixture.  They are tested
    # separately because affine W4A16 cannot express every degenerate Q4_K.
    raw[..., 4:8] |= 1
    raw[..., 12:16] |= 1
    return raw.reshape(rows, -1).contiguous()


def _gguf_reference(raw: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(
        gguf.dequantize(raw.numpy(), gguf.GGMLQuantizationType.Q4_K).copy()
    )


def test_q4_k_w4a16_matches_gguf_019_reference():
    raw = _finite_q4_k()
    repacked = repack_q4_k_to_w4a16(raw)

    actual = dequantize_w4a16(repacked)
    expected = _gguf_reference(raw)

    assert repacked.qweight.dtype == torch.uint8
    assert repacked.scales.dtype == torch.float32
    assert repacked.zeros.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_q4_k_w4a16_nibble_order_and_group_size():
    raw = torch.zeros((1, 1, Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    _set_bases(raw, scale=1.0, minimum=0.0)
    raw[..., 4:8] = 1
    raw[..., 12:16] = 1  # make the remaining four scale6 values non-zero
    # Four source regions.  Each source byte encodes a distinct low/high pair
    # so the test detects both the Q4_K 32+32 order and W4A16 byte packing.
    low = torch.arange(32, dtype=torch.uint8) & 0x0F
    high = (15 - torch.arange(32, dtype=torch.uint8)) & 0x0F
    for region in range(4):
        raw[0, 0, 16 + 32 * region : 16 + 32 * (region + 1)] = (
            (high + region) & 0x0F
        ) << 4 | ((low + region) & 0x0F)

    repacked = repack_q4_k_to_w4a16(raw.reshape(1, -1))
    unpacked = torch.stack(
        (repacked.qweight & 0x0F, repacked.qweight >> 4), dim=-1
    ).reshape(-1)
    expected = torch.cat(
        tuple(
            torch.cat(((low + region) & 0x0F, (high + region) & 0x0F))
            for region in range(4)
        )
    )

    assert repacked.qweight.shape == (1, 128)
    assert repacked.scales.shape == (1, 8)
    assert W4A16_GROUP_SIZE == 32
    torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)


def test_q4_k_w4a16_is_the_stated_affine_formula():
    raw = _finite_q4_k(rows=1, blocks=1)
    repacked = repack_q4_k_to_w4a16(raw)
    q = (
        torch.stack((repacked.qweight & 0x0F, repacked.qweight >> 4), dim=-1)
        .reshape(1, -1)
        .to(torch.float32)
    )
    formula = (
        (q.reshape(1, -1, W4A16_GROUP_SIZE) - repacked.zeros.unsqueeze(-1))
        * repacked.scales.unsqueeze(-1)
    ).reshape(1, -1)

    torch.testing.assert_close(dequantize_w4a16(repacked), formula, rtol=0, atol=0)
    torch.testing.assert_close(formula, _gguf_reference(raw), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "raw, message",
    [
        (torch.zeros((1, Q4_K_BLOCK_BYTES - 1), dtype=torch.uint8), "multiple"),
        (torch.zeros((1, Q4_K_BLOCK_BYTES), dtype=torch.int8), "uint8"),
        (torch.zeros((Q4_K_BLOCK_BYTES,), dtype=torch.uint8), "rank 2"),
    ],
)
def test_q4_k_w4a16_rejects_bad_input_boundaries(raw, message):
    with pytest.raises((TypeError, ValueError), match=message):
        repack_q4_k_to_w4a16(raw)


def test_q4_k_w4a16_handles_zero_scale_zero_minimum_boundary():
    raw = torch.zeros((1, 1, Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    _set_bases(raw, scale=0.0, minimum=0.0)
    repacked = repack_q4_k_to_w4a16(raw.reshape(1, -1))
    torch.testing.assert_close(dequantize_w4a16(repacked), torch.zeros((1, 256)))


def test_q4_k_w4a16_represents_zero_scale_nonzero_minimum_boundary():
    raw = torch.zeros((1, 1, Q4_K_BLOCK_BYTES), dtype=torch.uint8)
    _set_bases(raw, scale=0.0, minimum=1.0)
    raw[..., 8:12] = 1  # first four min6 values; leave all scale6 values zero
    raw = raw.reshape(1, -1)
    repacked = repack_q4_k_to_w4a16(raw)
    torch.testing.assert_close(dequantize_w4a16(repacked), _gguf_reference(raw))


def test_reference_fixture_is_gguf_019_q4_k():
    """Keep the test's external-layout contract explicit and versioned."""
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q4_K] == (256, 144)
    assert np.dtype(np.float16).itemsize == 2

import pytest
import torch

from sglang.srt.layers.quantization.mxfp8_block_convert import (
    bf16_to_block_fp8_128,
    convert_mxfp8_weight_to_block_fp8,
    dequant_mxfp8_2d_to_bf16,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


def _mxfp8_values(shape):
    values = torch.linspace(-2, 2, torch.tensor(shape).prod().item()).reshape(shape)
    return values.to(torch.float8_e4m3fn)


def test_chunked_conversion_matches_full_conversion():
    # Ragged N and K exercise both row and column padding. A one-byte budget
    # clamps each conversion chunk to one 128-row block.
    weight = _mxfp8_values((257, 160))
    scale = torch.randint(124, 131, (257, 5), dtype=torch.uint8)

    expected_qweight, expected_scale = bf16_to_block_fp8_128(
        dequant_mxfp8_2d_to_bf16(weight, scale)
    )
    actual_qweight, actual_scale = convert_mxfp8_weight_to_block_fp8(
        weight, scale, max_workspace_bytes=1
    )

    torch.testing.assert_close(
        actual_qweight.float(), expected_qweight.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)


def test_chunked_conversion_rejects_invalid_workspace():
    weight = _mxfp8_values((2, 32))
    scale = torch.full((2, 1), 127, dtype=torch.uint8)

    with pytest.raises(ValueError, match="max_workspace_bytes must be positive"):
        convert_mxfp8_weight_to_block_fp8(
            weight, scale, max_workspace_bytes=0
        )

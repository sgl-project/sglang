import math
import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.fp8_kernel import block_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BLOCK_SHAPE = (128, 128)
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def _torch_block_quant_reference(x: torch.Tensor):
    block_rows, block_columns = _BLOCK_SHAPE
    rows, columns = x.shape
    padded_rows = math.ceil(rows / block_rows) * block_rows
    padded_columns = math.ceil(columns / block_columns) * block_columns
    padded = F.pad(x, (0, padded_columns - columns, 0, padded_rows - rows))
    blocks = padded.reshape(
        padded_rows // block_rows,
        block_rows,
        padded_columns // block_columns,
        block_columns,
    )
    scales = blocks.abs().float().amax(dim=(1, 3)).clamp(min=1e-10) / _FP8_MAX
    quantized = (
        (blocks.float() / scales[:, None, :, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(_FP8_DTYPE)
        .reshape(padded_rows, padded_columns)
    )
    return quantized[:rows, :columns].contiguous(), scales.contiguous()


def _make_exactly_quantizable_input(shape):
    """Build blocks with power-of-two scales and exactly representable values."""
    rows, columns = shape
    values = torch.tensor(
        [
            -448,
            -256,
            -128,
            -64,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            448,
        ],
        dtype=torch.float32,
    )
    x = torch.empty(shape, dtype=torch.float32)
    for block_row, row_start in enumerate(range(0, rows, _BLOCK_SHAPE[0])):
        row_end = min(row_start + _BLOCK_SHAPE[0], rows)
        for block_column, column_start in enumerate(range(0, columns, _BLOCK_SHAPE[1])):
            column_end = min(column_start + _BLOCK_SHAPE[1], columns)
            scale = 2.0 ** ((block_row * 3 + block_column) % 5 - 2)
            tile = x[row_start:row_end, column_start:column_end]
            pattern = values.repeat(math.ceil(tile.numel() / values.numel()))
            tile.copy_(pattern[: tile.numel()].reshape_as(tile) * scale)
            tile[0, 0] = _FP8_MAX * scale
    return x.to(device="cuda", dtype=torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFP8BlockQuantTail(CustomTestCase):
    def test_partial_row_and_column_blocks(self):
        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("FP8 requires compute capability 8.9 or newer")

        shapes = [
            (128, 128),  # no tail
            (129, 256),  # partial M block
            (256, 160),  # partial K block (Bailing V3 TP8)
            (257, 161),  # partial M and K blocks
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = _make_exactly_quantizable_input(shape)
                expected, expected_scales = _torch_block_quant_reference(x)

                actual, actual_scales = block_quant_fp8(x, _BLOCK_SHAPE)

                self.assertEqual(actual.shape, shape)
                self.assertEqual(actual.dtype, _FP8_DTYPE)
                self.assertEqual(
                    actual_scales.shape,
                    (
                        math.ceil(shape[0] / _BLOCK_SHAPE[0]),
                        math.ceil(shape[1] / _BLOCK_SHAPE[1]),
                    ),
                )
                self.assertEqual(actual_scales.dtype, torch.float32)
                torch.testing.assert_close(
                    actual_scales, expected_scales, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.float(), expected.float(), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()

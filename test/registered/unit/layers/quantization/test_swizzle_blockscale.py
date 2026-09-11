"""Exact NVFP4 scale layout and device-placement checks, without a model."""

import math
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _expected_bytes(shape, source_bytes):
    """Place each input byte at its interleaved offset; padding stays zero."""
    batches, rows, cols = (1, *shape) if len(shape) == 2 else shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    output = bytearray(batches * padded_rows * padded_cols)
    for batch in range(batches):
        for row in range(rows):
            for col in range(cols):
                block = (row // 128) * (padded_cols // 4) + col // 4
                offset = ((block * 32 + row % 32) * 4 + row % 128 // 32) * 4
                output[batch * padded_rows * padded_cols + offset + col % 4] = (
                    source_bytes[(batch * rows + row) * cols + col]
                )
    output_shape = (padded_rows, padded_cols)
    if len(shape) == 3:
        output_shape = (batches, *output_shape)
    return torch.tensor(list(output), dtype=torch.uint8, device="cpu").reshape(
        output_shape
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSwizzleBlockscale(CustomTestCase):
    def _check_layout(self, shape, device, noncontiguous):
        # Include every FP8 bit pattern, including NaNs; compare bytes, not values.
        source = (torch.arange(math.prod(shape), device="cpu") % 256).to(torch.uint8)
        expected = _expected_bytes(shape, source.tolist())
        raw = source.reshape(shape).to(device)
        if noncontiguous:
            backing = torch.empty(
                (*shape[:-1], shape[-1] * 2), dtype=torch.uint8, device=device
            )
            backing[..., ::2] = raw
            raw = backing[..., ::2]
            self.assertFalse(raw.is_contiguous())
        scale = raw.view(torch.float8_e4m3fn)
        allocated_devices = []
        zeros = torch.zeros

        def record_zeros(*args, **kwargs):
            result = zeros(*args, **kwargs)
            allocated_devices.append(result.device)
            return result

        # Reproduce hot reload: input can be CUDA while the default device is CPU.
        with torch.device("cpu"), patch("torch.zeros", side_effect=record_zeros):
            result = swizzle_blockscale(scale)
        self.assertEqual(allocated_devices, [scale.device])
        # Preserve the upstream input-device return contract, including CPU inputs.
        self.assertEqual(result.device, scale.device)
        self.assertEqual(result.dtype, torch.float8_e4m3fn)
        self.assertTrue(result.is_contiguous())
        torch.testing.assert_close(
            result.view(torch.uint8).cpu(), expected, rtol=0, atol=0
        )

    def test_cuda_input_layout_and_padding_stay_on_device(self):
        for shape in ((128, 4), (129, 5), (2, 256, 8), (2, 129, 5)):
            for noncontiguous in (False, True):
                with self.subTest(shape=shape, noncontiguous=noncontiguous):
                    self._check_layout(shape, "cuda", noncontiguous)

    def test_cpu_input_keeps_cpu_return_contract(self):
        for shape in ((129, 5), (2, 129, 5)):
            with self.subTest(shape=shape):
                self._check_layout(shape, "cpu", noncontiguous=True)


if __name__ == "__main__":
    unittest.main()

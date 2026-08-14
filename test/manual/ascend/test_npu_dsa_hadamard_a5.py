"""Manual Atlas A5 validation for the fused DSA Indexer quantizer.

This intentionally is not registered in the A2/A3 CI suites.  Run it on an
Atlas A5 worker until the repository has a dedicated A5 nightly runner.
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except (AttributeError, ImportError):
    _HAS_NPU = False

if _HAS_NPU:
    from sglang.srt.utils import is_npu_atlas_a5

    _IS_ATLAS_A5 = is_npu_atlas_a5()
else:
    _IS_ATLAS_A5 = False


def _hadamard_128() -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix


@unittest.skipUnless(_IS_ATLAS_A5, "requires an Atlas A5 NPU")
class TestDSAIndexerHadamardGemmQuant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
            _quantize_npu_indexer_activation,
        )

        cls.fused_quant = staticmethod(_quantize_npu_indexer_activation)
        cls.reference_hadamard = _hadamard_128()
        cls.hadamard = cls.reference_hadamard.to(device="npu", dtype=torch.bfloat16)

    def _reference_rotate(self, x):
        return (
            (
                x.cpu()
                .to(torch.float32)
                .reshape(-1, 128)
                .matmul(self.reference_hadamard)
                * (128**-0.5)
            )
            .to(torch.bfloat16)
            .reshape(x.shape)
            .to(x.device)
        )

    def _check_against_reference(self, shape):
        torch.manual_seed(3)
        x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
        actual_q, actual_scale = self.fused_quant(x, self.hadamard, torch.float8_e4m3fn)

        rotated = self._reference_rotate(x)
        expected_q, expected_scale = torch_npu.npu_dynamic_quant(
            rotated, dst_type=torch.float8_e4m3fn
        )
        expected_q = expected_q.reshape(shape)
        expected_scale = expected_scale.reshape(shape[:-1])

        self.assertEqual(actual_q.shape, x.shape)
        self.assertEqual(actual_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(actual_scale.shape, x.shape[:-1])
        torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-3, atol=1e-5)
        torch.testing.assert_close(
            (actual_q.float() * actual_scale.unsqueeze(-1)).cpu(),
            (expected_q.float() * expected_scale.unsqueeze(-1)).cpu(),
            rtol=2e-2,
            atol=5e-2,
        )

    def test_k_and_q_shapes_cover_all_block_sizes(self):
        for shape in ((1, 128), (3, 1, 128), (1, 32, 128), (3, 32, 128)):
            with self.subTest(shape=shape):
                self._check_against_reference(shape)

    def test_zero_row_and_empty_input(self):
        x = torch.zeros((4, 128), dtype=torch.bfloat16, device="npu")
        x[1, 0] = 1
        x[2, 0] = 16
        quantized, scales = self.fused_quant(x, self.hadamard, torch.float8_e4m3fn)
        self.assertEqual(scales[0].item(), 0.0)
        self.assertEqual(torch.count_nonzero(quantized[0].float()).item(), 0)
        self.assertGreater(scales[2].item(), scales[1].item())

        empty = torch.empty((0, 32, 128), dtype=torch.bfloat16, device="npu")
        empty_q, empty_scale = self.fused_quant(
            empty, self.hadamard, torch.float8_e4m3fn
        )
        self.assertEqual(empty_q.shape, empty.shape)
        self.assertEqual(empty_scale.shape, (0, 32))


if __name__ == "__main__":
    unittest.main()

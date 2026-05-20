"""Unit tests for compressed-tensors WNA16 scheme dispatch — no server, no model loading.

Covers asymmetric weight-only INT4 group/channel checkpoints (AWQ-style): the
dispatch gate must admit them, and the resulting scheme must carry the
`symmetric` flag through to the Marlin kernel path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.test.test_utils import CustomTestCase


def _weight_args(symmetric):
    return QuantizationArgs(
        num_bits=4, type="int", symmetric=symmetric, strategy="group", group_size=32
    )


class TestWNA16Dispatch(CustomTestCase):
    def setUp(self):
        # `_get_scheme_from_parts` only reads `quant_format` off the instance, so
        # build a bare config and skip the heavy __init__ (which expects a full
        # HF quantization_config dict).
        self.config = object.__new__(CompressedTensorsConfig)
        self.config.quant_format = CompressionFormat.pack_quantized.value

    def test_scheme_carries_asymmetric_flag(self):
        # The fix: an asymmetric weight-only checkpoint is admitted by the gate
        # and `symmetric=False` is forwarded to the scheme.
        scheme = self.config._get_scheme_from_parts(_weight_args(symmetric=False), None)
        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertFalse(scheme.symmetric)

    def test_scheme_carries_symmetric_flag(self):
        scheme = self.config._get_scheme_from_parts(_weight_args(symmetric=True), None)
        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertTrue(scheme.symmetric)


if __name__ == "__main__":
    unittest.main()

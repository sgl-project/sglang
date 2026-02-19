import unittest

import torch

from sglang.srt.layers.quantization import modelopt_quant


class TestCuteDslStandardInputScale(unittest.TestCase):
    def setUp(self):
        self.prev_strategy = modelopt_quant._CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY
        self.prev_logged = modelopt_quant._logged_flashinfer_cutedsl_input_scale_cast

    def tearDown(self):
        modelopt_quant._CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY = self.prev_strategy
        modelopt_quant._logged_flashinfer_cutedsl_input_scale_cast = self.prev_logged

    def test_scalar_input_passthrough(self):
        modelopt_quant._CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY = "scalar_max"
        input_scale = torch.tensor([0.125], dtype=torch.float32)
        resolved = modelopt_quant._resolve_cutedsl_standard_input_scale(input_scale)
        self.assertTrue(torch.equal(resolved, input_scale))

    def test_scalar_max_fallback_uses_max(self):
        modelopt_quant._CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY = "scalar_max"
        modelopt_quant._logged_flashinfer_cutedsl_input_scale_cast = False
        input_scale = torch.tensor([0.125, 0.5, 0.25], dtype=torch.float32)
        resolved = modelopt_quant._resolve_cutedsl_standard_input_scale(input_scale)
        self.assertEqual(tuple(resolved.shape), (1,))
        self.assertAlmostEqual(float(resolved.item()), 0.5, places=6)

    def test_strict_mode_rejects_non_scalar(self):
        modelopt_quant._CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY = "strict"
        input_scale = torch.tensor([0.125, 0.5], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "SGLANG_CUTEDSL_STANDARD_INPUT_SCALE_STRATEGY=strict"):
            modelopt_quant._resolve_cutedsl_standard_input_scale(input_scale)


if __name__ == "__main__":
    unittest.main()

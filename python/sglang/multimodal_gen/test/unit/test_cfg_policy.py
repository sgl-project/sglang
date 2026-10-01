import unittest
from unittest.mock import MagicMock

import torch

from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    CFGPolicy,
    _apply_cfg_normalization,
)


class TestCFGPolicyCombine(unittest.TestCase):
    def test_cfg_parallel_uses_parallel_arithmetic_order(self):
        policy = CFGPolicy()
        req = MagicMock()
        req.cfg_normalization = 0
        req.guidance_rescale = 0

        pipeline_config = MagicMock()
        pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise

        pos = torch.tensor([1.0], dtype=torch.bfloat16)
        neg = torch.tensor([0.1], dtype=torch.bfloat16)

        serial = policy.combine([pos, neg], req, 7.0, pipeline_config)
        parallel = policy.combine(
            [pos, neg], req, 7.0, pipeline_config, cfg_parallel=True
        )

        self.assertTrue(torch.equal(serial, neg + 7.0 * (pos - neg)))
        self.assertTrue(torch.equal(parallel, 7.0 * pos + (1 - 7.0) * neg))
        self.assertFalse(torch.equal(serial, parallel))


class TestCFGNormalizationDtype(unittest.TestCase):
    def test_normalization_preserves_prediction_dtype(self):
        """Per-sample fp32 scale must not promote a bf16/fp16 prediction to fp32."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for pred_value, cond_value in ((0.25, 1.0), (1.0, 0.25)):
                with self.subTest(dtype=dtype, pred=pred_value):
                    pred = torch.full((2, 2, 3), pred_value, dtype=dtype)
                    cond = torch.full((2, 2, 3), cond_value, dtype=dtype)
                    out = _apply_cfg_normalization(pred, cond, 0.7)
                    self.assertEqual(out.dtype, dtype)


if __name__ == "__main__":
    unittest.main()

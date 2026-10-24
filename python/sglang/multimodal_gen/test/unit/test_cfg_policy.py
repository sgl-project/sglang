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


class TestCFGNormalizationPerSample(unittest.TestCase):
    """CFG normalization clamps each sample by its own norm, not the batch's.

    A batch-flattened norm makes one request's output depend on whichever
    requests happen to be co-scheduled with it in the same DiT forward pass.
    """

    CFG_NORMALIZATION = 0.7

    @staticmethod
    def _batch(seed):
        generator = torch.Generator().manual_seed(seed)
        pred = torch.randn(4, 3, 8, 8, generator=generator)
        cond = torch.randn(4, 3, 8, 8, generator=generator) * 0.3
        return pred, cond

    @staticmethod
    def _per_sample_norm(tensor):
        dims = list(range(1, tensor.ndim))
        return torch.linalg.vector_norm(tensor.float(), dim=dims)

    def test_sample_result_is_independent_of_batch_composition(self):
        pred, cond = self._batch(seed=0)
        pred[1] *= 50.0

        co_batched = _apply_cfg_normalization(pred, cond, self.CFG_NORMALIZATION)
        solo = _apply_cfg_normalization(pred[:1], cond[:1], self.CFG_NORMALIZATION)

        self.assertTrue(torch.equal(solo, co_batched[:1]))

    def test_every_sample_respects_its_own_threshold(self):
        pred, cond = self._batch(seed=1)
        pred[0] *= 100.0

        out = _apply_cfg_normalization(pred, cond, self.CFG_NORMALIZATION)

        ratio = self._per_sample_norm(out) / self._per_sample_norm(cond)
        self.assertTrue(
            torch.all(ratio <= self.CFG_NORMALIZATION + 1e-5),
            f"per-sample norm ratios exceed threshold: {ratio.tolist()}",
        )

    def test_batch_size_one_matches_global_norm_reference(self):
        """At batch size 1 the per-sample norm must stay bit-identical to upstream."""
        pred, cond = self._batch(seed=2)
        pred, cond = pred[:1], cond[:1]

        out = _apply_cfg_normalization(pred, cond, self.CFG_NORMALIZATION)

        ori_norm = torch.linalg.vector_norm(cond.float())
        new_norm = torch.linalg.vector_norm(pred.float())
        max_norm = ori_norm * self.CFG_NORMALIZATION
        expected = pred * (max_norm / new_norm) if new_norm > max_norm else pred

        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()

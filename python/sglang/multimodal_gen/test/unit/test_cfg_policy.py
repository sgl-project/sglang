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


class TestCFGNormalizationUnderSequenceParallel(unittest.TestCase):
    """Under SP each rank combines only its latent shard; the norm must be the whole latent's."""

    @staticmethod
    def _request(sp_sharded):
        req = MagicMock()
        req.cfg_normalization = True
        req.guidance_rescale = 0
        req.did_sp_shard_latents = sp_sharded
        return req

    @staticmethod
    def _pipeline_config(gather):
        pipeline_config = MagicMock()
        pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise
        pipeline_config.gather_latents_for_sp.side_effect = gather
        return pipeline_config

    def test_sp_shards_match_single_rank_bitwise(self):
        from unittest.mock import patch

        from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
            ZImageCFGPolicy,
        )

        generator = torch.Generator().manual_seed(0)
        # [B, C, F, H, W] as in Z-Image, sharded along H over 2 ranks
        pos = torch.randn(1, 16, 1, 8, 8, generator=generator)
        neg = torch.randn(1, 16, 1, 8, 8, generator=generator)
        neg[..., :4, :] *= 0.1
        policy = ZImageCFGPolicy()

        single = policy.combine(
            [pos, neg], self._request(False), 5.0, self._pipeline_config(None)
        )

        full_pred = pos.float() + 5.0 * (pos.float() - neg.float())
        shards = []
        with patch(
            "sglang.multimodal_gen.runtime.distributed.parallel_state.get_sp_world_size",
            return_value=2,
        ):
            for rank in range(2):
                rows = slice(4 * rank, 4 * rank + 4)
                pos_r, neg_r = pos[..., rows, :], neg[..., rows, :]
                pred_r = pos_r.float() + 5.0 * (pos_r.float() - neg_r.float())

                def gather(tensor, batch, pred_r=pred_r):
                    return full_pred if torch.equal(tensor, pred_r) else pos.float()

                shards.append(
                    policy.combine(
                        [pos_r, neg_r],
                        self._request(True),
                        5.0,
                        self._pipeline_config(gather),
                    )
                )

        self.assertTrue(torch.equal(torch.cat(shards, dim=3), single))

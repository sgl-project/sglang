import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.swa_recompute import swa_recompute_batch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAX_CTX = 2048


class TestSwaRecomputeBatch(CustomTestCase):
    def _batch(self):
        req_to_token = torch.arange(3 * MAX_CTX, dtype=torch.int32).view(3, MAX_CTX)
        runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            model=SimpleNamespace(model=SimpleNamespace(engram_hasher=None)),
        )
        reqs = [
            SimpleNamespace(full_untruncated_fill_ids=list(range(n)))
            for n in (16, 1040, 1552)
        ]
        batch = SimpleNamespace(
            reqs=reqs,
            req_pool_indices=torch.tensor([0, 1, 2]),
            req_pool_indices_cpu=torch.tensor([0, 1, 2]),
            prefix_lens=[0, 1024, 1536],
            extend_lens=[16, 16, 16],
            extend_num_tokens=48,
            global_num_tokens=[48],
            global_num_tokens_for_logprob=[3],
            swa_recompute_starts=[None, 512, 1280],
            swa_recompute_replay=None,
        )
        return batch, runner

    def test_one_replay_batch_syncs_its_own_rows(self):
        """All replays of a step run as one batch whose MLP-sync count is its own
        rows, not the scheduled extend's; the draft reuses the same batch."""
        batch, runner = self._batch()

        replay = swa_recompute_batch(batch, runner)

        self.assertEqual(replay.prefix_lens, [512, 1280])
        self.assertEqual(replay.extend_lens, [512, 256])
        self.assertEqual(replay.global_num_tokens, [768])
        self.assertEqual(replay.out_cache_loc.shape[0], 768)
        self.assertIsNone(replay.swa_recompute_starts)
        self.assertIs(swa_recompute_batch(batch, runner), replay)


if __name__ == "__main__":
    unittest.main()

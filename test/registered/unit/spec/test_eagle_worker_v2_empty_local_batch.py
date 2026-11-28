"""A zero-token attention-DP rank must not enter the speculative prefill path."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TestEagleWorkerV2EmptyLocalBatch(CustomTestCase):
    def _make_worker(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker.device = DEVICE
        worker.topk = 1
        worker.speculative_num_draft_tokens = 4
        worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: True)
        worker._draft_worker = None
        worker._wait_for_fused_hicache_load = MagicMock()
        return worker

    def _make_batch(self, input_ids_none=False):
        return SimpleNamespace(
            input_ids=(
                None
                if input_ids_none
                else torch.empty((0,), dtype=torch.int64, device=DEVICE)
            ),
            seq_lens=torch.empty((0,), dtype=torch.int32, device=DEVICE),
            forward_mode=ForwardMode.EXTEND,
            is_extend_in_batch=False,
        )

    def test_empty_extend_batch_returns_empty_result(self):
        worker = self._make_worker()
        batch = self._make_batch()
        published = []
        result = worker.forward_batch_generation(
            batch, on_publish=published.append, pp_proxy_tensors=None
        )

        worker._wait_for_fused_hicache_load.assert_called_once_with(batch)
        self.assertIsNone(result.logits_output)
        self.assertEqual(result.next_token_ids.numel(), 0)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertEqual(len(published), 1)
        self.assertIs(published[0], batch.seq_lens)
        self.assertIsNotNone(batch.spec_info)

    def test_empty_seq_lens_without_input_ids(self):
        worker = self._make_worker()
        batch = self._make_batch(input_ids_none=True)
        result = worker.forward_batch_generation(batch)

        self.assertIsNone(result.logits_output)
        self.assertEqual(result.next_token_ids.numel(), 0)


if __name__ == "__main__":
    unittest.main()

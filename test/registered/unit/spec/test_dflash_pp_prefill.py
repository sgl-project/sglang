"""DFLASH on a pipelined PD prefill.

An earlier pipeline stage holds no draft: it runs the target forward with aux
hidden capture (riding the PP proxy to the last stage) and skips every draft
step.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _earlier_stage_worker():
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker._target_worker = Mock()
    worker.model_runner = worker._target_worker.model_runner
    worker._hosts_draft = False
    worker._draft_worker = None
    worker.draft_model_runner = None
    worker.draft_model = None
    return worker


class TestDFlashEarlierPipelineStage(CustomTestCase):
    def test_forward_runs_the_target_with_aux_capture_only(self):
        worker = _earlier_stage_worker()
        target_output = SimpleNamespace(new_seq_lens=None)
        worker._target_worker.forward_batch_generation.return_value = target_output
        batch = SimpleNamespace(seq_lens=object())
        proxy = object()
        published = []

        out = worker.forward_batch_generation(
            batch, on_publish=published.append, pp_proxy_tensors=proxy
        )

        worker._target_worker.forward_batch_generation.assert_called_once_with(
            batch,
            pp_proxy_tensors=proxy,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        self.assertIs(out, target_output)
        self.assertIs(out.new_seq_lens, batch.seq_lens)
        self.assertEqual(published, [batch.seq_lens])

    def test_draft_setup_is_skipped(self):
        worker = _earlier_stage_worker()
        self.assertIsNone(worker.alloc_memory_pool())
        self.assertIsNone(worker.init_attention_backends())
        self.assertIsNone(worker.init_cuda_graphs())
        self.assertEqual(worker.weight_update_runners(), [])
        self.assertEqual(
            worker.spec_v2_attn_backends,
            (worker._target_worker.model_runner.attn_backend,),
        )
        worker.prewarm_sampling()
        worker.model_runner.prewarm_sampling.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

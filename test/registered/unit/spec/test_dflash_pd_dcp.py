"""DFLASH on a pipelined PD prefill and the MLA-only draft backends.

- An earlier pipeline stage holds no draft: it runs the target forward with
  aux hidden capture (riding the PP proxy to the last stage) and skips every
  draft step.
- cute-dsl / TRT-LLM MLA serve only MLA layers, so a draft worker takes them
  only for DFLASH, whose hook admits them for MLA drafts.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative import draft_worker_common
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


class TestMlaOnlyDraftBackends(CustomTestCase):
    def _resolve(self, backend, algo_label):
        with (
            patch.object(
                draft_worker_common,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_attention_backend=backend
                ),
            ),
            patch.object(
                draft_worker_common,
                "get_platform",
                return_value=SimpleNamespace(is_xpu=False),
            ),
        ):
            return draft_worker_common.resolve_draft_worker_attention_backend(
                algo_label=algo_label
            )

    def test_dflash_keeps_mla_only_backends(self):
        for backend in draft_worker_common.DFLASH_MLA_ONLY_DRAFT_BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(self._resolve(backend, "DFLASH"), backend)

    def test_other_algorithms_fall_back(self):
        for algo_label in ("DSPARK", None):
            with self.subTest(algo_label=algo_label):
                self.assertEqual(self._resolve("cutedsl_mla", algo_label), "flashinfer")


if __name__ == "__main__":
    unittest.main()

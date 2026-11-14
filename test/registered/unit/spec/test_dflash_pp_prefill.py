"""DFLASH on a pipelined PD prefill.

Only the last pipeline stage builds the draft. An earlier stage holds no draft:
it runs the target forward with aux hidden capture (riding the PP proxy to the
last stage) and skips every draft step.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.distributed import parallel_state
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DraftBuilt(Exception):
    """Cuts the last stage's construction off once it builds its draft."""


class TestDFlashPipelineStage(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.tp_group = SimpleNamespace(world_size=1, rank_in_group=0)
        self.target = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_aux_config=SimpleNamespace(draft_model_config=None),
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                model_config=object(),
                attn_backend=object(),
                prewarm_sampling=Mock(),
            ),
            device="cpu",
            random_seed=7,
            forward_batch_generation=Mock(),
        )
        self.draft_builds = []

    def _build_draft(self, **kwargs):
        # The draft never spans stages: it builds in a single-member pp scope.
        self.draft_builds.append((kwargs, get_parallel().pp_size))
        raise _DraftBuilt

    def _construct(self, *, pp_rank):
        pp_size = 2
        pp_group = SimpleNamespace(
            world_size=pp_size,
            rank_in_group=pp_rank,
            is_first_rank=pp_rank == 0,
            is_last_rank=pp_rank == pp_size - 1,
        )
        self_pp_group = SimpleNamespace(
            world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
        )
        with (
            get_parallel().override(
                pp_size=pp_size,
                pp_rank=pp_rank,
                pp_group=pp_group,
                tp_rank=0,
                tp_group=self.tp_group,
            ),
            patch.object(parallel_state, "_SELF_PP", self_pp_group),
            patch(
                "sglang.srt.speculative.dflash_worker_v2.build_draft_tp_worker",
                side_effect=self._build_draft,
            ),
        ):
            return DFlashWorkerV2(None, 0, 0, self.target)

    def test_earlier_stage_builds_no_draft(self):
        worker = self._construct(pp_rank=0)

        self.assertEqual(self.draft_builds, [])
        self.assertIsNone(worker.draft_model_runner)
        self.assertIsNone(worker.draft_model)
        self.assertEqual(worker.weight_update_runners(), [])

    def test_last_stage_builds_the_draft_within_its_stage(self):
        with self.assertRaises(_DraftBuilt):
            self._construct(pp_rank=1)

        [(kwargs, draft_pp_size)] = self.draft_builds
        self.assertEqual(draft_pp_size, 1)
        self.assertEqual(kwargs["algo_label"], "DFLASH")
        self.assertEqual(kwargs["random_seed"], self.target.random_seed)
        # Loads synchronize within the stage, not across every rank.
        self.assertIs(kwargs["load_group"], self.tp_group)

    def test_earlier_stage_forward_runs_the_target_with_aux_capture_only(self):
        worker = self._construct(pp_rank=0)
        target_output = SimpleNamespace(new_seq_lens=None)
        self.target.forward_batch_generation.return_value = target_output
        batch = SimpleNamespace(seq_lens=object())
        proxy = object()
        published = []

        out = worker.forward_batch_generation(
            batch, on_publish=published.append, pp_proxy_tensors=proxy
        )

        self.target.forward_batch_generation.assert_called_once_with(
            batch,
            pp_proxy_tensors=proxy,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        self.assertIs(out, target_output)
        self.assertIs(out.new_seq_lens, batch.seq_lens)
        self.assertEqual(published, [batch.seq_lens])

    def test_earlier_stage_skips_draft_setup(self):
        worker = self._construct(pp_rank=0)
        self.assertIsNone(worker.alloc_memory_pool())
        self.assertIsNone(worker.init_attention_backends())
        self.assertIsNone(worker.init_cuda_graphs())
        self.assertEqual(
            worker.spec_v2_attn_backends,
            (self.target.model_runner.attn_backend,),
        )
        worker.prewarm_sampling()
        self.target.model_runner.prewarm_sampling.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

"""Target capture and draft execution must remain separate at the B1 boundary."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.dflash_cohort import validate_target_verify_graph_mode
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TargetVerifyGraphTests(CustomTestCase):
    def test_only_cuda_cohort_two_or_four_enables_target_graph_for_independent_kv(self):
        for layout in ["independent", "hybrid"]:
            for size in [2, 4]:
                validate_target_verify_graph_mode(
                    draft_kv_layout=layout,
                    cohort_size=size,
                    graph_enabled=True,
                    cuda=True,
                )
            for size, cuda in [(1, True), (8, True), (2, False), (4, False)]:
                with self.assertRaisesRegex(ValueError, "cohort of size 2 or 4"):
                    validate_target_verify_graph_mode(
                        draft_kv_layout=layout,
                        cohort_size=size,
                        graph_enabled=True,
                        cuda=cuda,
                    )
        validate_target_verify_graph_mode(
            draft_kv_layout="hybrid", cohort_size=1, graph_enabled=False, cuda=True
        )
        validate_target_verify_graph_mode(
            draft_kv_layout="shared", cohort_size=1, graph_enabled=True, cuda=True
        )

    def test_cohort_initialization_never_captures_the_draft(self):
        for size in [2, 4]:
            worker = SimpleNamespace(
                _hosts_draft=True,
                _cohort=SimpleNamespace(owner_count=size),
                _prefill_context_role=False,
                draft_parallel_layout=object(),
                _draft_worker=SimpleNamespace(init_cuda_graphs=Mock()),
            )
            with (
                patch(
                    "sglang.srt.speculative.dflash_worker_v2.get_spec",
                    return_value=SimpleNamespace(
                        speculative_draft_cohort_cuda_graph=False
                    ),
                ),
                patch(
                    "sglang.srt.speculative.dflash_worker_v2.draft_pp_context",
                    return_value=nullcontext(),
                ),
                patch(
                    "sglang.srt.speculative.dflash_worker_v2.draft_tp_context",
                    return_value=nullcontext(),
                ),
            ):
                DFlashWorkerV2.init_cuda_graphs(worker)
            worker._draft_worker.init_cuda_graphs.assert_called_once_with(
                capture_decode_cuda_graph=False
            )

    def test_b1_active_and_idle_cannot_replay_a_b12_target_capture(self):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.model_runner = SimpleNamespace(model=object())
        runner.ragged_verify_mode = False
        runner.captured_req_width = 12
        for active in [True, False]:
            batch = SimpleNamespace(
                replace_embeds=None,
                spec_info=SimpleNamespace(num_tokens_per_req=1),
                can_run_decode_cuda_graph=False,
                batch_size=1 if active else 0,
            )
            with patch(
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner.megakernel_can_run_graph",
                return_value=True,
            ):
                self.assertFalse(runner.can_run_graph(batch))

    def test_replay_logging_distinguishes_active_and_idle_once(self):
        worker = SimpleNamespace(_cohort=SimpleNamespace(owner_count=4), gpu_id=0)
        with patch("sglang.srt.speculative.dflash_worker_v2.logger.info") as log:
            DFlashWorkerV2._log_cohort_target_verify_graph(worker, False, idle=False)
            self.assertEqual(log.call_count, 0)
            for _ in range(2):
                DFlashWorkerV2._log_cohort_target_verify_graph(worker, True, idle=False)
                DFlashWorkerV2._log_cohort_target_verify_graph(worker, True, idle=True)
            self.assertEqual(log.call_count, 2)


if __name__ == "__main__":
    unittest.main()

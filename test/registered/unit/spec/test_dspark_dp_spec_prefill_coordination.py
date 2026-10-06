import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

MODULE = "sglang.srt.speculative.dspark_components.dspark_worker_v2"


class TestDSparkDPSpecPrefillCoordination(CustomTestCase):
    def test_draft_counts_are_not_scaled_twice(self):
        for width in (6, 7):
            for rank, local_tokens in ((0, 0), (1, 3 * width), (2, 0)):
                with self.subTest(width=width, rank=rank):
                    plan = DPSpecPrefillCoordinationPlan(
                        [8192, 3, 0], [1, 3, 0], torch.tensor([1, 0, 0]), width, 7
                    )
                    batch = SimpleNamespace()
                    plan.apply(batch, "draft", rank, local_only=False)
                    proposer = object.__new__(DraftBlockProposer)
                    proposer._dp_moe_sync = True
                    proposer._num_token_non_padded = None
                    proposer.draft_model_runner = SimpleNamespace(device="cpu")
                    proposer._draft_block_spec_info = SimpleNamespace(
                        num_tokens_per_req=width, num_tokens_for_logprob_per_req=1
                    )
                    forward = SimpleNamespace(input_ids=torch.arange(local_tokens))
                    with patch(
                        "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
                        return_value=True,
                    ):
                        proposer._fill_dp_moe_sync_metadata(forward, batch)
                    self.assertEqual(forward.global_num_tokens_cpu, [0, 3 * width, 0])
                    self.assertEqual(
                        forward.global_num_tokens_for_logprob_cpu, [0, 3 * width, 0]
                    )
                    self.assertTrue(forward.dp_spec_prefill_coordination_applied)
                    self.assertTrue(forward.is_extend_in_batch)
                    self.assertFalse(forward.can_run_decode_cuda_graph)
                    plan.apply(batch, "target", rank, local_only=False)
                    self.assertEqual(batch.global_num_tokens, [8192, 21, 0])
                    self.assertEqual(batch.global_num_tokens_for_logprob, [1, 21, 0])

    def test_prefill_joins_draft_before_target_with_local_counts_preserved(self):
        plan = DPSpecPrefillCoordinationPlan(
            [8192, 3, 0], [1, 3, 0], torch.tensor([1, 0, 0]), 6, 7
        )
        for local_draft in (False, True):
            with self.subTest(local_draft=local_draft):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    global_num_tokens=plan.counts,
                )
                events = []
                worker = object.__new__(DSparkWorkerV2)
                worker._draft_dp_context_enabled = local_draft
                worker._draft_context = nullcontext
                worker._verify_planner = MagicMock()
                worker._observers = MagicMock()

                def draft(current):
                    self.assertIs(current, batch)
                    self.assertEqual(
                        current.global_num_tokens, [0] if local_draft else [0, 18, 0]
                    )
                    self.assertFalse(current.can_run_decode_cuda_graph)
                    events.append("draft")

                def target(current, publish, proxy):
                    self.assertEqual(current.global_num_tokens, [8192, 21, 0])
                    self.assertEqual(current.forward_mode, ForwardMode.EXTEND)
                    self.assertFalse(current.can_run_decode_cuda_graph)
                    events.append("target")
                    return "result"

                worker._proposer = SimpleNamespace(run_idle_participation=draft)
                worker._forward_prefill = target
                with patch(
                    f"{MODULE}.get_parallel",
                    return_value=SimpleNamespace(attn_dp_rank=0),
                ):
                    result = worker._forward_dp_spec_prefill_coordination(
                        batch, plan, None, None, None
                    )
                self.assertEqual(result, "result")
                self.assertEqual(events, ["draft", "target"])

    def test_dispatch_keeps_uniform_and_disabled_paths(self):
        cases = [
            (False, ForwardMode.EXTEND, True, [8192, 3], [1, 0], "prefill"),
            (True, ForwardMode.EXTEND, True, [8192, 0], [1, 0], "prefill"),
            (True, ForwardMode.DECODE, False, [3, 2], [0, 0], "decode"),
            (True, ForwardMode.DECODE, True, [8192, 3], [1, 0], "coordinate"),
            (True, ForwardMode.IDLE, True, [8192, 3, 0], [1, 0, 0], "coordinate"),
        ]
        for enabled, mode, extend, counts, prefills, expected in cases:
            with self.subTest(enabled=enabled, mode=mode, expected=expected):
                worker = object.__new__(DSparkWorkerV2)
                worker._hosts_draft = True
                worker.enable_dp_spec_prefill_coordination = enabled
                worker.verify_num_draft_tokens = 7
                worker._proposer = SimpleNamespace(query_token_num=6)
                worker._verify_planner = MagicMock()
                worker._observers = MagicMock()
                worker._forward_prefill = MagicMock(return_value="prefill")
                worker._forward_decode = MagicMock(return_value="decode")
                worker._forward_dp_spec_prefill_coordination = MagicMock(
                    return_value="coordinate"
                )
                batch = SimpleNamespace(
                    forward_mode=mode,
                    is_extend_in_batch=extend,
                    dp_spec_prefill_coordination_metadata=(
                        counts,
                        counts,
                        torch.tensor(prefills),
                    ),
                )
                self.assertEqual(worker.forward_batch_generation(batch), expected)
                if expected == "coordinate":
                    plan = worker._forward_dp_spec_prefill_coordination.call_args.args[
                        1
                    ]
                    self.assertEqual(plan.draft_width, 6)
                    self.assertEqual(plan.verify_width, 7)


if __name__ == "__main__":
    unittest.main()

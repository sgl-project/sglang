import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    _slice_draft_output_to_local_tokens,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestEaglePDDPFallback(CustomTestCase):
    def test_draft_graph_gate_has_independent_dp_vote(self):
        sync_info = MLPSyncBatchInfo(
            dp_size=1,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            can_cuda_graph=True,
            can_draft_cuda_graph=False,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
            can_run_breakable_cuda_graph=False,
        )

        local = sync_info._get_local_tensor(device="cpu")
        fallback = sync_info._get_fallback_tensor(device="cpu")
        self.assertEqual(local[2].item(), 1)
        self.assertEqual(local[7].item(), 0)
        # Idle/inactive DP ranks remain permissive for the draft vote.
        self.assertEqual(fallback[7].item(), 1)

    def test_seedless_gate_only_disables_draft_graph(self):
        runner = object.__new__(EAGLEDraftCudaGraphRunner)
        runner.require_mlp_tp_gather = False
        runner.require_mlp_sync = True
        runner.disable_padding = False
        runner.captured_req_width = 1
        runner.max_bs = 8

        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=1),
            batch_size=1,
            can_run_dp_cuda_graph=True,
            can_run_dp_draft_cuda_graph=False,
        )
        self.assertFalse(runner.can_run_graph(forward_batch))

        # The ordinary DP graph gate remains enabled for target verify and
        # draft-extend; only the draft runner consumes the extra gate.
        forward_batch.can_run_dp_draft_cuda_graph = True
        self.assertTrue(runner.can_run_graph(forward_batch))

    def test_seedless_pd_draft_requests_rank_consistent_eager_forward(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True)

        for seed, future_indices, future_seed, expect_eager in (
            (None, None, False, True),
            (torch.ones((1, 1)), None, False, False),
            (torch.ones((1, 1)), torch.tensor([1]), False, True),
            (None, torch.tensor([1]), True, False),
        ):
            with self.subTest(
                seed_present=seed is not None,
                overlap=future_indices is not None,
                future_seed=future_seed,
            ):
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace(
                        dsa_topk_indices=seed,
                        future_indices=future_indices,
                        future_dsa_topk_indices_available=future_seed,
                    )
                )
                self.assertEqual(
                    worker.requires_dp_attention_eager_forward(batch),
                    expect_eager,
                )

        worker._draft_worker.seed_dsa_topk_from_draft_extend = False
        self.assertFalse(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=SimpleNamespace(dsa_topk_indices=None))
            )
        )

    def test_seeded_running_batch_merged_with_seedless_prebuilt_forces_eager(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(seed_dsa_topk_from_draft_extend=True)

        running_input = EagleDraftInput(
            dsa_topk_indices=torch.ones((1, 2), dtype=torch.int64),
            future_indices=torch.tensor([1]),
            future_dsa_topk_indices_available=True,
        )
        prebuilt_input = EagleDraftInput(
            dsa_topk_indices=None,
            future_indices=torch.tensor([2]),
            future_dsa_topk_indices_available=False,
        )
        running_input.merge_batch(prebuilt_input)

        self.assertFalse(running_input.future_dsa_topk_indices_available)
        # merge_batch intentionally leaves the currently materialized tensor
        # untouched; FutureMap will clear it immediately before the forward.
        self.assertIsNotNone(running_input.dsa_topk_indices)
        self.assertTrue(
            worker.requires_dp_attention_eager_forward(
                SimpleNamespace(spec_info=running_input)
            )
        )

    def test_eager_draft_discards_dp_padding_rows(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        positions = torch.tensor([7, 100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits,
                hidden_states,
                positions,
                num_local_tokens=1,
            )
        )

        self.assertEqual(local_logits.shape, (1, 8))
        self.assertEqual(local_hidden_states.shape, (1, 4))
        self.assertEqual(local_positions.tolist(), [7])
        local_positions.add_(1)
        self.assertEqual(positions.tolist(), [8, 100, 100])

    def test_idle_eager_draft_discards_all_dp_padding_rows(self):
        logits = torch.empty((2, 8))
        hidden_states = torch.empty((2, 4))
        positions = torch.tensor([100, 100])

        local_logits, local_hidden_states, local_positions = (
            _slice_draft_output_to_local_tokens(
                logits,
                hidden_states,
                positions,
                num_local_tokens=0,
            )
        )

        self.assertEqual(local_logits.shape, (0, 8))
        self.assertEqual(local_hidden_states.shape, (0, 4))
        self.assertEqual(local_positions.shape, (0,))

    def test_eager_draft_rejects_missing_local_rows(self):
        with self.assertRaisesRegex(RuntimeError, "next_token_logits has 0 rows"):
            _slice_draft_output_to_local_tokens(
                torch.empty((0, 8)),
                torch.empty((1, 4)),
                torch.tensor([7]),
                num_local_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()

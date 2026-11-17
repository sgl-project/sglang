"""Unit tests for deterministic EAGLE draft-proposal stream selection."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleDraftSamplingPosition(CustomTestCase):
    def test_draft_extend_uses_post_verify_position(self):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.speculative_num_draft_tokens = 4
        worker.cuda_graph_runner_for_draft_extend = None
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        worker.seed_dsa_topk_from_draft_extend = False
        worker.topk = 1
        worker.use_fused_draft_sample = False
        worker.device = torch.device("cpu")

        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros(8, 16), hidden_states=None
        )
        worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=lambda _: SimpleNamespace(logits_output=logits_output),
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(2, 1),
            top_ks=torch.ones(2, dtype=torch.int32),
            sampling_seed=torch.tensor([11, 22], dtype=torch.int64),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            sampling_info=sampling_info,
        )
        batch_result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=torch.tensor([1, 2], dtype=torch.int64),
            accept_lens=torch.tensor([2, 4], dtype=torch.int64),
            new_seq_lens=torch.tensor([12, 24], dtype=torch.int64),
            next_draft_input=SimpleNamespace(),
            prepared_draft_extend_inputs=None,
        )
        forward_batch = SimpleNamespace(sampling_info=sampling_info)
        sample_result = (
            torch.zeros(2, 16),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
        )
        resolved_spec = SimpleNamespace(speculative_use_rejection_sampling=True)

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                return_value=forward_batch,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.sample_draft_proposal",
                return_value=sample_result,
            ) as sample,
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=resolved_spec,
            ),
        ):
            worker._draft_extend_for_decode(batch, batch_result)

        sample.assert_called_once()
        torch.testing.assert_close(
            sample.call_args.kwargs["positions"], batch_result.new_seq_lens
        )

    @staticmethod
    def _run(out, runner=None, **patches):
        """One topk-1 draft extend of two requests; returns the next draft input."""
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.speculative_num_draft_tokens = 4
        worker.cuda_graph_runner_for_draft_extend = runner
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        worker.seed_dsa_topk_from_draft_extend = False
        worker.topk = 1
        worker.device = torch.device("cpu")
        # a fresh output per forward: the extend reassigns its fields
        forward = lambda _: SimpleNamespace(logits_output=SimpleNamespace(**vars(out)))
        worker.draft_runner = SimpleNamespace(canary_manager=None, forward=forward)
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20]), sampling_info=SimpleNamespace()
        )
        batch_result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=torch.tensor([1, 2]),
            accept_lens=torch.tensor([1, 1]),
            new_seq_lens=torch.tensor([12, 24]),
            next_draft_input=SimpleNamespace(),
            prepared_draft_extend_inputs=None,
        )
        v2 = "sglang.srt.speculative.eagle_worker_v2."
        spec = SimpleNamespace(speculative_use_rejection_sampling=False)
        with (
            patch(v2 + "prepare_for_draft_extend", return_value=SimpleNamespace()),
            patch(v2 + "get_spec", return_value=spec),
            patch(v2 + patches["name"], **patches["kw"]) as mock,
        ):
            worker._draft_extend_for_decode(batch, batch_result)
        return batch_result.next_draft_input, mock

    def test_draft_extend_top1_takes_the_sampler_argmax_for_large_logits(self):
        logits = torch.randn(8, 16)
        logits[:, 5] = logits[:, 9] = logits.max() + 1  # tie: lowest index
        out = SimpleNamespace(next_token_logits=logits, hidden_states=None)
        argmax = dict(side_effect=lambda x: torch.argmax(x, dim=-1))
        selected = 2 * logits.shape[1]  # one row per request
        for min_logits, called in ((selected + 1, 0), (selected, 1)):
            with (
                self.subTest(min_logits=min_logits),
                patch(
                    "sglang.srt.speculative.eagle_worker_v2."
                    "_DRAFT_TOP1_SPLIT_ARGMAX_MIN_LOGITS",
                    min_logits,
                ),
            ):
                draft, sampler = self._run(out, name="sampling_argmax", kw=argmax)
                self.assertEqual(sampler.call_count, called)
                self.assertTrue(torch.equal(draft.topk_index, torch.full((2, 1), 5)))


if __name__ == "__main__":
    unittest.main()

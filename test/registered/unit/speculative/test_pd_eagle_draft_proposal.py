"""A PD decode's first EAGLE draft input carries a rejection-sampling proposal.

PD prefill sends the first draft token, its probability and hidden state, but
not the distribution the token was drawn from. With rejection sampling on
(the ROCm default for EAGLE top-k 1), the decode's first draft step stacks
that missing proposal with the steps it drafts itself.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative import eagle_disaggregation
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
    pd_first_draft_proposal,
    snapshot_retracted_eagle_input,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

VOCAB = 11


def _spec(rejection_sampling=True, multi_layer=False):
    return SimpleNamespace(
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        enable_multi_layer_eagle=multi_layer,
        speculative_use_rejection_sampling=rejection_sampling,
    )


def _batch(tokens):
    # One entry per request: a token, or the token chain of multi-layer EAGLE.
    chains = [list(t) if isinstance(t, tuple) else [t] for t in tokens]
    reqs = [
        SimpleNamespace(
            output_topk_p=[0.25] * len(chain),
            output_topk_index=chain,
            hidden_states_tensor=torch.full((4,), float(i)),
            output_dsa_topk_indices=None,
        )
        for i, chain in enumerate(chains)
    ]
    return SimpleNamespace(
        reqs=reqs,
        device="cpu",
        enable_overlap=False,
        # A model without DSA seed metadata: no MTP index sharing.
        model_config=SimpleNamespace(vocab_size=VOCAB, hf_config=SimpleNamespace()),
    )


class TestPdEagleDraftProposal(CustomTestCase):
    def build(self, spec, tokens=(3, 7)):
        with patch.object(eagle_disaggregation, "get_spec", return_value=spec):
            return build_eagle_disagg_draft_input(
                _batch(tokens), torch.tensor([1, 2]), future_map=None
            )

    def test_rejection_sampling_gets_one_hot_proposal_at_drafted_token(self):
        spec_info = self.build(_spec())
        expected = torch.zeros((2, VOCAB))
        expected[0, 3] = expected[1, 7] = 1.0
        self.assertEqual(spec_info.draft_probs.dtype, torch.float32)
        torch.testing.assert_close(spec_info.draft_probs, expected)
        torch.testing.assert_close(spec_info.topk_index, torch.tensor([[3], [7]]))

    def test_proposal_matches_target_vocab_width(self):
        # eagle_sample refuses a proposal whose width differs from the target's.
        spec_info = self.build(_spec())
        self.assertEqual(spec_info.draft_probs.shape[-1], VOCAB)

    def test_no_proposal_without_rejection_sampling(self):
        self.assertIsNone(self.build(_spec(rejection_sampling=False)).draft_probs)
        self.assertIsNone(
            self.build(_spec(rejection_sampling=False, multi_layer=True)).draft_probs
        )

    def test_multi_layer_gets_one_hot_proposal_per_chain_step(self):
        # Multi-layer EAGLE verifies the received chain with no draft forward,
        # so it needs a (b, num_steps, vocab) proposal; None fails the verify.
        chains = ((3, 5, 9), (7, 0, 10))
        spec_info = self.build(_spec(multi_layer=True), tokens=chains)
        expected = torch.zeros((2, 3, VOCAB))
        for row, chain in enumerate(chains):
            for step, token in enumerate(chain):
                expected[row, step, token] = 1.0
        self.assertEqual(spec_info.draft_probs.dtype, torch.float32)
        torch.testing.assert_close(spec_info.draft_probs, expected)
        torch.testing.assert_close(spec_info.topk_index, torch.tensor(chains))

    def test_retraction_snapshots_latest_relay_before_row_reuse(self):
        req = _batch((1,)).reqs[0]
        req.kv = SimpleNamespace(req_pool_idx=2)
        relay = SimpleNamespace(
            topk_p_buf=torch.tensor([[0.0], [0.0], [0.75]]),
            topk_index_buf=torch.tensor([[0], [0], [8]]),
            hidden_states_buf=torch.arange(12, dtype=torch.float32).view(3, 4),
            draft_probs_buf=torch.tensor([[0.1] * VOCAB, [0.1] * VOCAB, [0.2] * VOCAB]),
        )
        snapshot_retracted_eagle_input(
            req, 0, SimpleNamespace(enable_overlap=True), relay
        )
        relay.topk_index_buf[2].fill_(3)
        relay.hidden_states_buf[2].zero_()
        relay.draft_probs_buf[2].zero_()
        torch.testing.assert_close(req.output_topk_p, torch.tensor([0.75]))
        torch.testing.assert_close(req.output_topk_index, torch.tensor([8]))
        torch.testing.assert_close(
            req.hidden_states_tensor, torch.tensor([8.0, 9.0, 10.0, 11.0])
        )
        torch.testing.assert_close(
            req.retraction_draft_probs, torch.full((VOCAB,), 0.2)
        )

        batch = _batch((2, 7))
        batch.reqs[0] = req
        with patch.object(eagle_disaggregation, "get_spec", return_value=_spec()):
            info = build_eagle_disagg_draft_input(batch, torch.tensor([1, 2]), None)
        torch.testing.assert_close(info.topk_index, torch.tensor([[8], [7]]))
        torch.testing.assert_close(info.draft_probs[0], torch.full((VOCAB,), 0.2))
        expected_fresh = torch.nn.functional.one_hot(torch.tensor(7), VOCAB).float()
        torch.testing.assert_close(info.draft_probs[1], expected_fresh)
        self.assertIsNone(req.retraction_draft_probs)

    def test_retraction_without_hidden_state_relay_keeps_placeholder(self):
        req = _batch((1,)).reqs[0]
        req.kv = SimpleNamespace(req_pool_idx=0)
        original = req.hidden_states_tensor.clone()
        relay = SimpleNamespace(
            topk_p_buf=torch.tensor([[0.6, 0.7, 0.8]]),
            topk_index_buf=torch.tensor([[2, 3, 4]]),
            hidden_states_buf=None,
            draft_probs_buf=None,
        )
        snapshot_retracted_eagle_input(
            req, 0, SimpleNamespace(enable_overlap=True), relay
        )
        torch.testing.assert_close(req.hidden_states_tensor, original)
        torch.testing.assert_close(req.output_topk_index, torch.tensor([2, 3, 4]))
        self.assertIsNone(req.retraction_draft_probs)

    def test_retraction_snapshots_synchronous_draft_input(self):
        req = _batch((1,)).reqs[0]
        info = SimpleNamespace(
            topk_p=torch.tensor([[0.3], [0.9]]),
            topk_index=torch.tensor([[4], [9]]),
            hidden_states=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            draft_probs=None,
        )
        snapshot_retracted_eagle_input(
            req, 1, SimpleNamespace(enable_overlap=False, spec_info=info), None
        )
        torch.testing.assert_close(req.output_topk_p, torch.tensor([0.9]))
        torch.testing.assert_close(req.output_topk_index, torch.tensor([9]))
        torch.testing.assert_close(req.hidden_states_tensor, torch.tensor([3.0, 4.0]))
        self.assertIsNone(req.retraction_draft_probs)

    def test_snapshot_callback_runs_before_retraction_frees_row(self):
        from sglang.srt.managers import schedule_batch as batch_module

        reqs = [
            SimpleNamespace(beam_group=None, kv=SimpleNamespace(req_pool_idx=i))
            for i in range(2)
        ]
        events = []

        def release(idx, remaining):
            events.append(("release", reqs[idx].kv.req_pool_idx))
            reqs[idx].kv.req_pool_idx = None
            return True

        batch = SimpleNamespace(
            reqs=reqs,
            _get_decode_retraction_order=lambda _: [0, 1],
            check_decode_mem=lambda selected_indices=None: True,
            release_req=release,
            filter_batch=MagicMock(),
        )
        with (
            patch.object(batch_module, "beam_retraction_order", return_value=[0, 1]),
            patch.object(
                batch_module.NewTokenRatioTracker,
                "estimate_new_token_ratio_after_retract",
                return_value=0.5,
            ),
        ):
            retracted, _, aborted = batch_module.ScheduleBatch.retract_decode(
                batch,
                before_retract=lambda req, idx: events.append(
                    ("snapshot", req.kv.req_pool_idx)
                ),
            )
        self.assertEqual(events, [("snapshot", 1), ("release", 1)])
        self.assertEqual(retracted, [reqs[1]])
        self.assertEqual(aborted, [])

    def test_overlap_retraction_waits_for_pending_result(self):
        from sglang.srt.managers import scheduler as scheduler_module

        req = object()
        scheduler = SimpleNamespace(
            uocr_runtime=None,
            enable_overlap=True,
            result_queue=[(SimpleNamespace(reqs=[req]), object())],
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Qwen3_5MoeForConditionalGeneration"]
                )
            ),
            _reqs_finishing_inflight=lambda batch: [],
        )
        batch = SimpleNamespace(
            reqs=[req],
            batch_size=lambda: 2,
            filter_batch=MagicMock(),
            is_empty=lambda: False,
            check_decode_mem=lambda: False,
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: True, is_frozen_kv_mtp=lambda: False
            ),
            retract_decode=MagicMock(),
        )
        for architecture in (
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen4ExpForConditionalGeneration",
        ):
            scheduler.model_config.hf_config.architectures = [architecture]
            for cpu_buffer in (False, True):
                cfg = SimpleNamespace(
                    disaggregation_mode="decode", nixl_use_cpu_buffer=cpu_buffer
                )
                with (
                    self.subTest(architecture=architecture, cpu_buffer=cpu_buffer),
                    patch.object(scheduler_module, "get_disagg", return_value=cfg),
                ):
                    result = scheduler_module.Scheduler.update_running_batch(
                        scheduler, batch
                    )
                    self.assertIsNone(result)
                    batch.retract_decode.assert_not_called()

    def test_overlap_retraction_does_not_wait_for_unrelated_idle_result(self):
        from sglang.srt.managers import scheduler as scheduler_module

        req = object()
        scheduler = SimpleNamespace(
            uocr_runtime=None,
            enable_overlap=True,
            result_queue=[(SimpleNamespace(reqs=[]), object())],
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Qwen4ExpForConditionalGeneration"]
                )
            ),
            _reqs_finishing_inflight=lambda batch: [],
            schedule_stream=MagicMock(),
            forward_stream=object(),
            decode_offload_manager=None,
            token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 16),
            new_token_ratio_tracker=SimpleNamespace(current=0.5),
            tree_cache=SimpleNamespace(
                token_to_kv_pool_allocator=SimpleNamespace(state_slots=None)
            ),
        )
        batch = SimpleNamespace(
            reqs=[req],
            batch_size=lambda: 2,
            filter_batch=MagicMock(),
            is_empty=lambda: False,
            check_decode_mem=lambda: False,
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: True, is_frozen_kv_mtp=lambda: False
            ),
            retract_decode=MagicMock(side_effect=RuntimeError("retraction reached")),
        )
        with (
            patch.object(
                scheduler_module,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
            self.assertRaisesRegex(RuntimeError, "retraction reached"),
        ):
            scheduler_module.Scheduler.update_running_batch(scheduler, batch)
        scheduler.schedule_stream.wait_stream.assert_called_once_with(
            scheduler.forward_stream
        )

    def test_pd_decode_keeps_running_batch_when_retraction_waits(self):
        from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin

        batch = SimpleNamespace(is_empty=lambda: False)
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            get_new_prebuilt_batch=lambda _: None,
            update_running_batch=lambda _: None,
            dp_attn_adapter=SimpleNamespace(maybe_prepare_mlp_sync_batch=lambda x: x),
        )
        plan = SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
            scheduler, batch
        )
        self.assertIsNone(plan.batch_to_run)
        self.assertIs(plan.running_batch, batch)

    def test_one_hot_proposal_keeps_verify_exact(self):
        # Chain verify of one token: accept X if coin * q(X) < p(X), otherwise
        # resample from (p - q)+. Enumerating X ~ q_draft and both outcomes, the
        # committed token's distribution must equal p for any drafting q.
        p = torch.tensor([0.5, 0.3, 0.2])
        q_draft = torch.tensor([0.1, 0.6, 0.3])
        committed = torch.zeros(3)
        for x in range(3):
            q = pd_first_draft_proposal(torch.tensor([[x]]), 3)[0, 0]
            accept = torch.clamp(p[x] / q[x], max=1.0)
            residual = torch.clamp(p - q, min=0.0)
            residual = residual / residual.sum()
            committed[x] += q_draft[x] * accept
            committed += q_draft[x] * (1 - accept) * residual
        torch.testing.assert_close(committed, p)


if __name__ == "__main__":
    unittest.main()

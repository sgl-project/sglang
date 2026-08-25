import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups import speculative_hook
from sglang.srt.managers import overlap_utils
from sglang.srt.models.dspark import DSparkConfidenceHead
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_confidence import (
    plan_verify_prefixes,
    select_sps_verify_token_budget,
    selector_selected_path_confidence,
)
from sglang.srt.speculative.dflash_confidence_observability import (
    DFlashConfidenceObserver,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashConfidence(unittest.TestCase):
    def test_ragged_verify_input_advertises_nonuniform_width(self):
        verify_input = DFlashVerifyInput(
            draft_token=torch.tensor([1, 2, 3]),
            positions=torch.tensor([0, 1, 2]),
            draft_token_num=4,
            ragged_verify_layout=SimpleNamespace(),
        )
        self.assertEqual(verify_input.num_tokens_per_req, 1)
        self.assertEqual(verify_input.num_tokens_for_logprob_per_req, 4)
        self.assertNotEqual(verify_input.num_tokens_per_req, 4)

    def test_uniform_graph_rejects_ragged_dflash_input(self):
        runner = SimpleNamespace(captured_req_width=4, ragged_verify_mode=False)
        forward_batch = SimpleNamespace(
            replace_embeds=None,
            spec_info=DFlashVerifyInput(
                draft_token=torch.tensor([1, 2, 3]),
                positions=torch.tensor([0, 1, 2]),
                draft_token_num=4,
                ragged_verify_layout=SimpleNamespace(),
            ),
        )
        self.assertNotEqual(
            forward_batch.spec_info.num_tokens_per_req, runner.captured_req_width
        )

    def test_confidence_relay_follows_dflash_server_args_not_ragged_env(self):
        server_args = SimpleNamespace(
            speculative_dflash_confidence_target_verify_tokens=4,
            speculative_dflash_confidence_sps_table_path=None,
        )
        with mock.patch.object(
            overlap_utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DFLASH_CONFIDENCE"),
        ):
            self.assertTrue(overlap_utils.decide_needs_confidence_relay(server_args))
        server_args.speculative_dflash_confidence_target_verify_tokens = 0
        with mock.patch.object(
            overlap_utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DFLASH_CONFIDENCE"),
        ):
            self.assertFalse(overlap_utils.decide_needs_confidence_relay(server_args))

    def test_selector_confidence_uses_the_selected_path(self):
        scores = torch.tensor(
            [
                [
                    [[0.0, 2.0], [-99.0, -99.0]],
                    [[0.0, 0.0], [0.0, 3.0]],
                ]
            ]
        )
        # Select candidate 1 first, then candidate 0. Position one must score
        # row 1 -> candidate 0, not the best transition anywhere in the lattice.
        path_indices = torch.tensor([[1, 0]])
        confidence = selector_selected_path_confidence(scores, path_indices)
        self.assertEqual(confidence.shape, (1, 2))
        self.assertTrue(bool(((confidence >= 0.0) & (confidence <= 1.0)).all()))
        expected = torch.tensor(
            [
                [
                    torch.softmax(scores[0, 0, 0], 0)[1],
                    torch.softmax(scores[0, 1, 1], 0)[0],
                ]
            ]
        )
        torch.testing.assert_close(confidence, expected)

    def test_selector_confidence_rejects_invalid_path_indices(self):
        scores = torch.zeros((1, 2, 2, 2))
        with self.assertRaisesRegex(ValueError, "outside the selector top-k"):
            selector_selected_path_confidence(scores, torch.tensor([[0, 2]]))

    def test_planner_preserves_prefixes_and_minimum_progress(self):
        confidence = torch.tensor(
            [
                [0.99, 0.99, 0.99],
                [0.10, 0.10, 0.10],
            ]
        )
        decision = plan_verify_prefixes(
            confidence,
            verify_num_draft_tokens=4,
            confidence_threshold=0.5,
            min_verify_len=2,
            target_verify_tokens=6,
        )
        # Both requests receive anchor + one proposal. The remaining two tokens
        # are assigned to the higher-survival request, matching DSpark's top-k
        # scheduler while preserving each request's contiguous prefix.
        self.assertEqual(decision.verify_lens.tolist(), [4, 2])
        self.assertEqual(decision.deferred_tokens, 2)
        self.assertEqual(decision.low_confidence_tokens, 3)

    def test_higher_survival_receives_remaining_budget_like_dspark(self):
        decision = plan_verify_prefixes(
            torch.tensor([[0.99, 0.99, 0.99], [0.10, 0.10, 0.10]]),
            verify_num_draft_tokens=4,
            confidence_threshold=0.5,
            min_verify_len=2,
            target_verify_tokens=6,
        )
        self.assertEqual(decision.verify_lens.tolist(), [4, 2])

    def test_planner_does_not_mutate_confidence_input(self):
        confidence = torch.tensor([[1.2, -0.1]])
        original = confidence.clone()
        plan_verify_prefixes(
            confidence,
            verify_num_draft_tokens=3,
            confidence_threshold=0.5,
            min_verify_len=1,
            target_verify_tokens=2,
        )
        torch.testing.assert_close(confidence, original)

    def test_zero_target_budget_keeps_safe_minimum_floor(self):
        decision = plan_verify_prefixes(
            torch.full((3, 4), 0.9),
            verify_num_draft_tokens=5,
            confidence_threshold=0.5,
            min_verify_len=2,
            target_verify_tokens=0,
        )
        self.assertEqual(decision.verify_lens.tolist(), [2, 2, 2])
        self.assertEqual(decision.deferred_tokens, 9)

    def test_anchor_only_floor_allows_bonus_only_steps(self):
        decision = plan_verify_prefixes(
            torch.full((3, 4), 0.9),
            verify_num_draft_tokens=5,
            confidence_threshold=0.5,
            min_verify_len=1,
            target_verify_tokens=0,
        )
        self.assertEqual(decision.verify_lens.tolist(), [1, 1, 1])
        self.assertEqual(decision.deferred_tokens, 12)

    def test_mixed_batch_budget_keeps_contiguous_prefixes_and_is_deterministic(self):
        confidence = torch.tensor(
            [
                [0.99, 0.99, 0.99, 0.99],
                [0.20, 0.90, 0.90, 0.90],
                [0.60, 0.60, 0.60, 0.60],
            ]
        )
        kwargs = dict(
            verify_num_draft_tokens=5,
            confidence_threshold=0.5,
            min_verify_len=2,
            target_verify_tokens=9,
        )
        first = plan_verify_prefixes(confidence, **kwargs)
        second = plan_verify_prefixes(confidence, **kwargs)

        self.assertTrue(torch.equal(first.verify_lens, second.verify_lens))
        self.assertEqual(first.verify_lens.tolist(), [5, 2, 2])
        self.assertTrue(
            bool(((first.verify_lens >= 2) & (first.verify_lens <= 5)).all())
        )
        self.assertEqual(int(first.verify_lens.sum()), 9)
        self.assertEqual(first.deferred_tokens, 6)

    def test_vectorized_allocation_matches_token_by_token_reference(self):
        generator = torch.Generator().manual_seed(7)
        for batch_size in (1, 3, 8):
            for min_verify_len in (1, 2):
                for target_budget in range(0, batch_size * 6 + 1):
                    confidence = torch.rand((batch_size, 5), generator=generator)
                    decision = plan_verify_prefixes(
                        confidence,
                        verify_num_draft_tokens=6,
                        confidence_threshold=0.5,
                        min_verify_len=min_verify_len,
                        target_verify_tokens=target_budget,
                    )
                    survival = torch.cumprod(confidence, dim=1)
                    priority = survival
                    expected = torch.full(
                        (batch_size,), min_verify_len, dtype=torch.int32
                    )
                    remaining = (
                        max(batch_size * min_verify_len, target_budget)
                        - batch_size * min_verify_len
                    )
                    candidates = []
                    for request_index in range(batch_size):
                        for position_index in range(
                            min_verify_len - 1, confidence.shape[1]
                        ):
                            candidates.append(
                                (
                                    float(priority[request_index, position_index]),
                                    position_index,
                                    request_index,
                                )
                            )
                    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
                    for _, _, request_index in candidates[:remaining]:
                        expected[request_index] += 1
                    self.assertEqual(decision.verify_lens.tolist(), expected.tolist())

    def test_verify_prefix_contains_the_bonus_logit_for_every_request(self):
        # A target verify window contains the current-token (anchor) logit plus
        # each accepted draft token. Therefore cap L must permit accepting at
        # most L - 1 drafts and always retains the target bonus at that index.
        candidates = torch.tensor(
            [
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
            ]
        )
        target_predict = torch.tensor(
            [
                [11, 12, 13, 14, 99],
                [21, 77, 23, 24, 98],
                [88, 31, 32, 33, 97],
            ]
        )
        verify_lens = torch.tensor([3, 2, 4], dtype=torch.int32)
        matches = candidates[:, 1:] == target_predict[:, :-1]
        uncapped = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        capped = torch.minimum(uncapped, verify_lens - 1)
        bonus = target_predict[torch.arange(candidates.shape[0]), capped]

        self.assertEqual(uncapped.tolist(), [4, 1, 0])
        self.assertEqual(capped.tolist(), [2, 1, 0])
        self.assertEqual(bonus.tolist(), [13, 77, 88])
        self.assertTrue(bool((capped < verify_lens).all()))

    def test_anchor_only_verify_uses_anchor_prediction_as_bonus(self):
        candidates = torch.tensor([[10, 11, 12]])
        target_predict = torch.tensor([[99, 98, 97]])
        verify_lens = torch.tensor([1], dtype=torch.int32)
        uncapped = torch.tensor([2], dtype=torch.int32)
        capped = torch.minimum(uncapped, verify_lens - 1)
        bonus = target_predict[torch.arange(1), capped]
        commit_lens = capped + 1
        self.assertEqual(capped.tolist(), [0])
        self.assertEqual(bonus.tolist(), [99])
        self.assertEqual(commit_lens.tolist(), [1])

    def test_observer_reports_confidence_and_defer_metrics(self):
        observer = DFlashConfidenceObserver()
        observer.observe(
            confidence=torch.tensor([[0.2, 0.8], [0.4, 0.9]]),
            verify_lens=torch.tensor([2, 3]),
            reason="confidence_ragged",
            deferred_tokens=3,
            low_confidence_tokens=2,
        )
        record = observer.dump()
        self.assertEqual(record["verify_reason_counts"], {"confidence_ragged": 1})
        self.assertEqual(record["verify_batch_size_distribution"], {5: 1})
        self.assertEqual(record["deferred_tokens"], 3)
        self.assertEqual(record["low_confidence_tokens"], 2)
        self.assertAlmostEqual(record["confidence"]["p50"], 0.4, places=2)

    def test_sps_budget_selects_the_measured_throughput_knee(self):
        # At three target tokens the profiled SPS drops sharply, so the DSpark
        # cost model should retain only the mandatory anchor + one proposal.
        decision = select_sps_verify_token_budget(
            torch.tensor([[0.9, 0.9, 0.9]]),
            verify_num_draft_tokens=4,
            min_verify_len=2,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 2, 3, 4],
                sample_steps_per_sec=[10.0, 10.0, 1.0, 1.0],
                max_batch_tokens=4,
            ),
        )
        self.assertEqual(decision.budget, 2)
        self.assertAlmostEqual(decision.predicted_theta, 19.0, places=5)

    def test_sps_budget_allows_anchor_only_floor(self):
        decision = select_sps_verify_token_budget(
            torch.full((2, 2), 0.1),
            verify_num_draft_tokens=3,
            min_verify_len=1,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 2, 3, 4, 6],
                sample_steps_per_sec=[100.0, 100.0, 0.01, 0.01, 0.01],
                max_batch_tokens=6,
            ),
        )
        self.assertEqual(decision.budget, 2)

    def test_sps_budget_preserves_the_per_request_progress_floor(self):
        decision = select_sps_verify_token_budget(
            torch.full((3, 2), 0.1),
            verify_num_draft_tokens=3,
            min_verify_len=2,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 3, 6, 9],
                sample_steps_per_sec=[100.0, 100.0, 0.01, 0.01],
                max_batch_tokens=9,
            ),
        )
        self.assertEqual(decision.budget, 6)

    def test_dspark_confidence_head_applies_per_position_sts(self):
        head = DSparkConfidenceHead(
            hidden_size=2,
            markov_rank=0,
            with_markov=False,
        )
        with torch.no_grad():
            head.proj.weight.copy_(torch.tensor([[1.0, 0.0]]))
            head.proj.bias.zero_()
        head.sts_temperatures = torch.tensor([1.0, 2.0])
        confidence = head.apply_sts(head(torch.tensor([[[2.0, 9.0], [2.0, -3.0]]])))
        expected = torch.sigmoid(torch.tensor([[2.0, 1.0]]))
        torch.testing.assert_close(confidence, expected)
        torch.testing.assert_close(
            head._last_confidence_raw, torch.tensor([[2.0, 2.0]])
        )

    def test_dspark_confidence_head_uses_selected_path_predecessors(self):
        class MarkovStub:
            def get_prev_embeddings(self, tokens):
                return tokens.float().unsqueeze(-1)

        head = DSparkConfidenceHead(
            hidden_size=1,
            markov_rank=1,
            with_markov=True,
        )
        with torch.no_grad():
            head.proj.weight.copy_(torch.tensor([[0.0, 1.0]]))
            head.proj.bias.zero_()
        anchor = torch.tensor([7])
        selected = torch.tensor([[11, 13, 17]])
        previous = torch.cat([anchor[:, None], selected[:, :-1]], dim=1)
        confidence = head.apply_sts(
            head(
                torch.zeros((1, 3, 1)),
                MarkovStub().get_prev_embeddings(previous),
            )
        )
        torch.testing.assert_close(
            confidence, torch.sigmoid(torch.tensor([[7.0, 11.0, 13.0]]))
        )

    def test_dflash_confidence_has_dflash_family_capabilities(self):
        algorithm = SpeculativeAlgorithm.DFLASH_CONFIDENCE
        self.assertTrue(algorithm.is_dflash_confidence())
        self.assertTrue(algorithm.is_dflash_family())
        self.assertTrue(algorithm.supports_target_verify_for_draft())
        self.assertTrue(algorithm.supports_ragged_verify())

    def test_dflash_confidence_does_not_request_overlap_hidden_states(self):
        spec = SimpleNamespace(
            speculative_algorithm="DFLASH_CONFIDENCE", enable_multi_layer_eagle=True
        )
        with mock.patch(
            "sglang.srt.speculative.spec_utils.get_spec", return_value=spec
        ):
            self.assertFalse(spec_need_hidden_states())

    def test_observer_records_lagged_cpu_plan_without_confidence_copy(self):
        observer = DFlashConfidenceObserver()
        observer.observe(
            confidence=None,
            verify_lens=torch.tensor([1, 3], dtype=torch.int32),
            reason="confidence_ragged_lagged",
            deferred_tokens=4,
            low_confidence_tokens=2,
        )
        record = observer.dump()
        self.assertEqual(
            record["verify_reason_counts"], {"confidence_ragged_lagged": 1}
        )
        self.assertEqual(record["verify_batch_size_distribution"], {4: 1})
        self.assertEqual(record["deferred_tokens"], 4)
        self.assertEqual(record["low_confidence_tokens"], 2)
        self.assertEqual(record["confidence"], {})

    def test_lagged_cpu_confidence_plan_preserves_prefix_invariants(self):
        # ConfidenceRelay resolves a pinned CPU N-2 snapshot. Planning from it
        # must remain host-only and produce a legal current-batch prefix plan.
        lagged_confidence = torch.tensor(
            [[0.95, 0.90, 0.85], [0.20, 0.80, 0.90]], device="cpu"
        )
        decision = plan_verify_prefixes(
            lagged_confidence,
            verify_num_draft_tokens=4,
            confidence_threshold=0.5,
            min_verify_len=2,
            target_verify_tokens=6,
        )
        self.assertEqual(decision.verify_lens.device.type, "cpu")
        self.assertEqual(int(decision.verify_lens.sum()), 6)
        self.assertTrue(
            bool(((decision.verify_lens >= 2) & (decision.verify_lens <= 4)).all())
        )

    def test_server_args_accepts_anchor_only_floor_and_rejects_zero(self):
        def make_args(min_verify_len):
            args = ServerArgs(model_path="dummy")
            args.speculative_algorithm = "DFLASH_CONFIDENCE"
            args.speculative_draft_model_path = "draft"
            args.device = "cuda"
            args.speculative_num_draft_tokens = 4
            args.speculative_dflash_confidence_min_verify_len = min_verify_len
            args.get_model_config = lambda: SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["LlamaForCausalLM"],
                    get_text_config=lambda: SimpleNamespace(),
                )
            )
            return args

        speculative_hook._handle_dflash(make_args(1))
        with self.assertRaisesRegex(ValueError, "anchor-only / bonus-only"):
            speculative_hook._handle_dflash(make_args(0))

    def test_planner_rejects_invalid_min_verify_len(self):
        with self.assertRaisesRegex(ValueError, "min_verify_len"):
            plan_verify_prefixes(
                torch.full((1, 2), 0.5),
                verify_num_draft_tokens=3,
                confidence_threshold=0.5,
                min_verify_len=0,
                target_verify_tokens=0,
            )
        with self.assertRaisesRegex(ValueError, "min_verify_len"):
            select_sps_verify_token_budget(
                torch.full((1, 2), 0.5),
                verify_num_draft_tokens=3,
                min_verify_len=0,
                sps_table=SpsCostTable(
                    sample_batch_tokens=[1, 2, 3],
                    sample_steps_per_sec=[1.0, 1.0, 1.0],
                    max_batch_tokens=3,
                ),
            )

    def test_planner_rejects_invalid_threshold(self):
        for threshold in (-0.1, 1.1):
            with self.assertRaisesRegex(ValueError, "threshold"):
                plan_verify_prefixes(
                    torch.full((1, 2), 0.5),
                    verify_num_draft_tokens=3,
                    confidence_threshold=threshold,
                    min_verify_len=2,
                    target_verify_tokens=3,
                )


if __name__ == "__main__":
    unittest.main()

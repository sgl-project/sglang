import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import overlap_utils
from sglang.srt.speculative.dflash_confidence import (
    select_sps_verify_token_budget,
    selector_selected_path_confidence,
)
from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildRawCommitInjectLayout,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
)
from sglang.srt.speculative.dflash_confidence_observability import (
    DFlashConfidenceObserver,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _require_dflash_ragged_graph_coverage,
    _verify_logits_adjustments_are_noop,
)
from sglang.srt.server_args import ServerArgs
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

    def test_compact_graph_coverage_rejects_schedulable_bucket_miss(self):
        server_args = SimpleNamespace(
            disable_cuda_graph=False,
            max_running_requests=32,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=[1, 2, 4, 8, 16])
            ),
        )
        with mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DFLASH_CONFIDENCE"),
        ), mock.patch(
            "sglang.srt.speculative.dflash_worker_v2.ragged_verify_compact_enabled",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "max_running_requests=32"):
                _require_dflash_ragged_graph_coverage(server_args, block_size=8)
            server_args.cuda_graph_config.decode.bs.append(32)
            _require_dflash_ragged_graph_coverage(server_args, block_size=8)

    def test_dflash_confidence_uses_dense_cuda_graph_tiers(self):
        args = object.__new__(ServerArgs)
        args.disable_cuda_graph_padding = False
        args.speculative_algorithm = "DFLASH_CONFIDENCE"
        self.assertEqual(
            args._generate_decode_cuda_graph_batch_sizes(80),
            list(range(1, 17))
            + list(range(18, 65, 2))
            + list(range(68, 81, 4)),
        )

    def test_other_speculative_algorithms_keep_generic_cuda_graph_tiers(self):
        args = object.__new__(ServerArgs)
        args.disable_cuda_graph_padding = False
        args.speculative_algorithm = "DSPARK"
        self.assertEqual(
            args._generate_decode_cuda_graph_batch_sizes(80),
            list(range(1, 9))
            + list(range(10, 33, 2))
            + list(range(40, 65, 4))
            + list(range(72, 81, 8)),
        )

    def test_dflash_confidence_rejects_return_logprob_like_dspark(self):
        worker = object.__new__(DFlashWorkerV2)
        batch = SimpleNamespace(
            return_logprob=True,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
        )
        with self.assertRaisesRegex(
            ValueError, "DFLASH speculative decoding does not support return_logprob"
        ):
            worker.forward_batch_generation(batch)

    def test_graph_tier_alignment_fills_existing_padding_with_real_tokens(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 5
        worker.server_args = SimpleNamespace(
            speculative_dflash_confidence_align_verify_tokens_to_graph_tier=True
        )
        # bs=4 has four anchor rows. An SPS budget of three produces seven
        # useful rows, and replay already rounds it to the eight-token graph.
        self.assertEqual(
            worker._align_confidence_budget_to_graph_tier(
                bs=4, budget_extra=3, graph_buckets=[4, 8, 16]
            ),
            4,
        )
        # No bucket can cover the requested shape: retain the SPS decision.
        self.assertEqual(
            worker._align_confidence_budget_to_graph_tier(
                bs=4, budget_extra=17, graph_buckets=[4, 8, 16]
            ),
            17,
        )

    def test_graph_tier_alignment_is_opt_in(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 5
        worker.server_args = SimpleNamespace(
            speculative_dflash_confidence_align_verify_tokens_to_graph_tier=False
        )
        self.assertEqual(
            worker._align_confidence_budget_to_graph_tier(
                bs=4, budget_extra=3, graph_buckets=[4, 8, 16]
            ),
            3,
        )

    def test_graph_accept_is_armed_only_for_noop_greedy_adjustments(self):
        greedy = SimpleNamespace(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            acc_linear_penalties=None,
            penalizer_orchestrator=None,
            vocab_mask=None,
            logit_bias=None,
        )
        sampling = SimpleNamespace(
            is_all_greedy=False,
            has_custom_logit_processor=False,
            acc_linear_penalties=None,
            penalizer_orchestrator=None,
            vocab_mask=None,
            logit_bias=None,
        )
        grammar = SimpleNamespace(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            acc_linear_penalties=None,
            penalizer_orchestrator=None,
            vocab_mask=torch.ones(1, 1, dtype=torch.bool),
            logit_bias=None,
        )

        # Ragged layout selection is deliberately independent of these request
        # types.  Only graph-folded greedy acceptance requires no adjustment.
        self.assertTrue(_verify_logits_adjustments_are_noop(greedy))
        self.assertTrue(_verify_logits_adjustments_are_noop(sampling))
        self.assertFalse(_verify_logits_adjustments_are_noop(grammar))

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
        # cost model keeps one optional position above the mandatory floor
        # (theta 1.9 * 10 steps/s beats both floor-only 10 and wider 2.71 * 1).
        decision = select_sps_verify_token_budget(
            torch.tensor([[0.9, 0.9, 0.9]]),
            verify_num_draft_tokens=4,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 2, 3, 4],
                sample_steps_per_sec=[10.0, 10.0, 1.0, 1.0],
                max_batch_tokens=4,
            ),
        )
        self.assertEqual(decision.budget, 1)
        self.assertAlmostEqual(decision.predicted_theta, 19.0, places=5)

    def test_sps_budget_allows_anchor_only_floor(self):
        # Every optional position lands in the 0.01 steps/s tier, so the
        # optimum is the floor itself: zero optional positions above it.
        decision = select_sps_verify_token_budget(
            torch.full((2, 2), 0.1),
            verify_num_draft_tokens=3,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 2, 3, 4, 6],
                sample_steps_per_sec=[100.0, 100.0, 0.01, 0.01, 0.01],
                max_batch_tokens=6,
            ),
        )
        self.assertEqual(decision.budget, 0)

    def test_sps_budget_is_optional_positions_above_the_floor(self):
        decision = select_sps_verify_token_budget(
            torch.full((3, 2), 0.1),
            verify_num_draft_tokens=3,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 3, 6, 9],
                sample_steps_per_sec=[100.0, 100.0, 0.01, 0.01],
                max_batch_tokens=9,
            ),
        )
        self.assertEqual(decision.budget, 2)

    def test_sps_budget_filters_near_zero_survival_candidates(self):
        decision = select_sps_verify_token_budget(
            torch.tensor([[1e-8, 1.0]]),
            verify_num_draft_tokens=3,
            sps_table=SpsCostTable(
                sample_batch_tokens=[1, 2, 3],
                sample_steps_per_sec=[1.0, 100.0, 100.0],
                max_batch_tokens=3,
            ),
        )
        self.assertEqual(decision.budget, 0)

    def test_current_confidence_gpu_scheduler_consumes_lagged_extra_budget(self):
        # N-2 confidence determines the scalar budget only. The current step
        # allocates it to its own high-survival row on device.
        cfg = DSparkScheduleConfig(gamma=2)
        current_confidence = torch.tensor(
            [[0.1, 0.1], [0.99, 0.99]], dtype=torch.float32
        )
        verify_lens = ScheduleVerifyLensTopk.execute(
            confidence=current_confidence, budget=2, cfg=cfg
        )
        self.assertEqual(verify_lens.tolist(), [1, 3])

    def test_fixed_verify_width_scales_with_batch_size(self):
        # A configured width of eight must remain width eight when the batch
        # grows. It is not a global eight-token batch budget.
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 8
        worker._confidence_sps_table = None
        worker.server_args = SimpleNamespace(
            speculative_dflash_confidence_target_verify_tokens=8
        )
        for batch_size in (1, 4, 8):
            confidence = torch.ones((batch_size, 7), dtype=torch.float32)
            budget = worker._confidence_budget_extra(confidence)
            verify_lens = worker._schedule_confidence_verify_lens(confidence, budget)
            self.assertEqual(verify_lens.tolist(), [8] * batch_size)

    def test_raw_commit_inject_layout_matches_dense_verify_slots(self):
        # Compact verify scatters positions/out_cache_loc into [bs, stride]
        # before injecting accepted target hidden states. The graph layout path
        # must reconstruct exactly those full-pool slots from request metadata.
        req_to_token = torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106],
                [200, 201, 202, 203, 204, 205, 206],
                [300, 301, 302, 303, 304, 305, 306],
            ],
            dtype=torch.int64,
        )
        layout = BuildRawCommitInjectLayout.execute(
            req_pool_indices=torch.tensor([2, 0], dtype=torch.int64),
            req_to_token=req_to_token,
            prefix_lens=torch.tensor([1, 3], dtype=torch.int32),
            block_pos_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
            stride=3,
        )
        self.assertEqual(layout.positions.tolist(), [1, 2, 3, 3, 4, 5])
        self.assertEqual(
            layout.cache_loc.tolist(), [301, 302, 303, 103, 104, 105]
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

    def test_observer_records_lagged_budget_without_confidence_copy(self):
        observer = DFlashConfidenceObserver()
        observer.observe(
            confidence=None,
            verify_lens=None,
            reason="confidence_ragged_lagged_budget",
        )
        record = observer.dump()
        self.assertEqual(
            record["verify_reason_counts"], {"confidence_ragged_lagged_budget": 1}
        )
        self.assertEqual(record["confidence"], {})

if __name__ == "__main__":
    unittest.main()

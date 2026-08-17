"""Unit tests for sampling-support capture and overflow policy."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.layers.sampler import (
    Sampler,
    _SamplingMaskCapture,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSamplingMaskCapture(CustomTestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.sampling_mask_max_tokens = 3
        self.sampler.tp_sync_group = None
        self.sampler.cp_sync_group = None

    def _capture(
        self,
        weights,
        sampled_tokens,
        token_ids=None,
        batch_indices=None,
        selected_weights=None,
    ):
        if batch_indices is None:
            batch_indices = torch.arange(weights.shape[0])
        return self.sampler._build_sampling_mask_output(
            torch.tensor(sampled_tokens),
            _SamplingMaskCapture(
                batch_indices=batch_indices,
                scores=weights,
                token_ids=token_ids,
                selected_scores=selected_weights,
            ),
        )

    def test_returns_complete_support_and_selected_logprob(self):
        output = self._capture(torch.tensor([[0.4, 0.3, 0.2, 0.0]]), sampled_tokens=[1])

        length = int(output.lengths[0])
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        self.assertEqual(set(output.token_ids[0, :length].tolist()), {0, 1, 2})
        self.assertAlmostEqual(
            float(output.selected_logprobs[0]), math.log(0.3 / 0.9), places=6
        )

    def test_support_above_cap_is_truncated_and_keeps_sampled_token(self):
        output = self._capture(torch.tensor([[0.4, 0.3, 0.2, 0.1]]), sampled_tokens=[3])

        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.TRUNCATED])
        self.assertEqual(output.lengths.tolist(), [3])
        self.assertEqual(set(output.token_ids[0].tolist()), {0, 1, 3})
        self.assertAlmostEqual(float(output.selected_logprobs[0]), math.log(0.1))
        self.assertNotAlmostEqual(
            float(output.selected_logprobs[0]), math.log(0.1 / 0.8)
        )

    def test_scheduler_applies_exact_and_bounded_overflow_policy(self):
        output = self._capture(torch.tensor([[0.4, 0.3, 0.2, 0.1]]), sampled_tokens=[3])
        materialized = LogitsProcessorOutput(
            next_token_logits=None, sampling_mask_output=output
        )
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [SimpleNamespace(return_sampling_mask=True)], materialized
        )

        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(
            processor,
            "server_args",
            SimpleNamespace(sampling_mask_max_tokens=3),
        )
        finish_reason = processor.get_sampling_mask_finish_reason(
            status=materialized.next_token_sampling_mask_status[0], mode="exact"
        )
        self.assertEqual(finish_reason.status_code, 400)
        self.assertIn("sampling_mask_mode='bounded'", finish_reason.message)

        bounded_req = SimpleNamespace(
            return_sampling_mask=True,
            sampling_mask_mode="bounded",
            output_token_sampling_mask=[],
            output_token_sampling_logprobs=[],
            output_token_sampling_mask_truncated=[],
        )
        self.assertIsNone(
            processor.get_sampling_mask_finish_reason(
                status=materialized.next_token_sampling_mask_status[0], mode="bounded"
            )
        )
        processor.add_sampling_mask_return_values(0, bounded_req, materialized)

        self.assertEqual(bounded_req.output_token_sampling_mask_truncated, [True])
        self.assertEqual(set(bounded_req.output_token_sampling_mask[0]), {0, 1, 3})
        self.assertAlmostEqual(
            bounded_req.output_token_sampling_logprobs[0], math.log(0.1)
        )

    def test_sampled_token_is_not_added_to_support(self):
        output = self._capture(torch.tensor([[0.6, 0.4, 0.0]]), sampled_tokens=[2])

        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.INVALID])

    def test_token_id_mapping_uses_sampler_order(self):
        output = self._capture(
            weights=torch.tensor([[0.7, 0.3, 0.0, 0.0]]),
            token_ids=torch.tensor([[3, 1, 2, 0]], dtype=torch.int32),
            sampled_tokens=[1],
            selected_weights=torch.tensor([0.3]),
        )

        length = int(output.lengths[0])
        self.assertEqual(set(output.token_ids[0, :length].tolist()), {1, 3})
        self.assertAlmostEqual(float(output.selected_logprobs[0]), math.log(0.3))

    def test_selected_logprob_follows_synchronized_token(self):
        """A post-sampling token sync must also update the selected logprob."""
        self.sampler.rl_on_policy_target = None
        self.sampler.use_log_softmax_logprob = False
        self.sampler.enable_deterministic = False
        self.sampler.use_ascend_backend = False
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.tensor([[2.0, 1.0, 0.0]])
        )
        sampling_info = SimpleNamespace(
            has_custom_logit_processor=False,
            sampling_mask_batch_indices=torch.tensor([0]),
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            temperatures=torch.ones(1, 1),
            grammars=None,
        )
        captured = _SamplingMaskCapture(
            batch_indices=torch.tensor([0]),
            scores=torch.tensor([[0.5, 0.3, 0.2]]),
            token_ids=torch.tensor([[0, 2, 1]], dtype=torch.int32),
            selected_scores=torch.tensor([0.3]),
        )

        def sync_to_token_one(token_ids, _):
            token_ids.fill_(1)

        with (
            patch.object(
                self.sampler,
                "_sample_from_probs",
                return_value=(torch.tensor([2]), captured),
            ),
            patch.object(
                self.sampler,
                "_sync_token_ids_across_tp",
                side_effect=sync_to_token_one,
            ),
            patch("sglang.srt.layers.sampler.SYNC_TOKEN_IDS_ACROSS_TP", True),
        ):
            sampled_tokens = self.sampler.forward(
                logits_output,
                sampling_info,
                return_logprob=False,
                top_logprobs_nums=[0],
                token_ids_logprobs=[None],
                positions=torch.tensor([0]),
            )

        output = logits_output.sampling_mask_output
        self.assertEqual(sampled_tokens.tolist(), [1])
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        self.assertAlmostEqual(float(output.selected_logprobs[0]), math.log(0.2))

    def test_min_p_capture_filters_without_mutating_sampler_probs(self):
        renormalized_probs = torch.tensor([[0.6, 0.3, 0.1]])
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            sampling_mask_batch_indices=torch.tensor([0]),
            need_min_p_sampling=True,
            top_ks=torch.tensor([3]),
            top_ps=torch.tensor([1.0]),
            min_ps=torch.tensor([0.5]),
        )
        with (
            patch(
                "sglang.srt.layers.sampler.get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="flashinfer")
                ),
            ),
            patch(
                "sglang.srt.layers.sampler.top_k_renorm_prob",
                return_value=renormalized_probs,
            ),
            patch(
                "sglang.srt.layers.sampler.top_p_renorm_prob",
                side_effect=lambda probs, _: probs,
            ),
            patch(
                "sglang.srt.layers.sampler.min_p_sampling_from_probs",
                return_value=torch.tensor([0]),
            ),
        ):
            _, sampling_mask_data = self.sampler._sample_from_probs(
                torch.tensor([[0.6, 0.3, 0.1]]),
                sampling_info,
                positions=torch.tensor([0]),
                simple_sampling_case=False,
            )

        self.assertTrue(
            torch.equal(sampling_mask_data.scores, torch.tensor([[0.6, 0.3, 0.0]]))
        )
        self.assertTrue(
            torch.equal(renormalized_probs, torch.tensor([[0.6, 0.3, 0.1]]))
        )

    def test_logprob_sampling_capture_uses_producer_distribution(self):
        self.sampler.rl_on_policy_target = "test"
        self.sampler.use_log_softmax_logprob = True
        self.sampler.enable_deterministic = True
        self.sampler.use_ascend_backend = False
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
        logits_output = LogitsProcessorOutput(next_token_logits=logits.clone())
        sampling_info = SimpleNamespace(
            has_custom_logit_processor=False,
            sampling_mask_batch_indices=torch.tensor([1]),
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            temperatures=torch.ones(2, 1),
            sampling_seed=torch.tensor([1, 2]),
            grammars=None,
        )

        with patch.object(
            self.sampler,
            "_sample_from_logprobs",
            return_value=torch.tensor([0, 1]),
        ):
            self.sampler.forward(
                logits_output,
                sampling_info,
                return_logprob=False,
                top_logprobs_nums=[0, 0],
                token_ids_logprobs=[None, None],
                positions=torch.tensor([0, 0]),
            )

        output = logits_output.sampling_mask_output
        self.assertEqual(output.token_ids.shape[0], 1)
        self.assertEqual(output.statuses.tolist(), [SamplingMaskStatus.OK])
        producer_logprobs = torch.log_softmax(logits.bfloat16(), dim=-1)
        producer_weights = producer_logprobs[1].float().exp()
        self.assertAlmostEqual(
            float(output.selected_logprobs[0]),
            float(torch.log(producer_weights[1] / producer_weights.sum())),
            places=6,
        )

    def test_greedy_support_is_singleton(self):
        output = self.sampler._build_greedy_sampling_mask_output(
            torch.tensor([0, 2]), torch.tensor([4, 5, 6])
        )

        self.assertEqual(output.token_ids.tolist(), [[4], [6]])
        self.assertEqual(output.lengths.tolist(), [1, 1])
        self.assertEqual(output.selected_logprobs.tolist(), [0.0, 0.0])

    def test_materialization_preserves_batch_rows(self):
        sampling_output = self._capture(
            weights=torch.tensor(
                [
                    [0.6, 0.4, 0.0, 0.0],
                    [0.4, 0.3, 0.2, 0.1],
                ]
            ),
            sampled_tokens=[1, 9, 0],
            batch_indices=torch.tensor([0, 2]),
        )
        output = LogitsProcessorOutput(
            next_token_logits=None, sampling_mask_output=sampling_output
        )

        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            [
                SimpleNamespace(return_sampling_mask=True),
                SimpleNamespace(return_sampling_mask=False),
                SimpleNamespace(return_sampling_mask=True),
            ],
            output,
        )

        self.assertEqual(set(output.next_token_sampling_mask_idx[0]), {0, 1})
        self.assertIsNone(output.next_token_sampling_mask_idx[1])
        self.assertEqual(set(output.next_token_sampling_mask_idx[2]), {0, 1, 2})
        self.assertEqual(
            output.next_token_sampling_mask_status,
            [SamplingMaskStatus.OK, None, SamplingMaskStatus.TRUNCATED],
        )
        self.assertIsNone(output.sampling_mask_output)

    def test_pipeline_parallel_round_trip_preserves_tensor_output(self):
        sampling_output = self._capture(
            weights=torch.tensor([[0.6, 0.4, 0.0]]), sampled_tokens=[0]
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None, sampling_mask_output=sampling_output
            )
        )

        received, _, _ = get_logprob_from_pp_outputs(
            PPProxyTensors(get_logprob_dict_from_result(result))
        )

        self.assertIsNotNone(received.sampling_mask_output)
        self.assertTrue(
            torch.equal(
                received.sampling_mask_output.token_ids, sampling_output.token_ids
            )
        )
        self.assertTrue(
            torch.equal(
                received.sampling_mask_output.statuses, sampling_output.statuses
            )
        )

    def test_pytorch_capture_uses_filtered_support_with_and_without_top_k(self):
        for top_k, top_p in ((2, 0.5), (TOP_K_ALL, 0.49)):
            with self.subTest(top_k=top_k, top_p=top_p):
                probs = torch.tensor([[0.30, 0.20, 0.18, 0.17, 0.15]])
                (
                    _,
                    filtered,
                    token_ids,
                    _,
                ) = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    top_ks=torch.tensor([top_k]),
                    top_ps=torch.tensor([top_p]),
                    min_ps=torch.tensor([0.0]),
                    need_min_p_sampling=False,
                    sampling_seed=None,
                    positions=torch.tensor([0]),
                    return_filtered_probs=True,
                )

                kept_ids = token_ids[filtered > 0].tolist()
                self.assertEqual(set(kept_ids), {0, 1})


if __name__ == "__main__":
    unittest.main()

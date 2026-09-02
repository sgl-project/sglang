"""CPU regression coverage for logits dispatch and sampling-mask fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSamplerFromLogitsDispatch(unittest.TestCase):
    def _forward(self, *, capture=False, return_logprob=False, disable=False):
        sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
        torch.nn.Module.__init__(sampler)
        sampler.enable_deterministic = False
        sampler.disable_sampling_from_logits = disable
        sampler.rl_on_policy_target = None
        sampler.use_ascend_backend = False
        sampler.use_log_softmax_logprob = False
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        original_logits = logits.clone()
        token_ids = torch.tensor([2, 0])
        captured = object()
        mask_output = object()
        info = SimpleNamespace(
            is_all_greedy=False,
            sampling_mask_batch_indices=torch.tensor([1]) if capture else None,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            sampling_seed=None,
            temperatures=torch.ones((2, 1)),
            grammars=None,
        )
        output = SimpleNamespace(next_token_logits=logits)
        sampler._preprocess_logits = Mock(return_value=logits)
        sampler._sample_from_scaled_logits = Mock(return_value=token_ids)
        sampler._sample_from_probs = Mock(return_value=(token_ids, captured))
        sampler._sync_token_ids_across_tp = Mock()
        sampler._build_sampling_mask_output = Mock(return_value=mask_output)
        sampler.output_logprob_processor = Mock()
        with (
            patch.object(sampler_module, "_HAS_FLASHINFER_LOGITS_SAMPLING", True),
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch.object(sampler_module, "SGLANG_RETURN_ORIGINAL_LOGPROB", False),
            patch.object(sampler_module, "_trace_e2e_sampler"),
            patch.object(
                sampler_module,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(sampling_backend="flashinfer")
                ),
            ),
        ):
            actual = sampler.forward(output, info, return_logprob, [0, 0], [], None)
        self.assertIs(actual, token_ids)
        return sampler, output, original_logits, captured, mask_output

    def test_fast_path_skips_probs_and_mask_capture(self):
        sampler, output, logits, _, _ = self._forward()
        sampler._sample_from_scaled_logits.assert_called_once()
        sampler._sample_from_probs.assert_not_called()
        sampler._build_sampling_mask_output.assert_not_called()
        self.assertTrue(torch.equal(output.next_token_logits, logits))

    def test_sampling_mask_preserves_tuple_capture_from_probs_path(self):
        sampler, output, logits, captured, mask_output = self._forward(capture=True)
        sampler._sample_from_scaled_logits.assert_not_called()
        sampler._sample_from_probs.assert_called_once()
        self.assertTrue(
            torch.equal(output.next_token_logits, torch.softmax(logits, -1))
        )
        self.assertIs(sampler._build_sampling_mask_output.call_args.args[1], captured)
        self.assertIs(output.sampling_mask_output, mask_output)

    def test_escape_hatch_preserves_probs_path(self):
        sampler, _, _, _, _ = self._forward(disable=True)
        sampler._sample_from_scaled_logits.assert_not_called()
        sampler._sample_from_probs.assert_called_once()

    def test_logprob_request_preserves_probs_path(self):
        sampler, _, logits, _, _ = self._forward(return_logprob=True)
        sampler._sample_from_scaled_logits.assert_not_called()
        sampler._sample_from_probs.assert_called_once()
        actual_logprobs = (
            sampler.output_logprob_processor.compute_logprobs.call_args.args[0]
        )
        self.assertTrue(
            torch.equal(actual_logprobs, torch.log(torch.softmax(logits, -1)))
        )

    def test_missing_logits_kernels_disables_fast_path(self):
        sampler = SimpleNamespace(disable_sampling_from_logits=False)
        with patch.object(sampler_module, "_HAS_FLASHINFER_LOGITS_SAMPLING", False):
            self.assertFalse(
                sampler_module.Sampler._should_sample_from_scaled_logits(
                    sampler, SimpleNamespace(), False, False
                )
            )


if __name__ == "__main__":
    unittest.main()

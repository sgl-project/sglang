"""Final split-prefill sampling must use the active lane's communicator."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import logits_processor, sampler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPDMuxSampling(unittest.TestCase):
    def test_sampling_resolves_active_tp_group_each_call(self):
        instance = sampler.Sampler.__new__(sampler.Sampler)
        instance._resolve_tp_sync_group_per_call = True
        instance.tp_sync_group = "construction-group"
        token_ids = torch.tensor([1, 2])
        sampling_info = SimpleNamespace(grammars=True)
        with (
            patch.object(sampler, "is_dp_attention_enabled", return_value=False),
            patch.object(sampler, "get_parallel") as parallel,
            patch.object(sampler.dist, "all_reduce") as reduce,
        ):
            for lane in ("prefill", "decode", "prefill"):
                parallel.return_value = SimpleNamespace(
                    tp_group=SimpleNamespace(device_group=lane)
                )
                instance._sync_token_ids_across_tp(token_ids, sampling_info)
                self.assertEqual(reduce.call_args.kwargs["group"], lane)

    def test_sampling_ordinary_path_keeps_cached_group(self):
        instance = sampler.Sampler.__new__(sampler.Sampler)
        instance._resolve_tp_sync_group_per_call = False
        instance.tp_sync_group = "cached"
        with patch.object(sampler, "get_parallel") as parallel:
            self.assertEqual(instance._get_tp_sync_group(), "cached")
            parallel.assert_not_called()

    def test_pdmux_does_not_enable_shared_logits_workspace(self):
        parallel = SimpleNamespace(
            enable_dp_lm_head=False,
            enable_tp_lm_head_all_to_all=False,
            tp_size=8,
            attn_dp_size=1,
            tp_group=SimpleNamespace(cpu_group="cpu"),
        )
        execution = SimpleNamespace(
            features=SimpleNamespace(enable_fp32_lm_head=False, enable_mis=False),
            deterministic=SimpleNamespace(rl_on_policy_target=None),
        )
        with (
            patch.object(logits_processor, "get_parallel", return_value=parallel),
            patch.object(logits_processor, "get_exec", return_value=execution),
            patch.object(logits_processor, "get_disagg") as disagg,
            patch.object(logits_processor, "InputLogprobProcessor", Mock()),
            patch.object(
                logits_processor.triton_symm_mem_ag,
                "recommended_max_tokens",
                return_value=128,
            ),
            patch.object(
                logits_processor.triton_symm_mem_ag, "MultimemAllGatherer"
            ) as gather,
        ):
            for pdmux in (False, True):
                disagg.return_value = SimpleNamespace(enable_pdmux=pdmux)
                logits_processor.LogitsProcessor(SimpleNamespace(vocab_size=32))
                self.assertEqual(gather.call_args.kwargs["enabled"], not pdmux)


if __name__ == "__main__":
    unittest.main()

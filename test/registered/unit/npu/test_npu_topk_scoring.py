"""Compare simple NPU routing against an independent FP32 scoring reference."""

import itertools
import unittest
from unittest.mock import Mock, patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.moe import topk as npu_topk
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="base-a-test-npu")


class TestNPUSimpleTopK(unittest.TestCase):
    def _check(self, dtype, scoring_func, renormalize, apply_scale, tokens, k, padded):
        generator = torch.Generator().manual_seed(42)
        logits_cpu = torch.randn(tokens, 32, generator=generator).to(dtype)
        logits = logits_cpu.npu()
        hidden = torch.zeros(tokens, 16, dtype=dtype, device=logits.device)
        config = TopKConfig(
            top_k=k,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=apply_scale,
        )
        valid_tokens = tokens // 2 if padded else tokens
        num_non_padded = (
            torch.tensor(valid_tokens, dtype=torch.int32, device=logits.device)
            if padded
            else None
        )
        recorder = Mock()
        with (
            patch.object(npu_topk, "capture_routed_experts_if_allowed"),
            patch.object(
                npu_topk, "topk_ids_logical_to_physical", side_effect=lambda ids, _: ids
            ),
            patch.object(
                npu_topk,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            output = npu_topk.fused_topk_npu(
                hidden, logits, config, num_token_non_padded=num_non_padded
            )

        scores = (
            logits_cpu.float().sigmoid()
            if scoring_func == "sigmoid"
            else logits_cpu.float().softmax(dim=-1)
        )
        expected_ids = scores.topk(k, dim=-1).indices
        actual_ids = output.topk_ids.cpu().long()
        torch.testing.assert_close(
            actual_ids.sort(dim=-1).values, expected_ids.sort(dim=-1).values
        )
        expected = scores.gather(1, actual_ids)
        if renormalize:
            expected /= expected.sum(dim=-1, keepdim=True) + 1e-20
        if apply_scale:
            expected *= 2.5
        expected[valid_tokens:] = 0
        tolerance = 2e-6 if dtype == torch.float32 else 5e-3
        torch.testing.assert_close(
            output.topk_weights.cpu(), expected, rtol=tolerance, atol=tolerance
        )
        self.assertEqual(output.topk_weights.dtype, torch.float32)
        self.assertEqual(output.topk_ids.dtype, torch.int32)
        self.assertFalse(config.torch_native)
        recorder.on_select_experts.assert_called_once()
        recorded_ids = recorder.on_select_experts.call_args.kwargs["topk_ids"].cpu()
        expected_recorded_ids = actual_ids.to(torch.int32)
        expected_recorded_ids[valid_tokens:] = -1
        torch.testing.assert_close(recorded_ids, expected_recorded_ids)

    def test_scoring_normalization_and_scaling(self):
        for case in itertools.product(
            (torch.float32, torch.float16, torch.bfloat16),
            ("softmax", "sigmoid"),
            (False, True),
            (False, True),
            (1, 4),
            (2, 4),
            (False, True),
        ):
            with self.subTest(case=case):
                self._check(*case)


if __name__ == "__main__":
    unittest.main()

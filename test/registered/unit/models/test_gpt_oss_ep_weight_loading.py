"""Unit tests for GPT-OSS expert-parallel checkpoint slicing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.gpt_oss import _narrow_fused_moe_ep_weight
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGptOssEpWeightLoading(CustomTestCase):
    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_global_expert_weights_are_sliced_for_ep_rank(self, get_parallel):
        """Global checkpoint experts must be narrowed to the current EP rank."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(128 * 2).view(128, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        torch.testing.assert_close(actual, weight[64:96])

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_ep_one_keeps_global_expert_weights(self, get_parallel):
        """EP1 must preserve the checkpoint tensor without an unnecessary copy."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        weight = torch.arange(128 * 2).view(128, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        self.assertIs(actual, weight)

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_local_expert_weights_are_not_sliced_again(self, get_parallel):
        """Already-local checkpoint experts must not be narrowed a second time."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(32 * 2).view(32, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        self.assertIs(actual, weight)

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_invalid_expert_dimension_is_rejected(self, get_parallel):
        """A checkpoint must contain either all experts or this rank's experts."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(64 * 2).view(64, 2)

        with self.assertRaisesRegex(
            ValueError, "Expected 128 global or 32 local experts, got 64"
        ):
            _narrow_fused_moe_ep_weight(weight, num_experts=128)


if __name__ == "__main__":
    unittest.main()

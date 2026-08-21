"""Regression tests for FlashInfer min-p sampling semantics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestFlashInferMinPSampling(CustomTestCase):
    def test_filtered_support_matches_joint_top_k_top_p_support(self):
        # With top-k applied first, its renormalization makes top-p reject token
        # 3. Joint filtering retains it because the original mass before token 3
        # is 0.8, below top_p=0.82.
        original_probs = torch.tensor(
            [[0.35, 0.25, 0.20, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]],
            device="cuda",
        )
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_min_p_sampling=True,
            top_ks=torch.tensor([4], dtype=torch.int32, device="cuda"),
            top_ps=torch.tensor([0.82], device="cuda"),
            min_ps=torch.tensor([0.1], device="cuda"),
        )
        captured = {}

        def capture_min_p_input(probs, _min_ps):
            captured["probs"] = probs.clone()
            return torch.zeros(1, dtype=torch.int32, device=probs.device)

        execution_context = SimpleNamespace(
            kernel=SimpleNamespace(sampling_backend="flashinfer")
        )
        with (
            patch.object(sampler_module, "get_exec", return_value=execution_context),
            patch.object(
                sampler_module,
                "min_p_sampling_from_probs",
                side_effect=capture_min_p_input,
            ),
        ):
            sampler_module.Sampler._sample_from_probs(
                None,
                original_probs.clone(),
                sampling_info,
                positions=torch.zeros(1, dtype=torch.int64, device="cuda"),
                simple_sampling_case=False,
            )

        filtered_probs = captured["probs"]
        min_p_threshold = filtered_probs.max(
            dim=-1, keepdim=True
        ).values * sampling_info.min_ps.view(-1, 1)
        actual_support = torch.nonzero(
            (filtered_probs > 0) & (filtered_probs >= min_p_threshold),
            as_tuple=False,
        )[:, 1].tolist()

        sorted_probs, sorted_ids = original_probs.sort(dim=-1, descending=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        positions = torch.arange(sorted_probs.shape[-1], device="cuda").view(1, -1)
        expected_keep = positions < sampling_info.top_ks.view(-1, 1)
        expected_keep &= mass_before <= sampling_info.top_ps.view(-1, 1)
        expected_keep &= sorted_probs >= (
            sorted_probs[:, :1] * sampling_info.min_ps.view(-1, 1)
        )
        expected_support = sorted_ids[expected_keep].tolist()

        self.assertEqual(expected_support, [0, 1, 2, 3])
        self.assertEqual(actual_support, expected_support)


if __name__ == "__main__":
    unittest.main()

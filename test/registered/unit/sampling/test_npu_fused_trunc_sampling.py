"""Unit tests for the Ascend fused truncation sampling path."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestFusedTruncSampling(unittest.TestCase):
    def test_top_k_one_returns_argmax(self):
        logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]] * 3)
        top_ks = torch.ones(3, dtype=torch.int32)
        top_ps = torch.ones(3)
        with envs.SGLANG_NPU_FUSED_TRUNC_K.override(4):
            out = sampler_module._npu_fused_trunc_sampling_from_logits(
                logits, top_ks, top_ps
            )
        self.assertEqual(out.tolist(), [1, 1, 1])

    def test_top_p_keeps_only_the_nucleus(self):
        # The first token holds nearly all of the mass, well past top_p.
        logits = torch.zeros(2, 64)
        logits[:, 0] = 10.0
        top_ks = torch.full((2,), 64, dtype=torch.int32)
        top_ps = torch.full((2,), 0.5)
        with envs.SGLANG_NPU_FUSED_TRUNC_K.override(64):
            out = sampler_module._npu_fused_trunc_sampling_from_logits(
                logits, top_ks, top_ps
            )
        self.assertEqual(out.tolist(), [0, 0])

    def test_candidate_set_bounds_the_sample(self):
        logits = torch.zeros(2, 128)
        logits[:, 5] = 3.0
        logits[:, 7] = 2.0
        top_ks = torch.full((2,), 2, dtype=torch.int32)
        top_ps = torch.ones(2)
        with envs.SGLANG_NPU_FUSED_TRUNC_K.override(128):
            out = sampler_module._npu_fused_trunc_sampling_from_logits(
                logits, top_ks, top_ps
            )
        self.assertTrue(set(out.tolist()).issubset({5, 7}))

    def test_dispatch_uses_the_fused_path_only_when_eligible(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        fused_token = torch.tensor([3], dtype=torch.int32)

        for eligible in (True, False):
            with self.subTest(eligible=eligible):
                fused = MagicMock(return_value=fused_token.clone())
                with (
                    patch.object(
                        sampler_module, "torch_npu", SimpleNamespace(), create=True
                    ),
                    envs.SGLANG_NPU_USE_FUSED_TRUNC.override(True),
                    patch.object(
                        sampler_module,
                        "_npu_fused_trunc_sampling_from_logits",
                        fused,
                    ),
                ):
                    out = sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
                        logits.clone(),
                        torch.tensor([2], dtype=torch.int32),
                        torch.ones(1),
                        torch.zeros(1),
                        False,
                        None,
                        torch.zeros(1, dtype=torch.int64),
                        trunc_fused_eligible=eligible,
                    )

                self.assertEqual(fused.called, eligible)
                self.assertEqual(tuple(out.shape), (1,))
                if eligible:
                    self.assertEqual(out.tolist(), [3])

    def test_min_p_and_seeded_requests_bypass_the_fused_path(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        fused = MagicMock()
        with (
            patch.object(sampler_module, "torch_npu", SimpleNamespace(), create=True),
            envs.SGLANG_NPU_USE_FUSED_TRUNC.override(True),
            patch.object(
                sampler_module, "_npu_fused_trunc_sampling_from_logits", fused
            ),
            patch.object(
                sampler_module,
                "multinomial_with_seed",
                MagicMock(return_value=torch.zeros((1, 1), dtype=torch.int64)),
            ),
        ):
            sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
                logits.clone(),
                torch.tensor([2], dtype=torch.int32),
                torch.ones(1),
                torch.zeros(1),
                True,  # need_min_p_sampling
                None,
                torch.zeros(1, dtype=torch.int64),
                trunc_fused_eligible=True,
            )
            sampler_module.top_k_top_p_min_p_sampling_from_logits_ascend(
                logits.clone(),
                torch.tensor([2], dtype=torch.int32),
                torch.ones(1),
                torch.zeros(1),
                False,
                torch.tensor([12345], dtype=torch.int64),  # sampling_seed
                torch.zeros(1, dtype=torch.int64),
                trunc_fused_eligible=True,
            )
        self.assertFalse(fused.called)


if __name__ == "__main__":
    unittest.main()

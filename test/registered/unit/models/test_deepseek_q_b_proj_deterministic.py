"""Hermetic unit tests: deterministic inference keeps q_b_proj on one GEMM.

`q_b_proj_forward` sends 1 <= M <= 16 rows to the min-latency
`dsv3_fused_a_gemm` for verified q_b_proj weight shapes and larger M to the
regular linear. The two kernels round differently, so under deterministic
inference a token's q (and its logprobs) would depend on how many tokens share
the forward pass. `prepare_qkv_latent` already turns its min-latency GEMM off
in that mode; this pins the same for q_b_proj.

Pure Python (no GPU, no model weights): the eligibility check, the fused GEMM
and `get_exec` are patched; `attn` is a lightweight fake.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models import deepseek_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_attn():
    return SimpleNamespace(
        _use_min_latency_q_b_gemm=None,
        _q_b_proj_verified_shape=True,
        q_b_proj=lambda x: (torch.zeros(x.shape[0], 8), None),
        fused_a_gemm_backend="auto",
        num_local_heads=2,
        qk_head_dim=4,
    )


def _exec(deterministic: bool):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )


class TestQBProjMinLatencyGemm(CustomTestCase):
    def _forward(self, deterministic: bool, rows: int):
        attn = _fake_attn()
        with (
            mock.patch.object(
                deepseek_v2, "get_exec", return_value=_exec(deterministic)
            ),
            mock.patch.object(
                deepseek_v2, "fused_a_gemm_weight_eligible", return_value=True
            ),
            mock.patch.object(
                deepseek_v2,
                "linear_with_fused_a_gemm",
                side_effect=lambda layer, x, backend: torch.ones(x.shape[0], 8),
            ) as fused,
        ):
            q = deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_forward(
                attn, torch.zeros(rows, 16)
            )
        return attn._use_min_latency_q_b_gemm, fused.called, tuple(q.shape)

    def test_min_latency_gemm_used_by_default(self):
        enabled, called, shape = self._forward(deterministic=False, rows=4)
        self.assertTrue(enabled)
        self.assertTrue(called)
        self.assertEqual(shape, (4, 2, 4))

    def test_deterministic_inference_disables_min_latency_gemm(self):
        enabled, called, shape = self._forward(deterministic=True, rows=4)
        self.assertFalse(enabled)
        self.assertFalse(called)
        self.assertEqual(shape, (4, 2, 4))


if __name__ == "__main__":
    unittest.main()

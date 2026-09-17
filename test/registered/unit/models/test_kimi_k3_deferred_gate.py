"""CPU regression for the ROCm fused input projection's deferred gate contract."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKimiK3DeferredGate(unittest.TestCase):
    def test_rocm_fused_projection_honors_defer_f_b(self):
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(2, 8, generator=generator)
        qkv = torch.randn(2, 6, generator=generator)
        gate = torch.randn(2, 2, generator=generator)
        f_a = torch.randn(2, 128, generator=generator)
        beta = torch.randn(2, 12, generator=generator)
        padding = torch.zeros(2, 4)
        fused = torch.cat((qkv, gate, f_a, beta, padding), dim=-1)
        projection = Mock(return_value=fused)
        owner = SimpleNamespace(
            use_full_rank_gate=True,
            _bfa_w=torch.empty(0),
            _bfa_fa_size=128,
            _bfa_b_size=12,
            _bfa_f_b_w=torch.randn(1536, 128, generator=generator),
            _qkvgbfa_sizes=(6, 2, 128, 12, 4),
            _qkvgbfa_bs_limit=2,
            _qkvgbfa_layer=object(),
            fused_qkvg_proj=SimpleNamespace(
                quant_method=SimpleNamespace(apply=projection)
            ),
        )
        for deferred in (False, True):
            with (
                self.subTest(defer_f_b=deferred),
                patch("sglang.srt.models.kimi_k3._is_hip", True),
                patch(
                    "sglang.kernels.ops.kimi_k3.kimi_k3_tiny_gemm", wraps=F.linear
                ) as gemm,
            ):
                actual = KimiK3DeltaAttention.forward_qkvbfg_fused(
                    owner, hidden_states, defer_f_b=deferred
                )
                expected_gate = f_a if deferred else F.linear(f_a, owner._bfa_f_b_w)
                for got, expected in zip(actual, (qkv, beta, expected_gate, gate)):
                    torch.testing.assert_close(got, expected)
                self.assertEqual(gemm.call_count, 0 if deferred else 1)


if __name__ == "__main__":
    unittest.main()

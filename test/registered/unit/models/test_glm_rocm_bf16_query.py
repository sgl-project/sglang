"""Guarded query caller tests; full TP8 checkpoint adoption is qualified separately."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.utils.common import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip(), "ROCm query caller requires HIP")
class TestGlmRocmBf16Query(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers import rocm_linear_utils
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        cls.utils = rocm_linear_utils
        cls.caller = staticmethod(DeepseekV2AttentionMLA.q_b_proj_forward_rocm)

    def setUp(self):
        self.exec = SimpleNamespace(
            kernel=SimpleNamespace(bf16_gemm_backend="auto"),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        for name, value in (
            ("get_exec", self.exec),
            ("get_lora", SimpleNamespace(enable_lora=False)),
            ("get_forward", SimpleNamespace(sp_active=False)),
        ):
            patch = mock.patch.object(self.utils, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        torch.manual_seed(0)
        self.layer = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * 0.02,
                requires_grad=False,
            ),
            quant_method=self.utils.UnquantizedLinearMethod(),
            bias=None,
            gather_output=False,
        )

    def test_public_auto_numerics_and_changed_input_graph(self):
        device = self.layer.weight.device
        if not self.utils._query_auto_device_supported(device):
            self.skipTest("qualification requires gfx950/256 CU")
        if self.utils._aiter_auto_bf16_gemm is None:
            self.fail("qualified dependency is missing the public automatic GEMM")
        for m in (64, 128):
            with self.subTest(m=m):
                q = torch.randn(m, 2048, device=device, dtype=torch.bfloat16)
                for _ in range(2):
                    output = self.utils.aiter_bf16_query_gemm(self.layer, q)
                    torch.testing.assert_close(
                        output, F.linear(q, self.layer.weight), atol=0.1, rtol=0.01
                    )
                misses = self.utils._query_auto_device_supported.cache_info().misses
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.utils.aiter_bf16_query_gemm(self.layer, q)
                self.assertEqual(
                    self.utils._query_auto_device_supported.cache_info().misses, misses
                )
                for _ in range(5):
                    q.copy_(torch.randn_like(q))
                    expected_q, expected_weight = q.clone(), self.layer.weight.clone()
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(q, expected_q, atol=0, rtol=0)
                    torch.testing.assert_close(
                        self.layer.weight, expected_weight, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        output,
                        F.linear(expected_q, expected_weight),
                        atol=0.1,
                        rtol=0.01,
                    )

    def test_guard_rejection_and_native_caller_fallback(self):
        q = torch.zeros(64, 2048, device="cuda", dtype=torch.bfloat16)
        negatives = (
            (self.layer, "quant_method", object()),
            (self.layer, "bias", torch.zeros(2048, device="cuda")),
            (self.layer, "gather_output", True),
            (self.layer, "set_lora", True),
            (self.layer.weight, "is_shuffled", True),
            (self.exec.kernel, "bf16_gemm_backend", "torch"),
            (self.exec.deterministic, "enable_deterministic_inference", True),
        )
        with mock.patch.object(
            self.utils,
            "_aiter_auto_bf16_gemm",
            side_effect=AssertionError("ineligible AUTO launch"),
        ):
            for owner, name, value in negatives:
                with (
                    self.subTest(name=name),
                    mock.patch.object(owner, name, value, create=True),
                ):
                    self.assertIsNone(self.utils.aiter_bf16_query_gemm(self.layer, q))
            for invalid in (
                q[:16],
                q[:63],
                q.to(torch.float16),
                q[:, ::2],
                torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)[1:].view(
                    q.shape
                ),
            ):
                self.assertIsNone(self.utils.aiter_bf16_query_gemm(self.layer, invalid))
        native = mock.Mock(return_value=(F.linear(q, self.layer.weight), None))
        owner = SimpleNamespace(
            use_dsa=True,
            q_lora_rank=2048,
            num_local_heads=8,
            qk_head_dim=256,
            q_b_proj=native,
        )
        with (
            mock.patch("sglang.srt.models.deepseek_v2._use_aiter", True),
            mock.patch(
                "sglang.srt.models.deepseek_v2.aiter_bf16_query_gemm",
                return_value=None,
                create=True,
            ) as optional_route,
        ):
            output = self.caller(owner, q)
        optional_route.assert_called_once_with(native, q)
        native.assert_called_once_with(q)
        self.assertEqual(output.shape, (64, 8, 256))


if __name__ == "__main__":
    unittest.main()

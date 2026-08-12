import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.kimi_k3 import KimiK3MLAAttention
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestKimiK3MlaGateFusion(unittest.TestCase):
    def test_merged_projection_matches_separate_projections(self):
        torch.manual_seed(0)
        qkv_proj = SimpleNamespace(
            weight=torch.nn.Parameter(torch.randn(12, 16, dtype=torch.bfloat16)),
            quant_method=UnquantizedLinearMethod(),
        )
        g_proj = SimpleNamespace(
            weight=torch.nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16)),
            quant_method=UnquantizedLinearMethod(),
        )
        attn = SimpleNamespace(
            use_output_gate=True,
            quant_config=object(),
            fused_qkv_a_proj_with_mqa=qkv_proj,
            g_proj=g_proj,
            _qkv_a_g_proj_weight=None,
            _qkv_a_g_proj_sizes=None,
            _use_min_latency_fused_a_gemm=False,
            _gate_precomputed=None,
        )
        x = torch.randn(3, 16, dtype=torch.bfloat16)
        expected_qkv = torch.nn.functional.linear(x, qkv_proj.weight)
        expected_gate = torch.nn.functional.linear(x, g_proj.weight)

        KimiK3MLAAttention._merge_qkv_a_g_proj_weights(attn)
        qkv = KimiK3MLAAttention.prepare_qkv_latent(attn, x, None)
        gate, stream = attn._gate_precomputed

        torch.testing.assert_close(qkv, expected_qkv)
        torch.testing.assert_close(gate, expected_gate)
        self.assertIsNone(stream)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_fused_a_cuda_graph_replay(self):
        if is_hip_runtime() or get_jit_cuda_arch().major < 9:
            self.skipTest("SM90+ required")

        # K3 TP8: replicated qkv-a [2112, 7168] plus local gate [1536, 7168].
        qkv_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
            ),
            quant_method=UnquantizedLinearMethod(),
        )
        g_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.randn(1536, 7168, dtype=torch.bfloat16, device="cuda")
            ),
            quant_method=UnquantizedLinearMethod(),
        )
        attn = SimpleNamespace(
            use_output_gate=True,
            quant_config=object(),
            fused_qkv_a_proj_with_mqa=qkv_proj,
            g_proj=g_proj,
            _qkv_a_g_proj_weight=None,
            _qkv_a_g_proj_sizes=None,
            _use_min_latency_fused_a_gemm=None,
            _gate_precomputed=None,
            fused_a_gemm_backend="jit",
        )
        KimiK3MLAAttention._merge_qkv_a_g_proj_weights(attn)

        exec_config = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        for num_tokens in (1, 8, 16):
            with self.subTest(num_tokens=num_tokens):
                static_x = torch.randn(
                    num_tokens, 7168, dtype=torch.bfloat16, device="cuda"
                )
                with patch(
                    "sglang.srt.models.kimi_k3.get_exec", return_value=exec_config
                ):
                    KimiK3MLAAttention.prepare_qkv_latent(attn, static_x, None)
                self.assertTrue(attn._use_min_latency_fused_a_gemm)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    qkv = KimiK3MLAAttention.prepare_qkv_latent(attn, static_x, None)
                gate = attn._gate_precomputed[0]

                static_x.copy_(torch.randn_like(static_x))
                graph.replay()
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    qkv,
                    torch.nn.functional.linear(static_x, qkv_proj.weight),
                    rtol=1e-2,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    gate,
                    torch.nn.functional.linear(static_x, g_proj.weight),
                    rtol=1e-2,
                    atol=1e-3,
                )


if __name__ == "__main__":
    unittest.main()

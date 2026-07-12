"""DSpark unit tests that require a CUDA device or GPU-only wheels.

Small checks that cannot run in the CPU suite: cpu-vs-gpu device parity of
torch code, and backend class-attribute checks whose modules import GPU-only
wheels (sgl_kernel) at module scope.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")


class TestConfidenceMetricsDeviceParity(CustomTestCase):
    def test_on_device_accumulation_matches_cpu(self):
        from sglang.srt.speculative.dspark_components.dspark_observability import (
            PerPositionConfidenceMetrics,
        )

        torch.manual_seed(0)
        bs, gamma = 24, 4
        survival = torch.rand(bs, gamma)
        prefix_mask = (torch.rand(bs, gamma) < 0.5).to(torch.float64)

        cpu = PerPositionConfidenceMetrics(gamma=gamma, device=torch.device("cpu"))
        cpu.update(survival=survival, prefix_mask=prefix_mask)

        gpu = PerPositionConfidenceMetrics(gamma=gamma, device=DEVICE)
        gpu.update(survival=survival.cuda(), prefix_mask=prefix_mask.cuda())

        for cpu_row, gpu_row in zip(cpu.compute(), gpu.compute()):
            for key in ("ece", "auc", "brier", "pred_mean", "target_mean"):
                self.assertAlmostEqual(cpu_row[key], gpu_row[key], places=9, msg=key)


class TestRaggedVerifyGraphCapability(CustomTestCase):
    def test_base_backend_defaults_false(self):
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

        self.assertFalse(AttentionBackend.supports_ragged_verify_graph)

    def test_ragged_implementing_backends_declare_the_flag(self):
        """Every backend with a ragged-verify metadata path must opt in; a
        dropped flag silently disables ragged graphs for that backend (the
        runner falls back to eager with no other test going red)."""
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        for backend in (
            TRTLLMHAAttnBackend,
            DeepseekV4AttnBackend,
            FlashAttentionBackend,
        ):
            with self.subTest(backend=backend.__name__):
                self.assertTrue(backend.supports_ragged_verify_graph)


if __name__ == "__main__":
    unittest.main()

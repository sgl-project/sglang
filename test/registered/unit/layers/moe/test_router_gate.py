"""Unit tests for the shared MoE RouterGate layer."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.layers.moe import RouterGate
from sglang.srt.layers.moe import router_gate as rg_mod
from sglang.srt.layers.moe import tiny_router_gemm_max_tokens
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import get_context, get_exec, reset_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRouterGate(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )

    def test_bf16_compute_always_emits_fp32_logits_and_separates_close_scores(self):
        """BF16 router weights must return FP32 logits so sub-BF16-ULP differences survive."""
        with set_default_torch_dtype(torch.bfloat16):
            gate = RouterGate(
                hidden_size=8,
                num_experts=4,
                fp32_compute=False,
                has_correction_bias=True,
            )
        self.assertEqual(gate.weight.dtype, torch.bfloat16)
        self.assertEqual(gate.e_score_correction_bias.dtype, torch.float32)

        with torch.no_grad():
            gate.weight.zero_()
            gate.weight[0, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
            gate.weight[1, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
            gate.weight[1, 1] = torch.tensor(2**-10, dtype=torch.bfloat16)

        x = torch.zeros((2, 8), dtype=torch.bfloat16)
        x[:, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
        x[:, 1] = torch.tensor(1.0, dtype=torch.bfloat16)

        logits = gate(x)
        self.assertEqual(logits.dtype, torch.float32)
        # In BF16, 1.0 + 2^-10 rounds back to 1.0; FP32 accumulation keeps expert 1 strictly higher.
        self.assertGreater(logits[0, 1].item(), logits[0, 0].item())
        self.assertEqual(
            logits.to(torch.bfloat16)[0, 1], logits.to(torch.bfloat16)[0, 0]
        )

    def test_deterministic_inference_bypasses_small_m_tiny_gemm_and_hpc(self):
        """Deterministic mode must not switch kernels at small M or take non-deterministic HPC splits."""
        with set_default_torch_dtype(torch.bfloat16):
            bf16_gate = RouterGate(hidden_size=16, num_experts=8, fp32_compute=False)
        bf16_gate.tiny_router_gemm_max_tokens = 16
        with torch.no_grad():
            bf16_gate.weight.copy_(torch.randn_like(bf16_gate.weight))

        x = torch.randn((4, 16), dtype=torch.bfloat16)
        with (
            get_exec().deterministic.override(enable_deterministic_inference=True),
            patch.object(
                rg_mod,
                "tiny_gemm_bf16",
                side_effect=AssertionError(
                    "tiny_gemm_bf16 must not run in deterministic mode"
                ),
            ),
        ):
            out = bf16_gate(x)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, F.linear(x.float(), bf16_gate.weight.float()))

        with envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.override("hpc"):
            fp32_gate = RouterGate(hidden_size=4096, num_experts=256, fp32_compute=True)
        self.assertEqual(fp32_gate.hpc_kernel_min_m, 8)
        with torch.no_grad():
            fp32_gate.weight.copy_(torch.randn_like(fp32_gate.weight))
        x_large = torch.randn((12, 4096), dtype=torch.bfloat16)
        with (
            get_exec().deterministic.override(enable_deterministic_inference=True),
            patch.object(
                rg_mod,
                "linear_bf16_fp32",
                side_effect=AssertionError(
                    "HPC split GEMM must not run in deterministic mode"
                ),
            ),
        ):
            out_fp32 = fp32_gate(x_large)
        self.assertEqual(out_fp32.dtype, torch.float32)
        torch.testing.assert_close(
            out_fp32, F.linear(x_large.float(), fp32_gate.weight)
        )

    def test_custom_correction_bias_name_and_shape_contracts(self):
        """Checkpoint-specific bias names and 2D shapes are registered as parameters."""
        bailing_gate = RouterGate(
            hidden_size=8,
            num_experts=6,
            params_dtype=torch.bfloat16,
            has_correction_bias=True,
            correction_bias_name="expert_bias",
        )
        self.assertIn("expert_bias", dict(bailing_gate.named_parameters()))
        self.assertEqual(bailing_gate.expert_bias.shape, (6,))
        self.assertEqual(bailing_gate.expert_bias.dtype, torch.float32)

        ernie_gate = RouterGate(
            hidden_size=8,
            num_experts=6,
            fp32_compute=False,
            has_correction_bias=True,
            correction_bias_shape=(1, 6),
        )
        self.assertEqual(ernie_gate.e_score_correction_bias.shape, (1, 6))

        no_bias_gate = RouterGate(
            hidden_size=8,
            num_experts=6,
            has_correction_bias=False,
            correction_bias_name="expert_bias",
        )
        self.assertIsNone(no_bias_gate.expert_bias)

    def test_tiny_router_gemm_threshold_guards_device_sm_and_dtype(self):
        """Small-batch threshold enables M<=16 only for SM90+ CUDA BF16 supported shapes."""
        with (
            patch.object(rg_mod, "_is_cuda", True),
            patch.object(rg_mod, "_device_sm", 90),
            patch.object(rg_mod, "can_use_tiny_gemm", return_value=True),
        ):
            self.assertEqual(
                tiny_router_gemm_max_tokens(
                    num_experts=256, hidden_size=7168, weight_dtype=torch.bfloat16
                ),
                16,
            )
            self.assertEqual(
                tiny_router_gemm_max_tokens(
                    num_experts=256, hidden_size=7168, weight_dtype=torch.float32
                ),
                -1,
            )

        with (
            patch.object(rg_mod, "_is_cuda", True),
            patch.object(rg_mod, "_device_sm", 80),
            patch.object(rg_mod, "can_use_tiny_gemm", return_value=True),
        ):
            self.assertEqual(
                tiny_router_gemm_max_tokens(
                    num_experts=256, hidden_size=7168, weight_dtype=torch.bfloat16
                ),
                -1,
            )

    def test_params_dtype_override_and_unpublished_context_forward(self):
        """params_dtype overrides fp32_compute=False only when set, and forward works pre-publish."""
        with set_default_torch_dtype(torch.bfloat16):
            default_gate = RouterGate(
                hidden_size=8,
                num_experts=4,
                fp32_compute=False,
                params_dtype=None,
            )
            explicit_fp32_gate = RouterGate(
                hidden_size=8,
                num_experts=4,
                fp32_compute=False,
                params_dtype=torch.float32,
            )
        self.assertEqual(default_gate.weight.dtype, torch.bfloat16)
        self.assertFalse(default_gate.fp32_compute)
        self.assertEqual(explicit_fp32_gate.weight.dtype, torch.float32)
        self.assertTrue(explicit_fp32_gate.fp32_compute)

        with torch.no_grad():
            default_gate.weight.copy_(torch.randn_like(default_gate.weight))
        x_3d = torch.randn((2, 3, 8), dtype=torch.bfloat16)
        reset_context()
        self.addCleanup(
            lambda: get_context().override_server_args(model_path="dummy", device="cpu")
        )
        out_3d = default_gate(x_3d)
        self.assertEqual(out_3d.shape, (2, 3, 4))
        self.assertEqual(out_3d.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

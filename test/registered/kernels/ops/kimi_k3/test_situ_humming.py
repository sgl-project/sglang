"""SM90 component CI for the K3 SiTU fused masked quantization path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.kimi_k3 import situ_and_mul_masked_post_quant
from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BETA = 4.0
_LINEAR_BETA = 25.0


def _situ_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    return (
        _BETA
        * torch.tanh(gate / _BETA)
        * torch.sigmoid(gate)
        * _LINEAR_BETA
        * torch.tanh(up / _LINEAR_BETA)
    )


class TestKimiK3Humming(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() != 90:
            raise unittest.SkipTest("Kimi-K3 Humming requires SM90")

    def test_situ_fused_masked_quant(self):
        num_experts = 56
        max_tokens = 16
        intermediate = 3072
        num_groups = intermediate // 128

        runner = humming_runner.HummingRunnerCore(
            MoeRunnerConfig(
                num_experts=896,
                num_local_experts=num_experts,
                intermediate_size_per_partition=intermediate,
                top_k=16,
                activation="situ",
                is_gated=True,
                gemm1_alpha=_BETA,
                gemm1_clamp_limit=_LINEAR_BETA,
                gate_up_interleaved=False,
            )
        )
        runner.layer = SimpleNamespace(
            humming_metas={
                "w2": SimpleNamespace(
                    a_dtype=humming_runner.dtypes.float8e4m3,
                    input_scale_group_size=128,
                )
            }
        )

        generator = torch.Generator(device="cuda").manual_seed(1)
        gate_up = (
            torch.randn(
                num_experts,
                max_tokens,
                2 * intermediate,
                generator=generator,
                device="cuda",
            )
            * 3.0
        ).to(torch.bfloat16)
        expert_num_tokens = torch.zeros(num_experts, device="cuda", dtype=torch.int32)
        expert_num_tokens[:15] = max_tokens
        expert_num_tokens[31] = 7
        expert_num_tokens[32] = 9
        self.assertEqual(int(expert_num_tokens.sum().item()), max_tokens * 16)

        activation_output = torch.full(
            (num_experts * max_tokens, intermediate),
            float("nan"),
            device="cuda",
            dtype=torch.bfloat16,
        )
        buffers = {
            "gate_up_output": gate_up.flatten(0, 1),
            "activation_output": activation_output,
        }

        def run_poisoned_kernel(**kwargs):
            kwargs["output"].view(torch.uint8).fill_(0x7F)
            kwargs["output_scale"].fill_(float("nan"))
            return situ_and_mul_masked_post_quant(**kwargs)

        with (
            patch.object(
                runner,
                "apply_activation",
                side_effect=AssertionError("SiTU fused path was not selected"),
            ) as fallback_activation,
            patch(
                "sglang.kernels.ops.kimi_k3.situ_and_mul_masked_post_quant",
                side_effect=run_poisoned_kernel,
            ) as fused_situ,
        ):
            down_input, down_scale = runner._grouped_masked_act_quant(
                gate_up, expert_num_tokens, buffers
            )

        fused_situ.assert_called_once()
        fallback_activation.assert_not_called()
        call = fused_situ.call_args.kwargs
        self.assertIs(call["input"], gate_up)
        self.assertIs(call["masked_m"], expert_num_tokens)
        self.assertEqual(call["quant_group_size"], 128)
        self.assertEqual(call["beta"], _BETA)
        self.assertEqual(call["linear_beta"], _LINEAR_BETA)
        self.assertEqual(call["topk"], 16)
        self.assertFalse(call["scale_ue8m0"])
        self.assertFalse(call["transposed"])
        self.assertFalse(call["swizzle"])
        self.assertEqual(call["output"].data_ptr(), down_input.data_ptr())
        self.assertEqual(call["output_scale"].data_ptr(), down_scale.data_ptr())

        self.assertEqual(down_input.shape, (num_experts * max_tokens, intermediate))
        self.assertEqual(down_input.dtype, torch.float8_e4m3fn)
        self.assertEqual(down_scale.shape, (num_experts * max_tokens, num_groups))
        self.assertEqual(down_scale.dtype, torch.float32)
        self.assertTrue(down_scale.is_contiguous())
        self.assertNotEqual(down_input.data_ptr(), gate_up.data_ptr())
        self.assertTrue(bool(activation_output.isnan().all()))

        actual = down_input.view(num_experts, max_tokens, intermediate)
        actual_scale = down_scale.view(num_experts, max_tokens, num_groups)
        self.assertEqual(
            actual_scale.stride(),
            (max_tokens * num_groups, num_groups, 1),
        )
        reference = _situ_reference(gate_up)
        grouped_reference = reference.view(num_experts, max_tokens, num_groups, 128)
        expected_scale = grouped_reference.abs().amax(dim=-1).clamp_min(1e-10) / 448.0

        for expert in range(num_experts):
            valid = int(expert_num_tokens[expert])
            if valid:
                torch.testing.assert_close(
                    actual_scale[expert, :valid],
                    expected_scale[expert, :valid],
                    rtol=5e-3,
                    atol=1e-6,
                )
                expanded_scale = actual_scale[expert, :valid].repeat_interleave(
                    128, dim=-1
                )
                dequantized = actual[expert, :valid].float() * expanded_scale
                self.assertTrue(
                    bool(
                        (
                            (dequantized - reference[expert, :valid]).abs()
                            <= expanded_scale * 17.0 + 1e-4
                        ).all()
                    )
                )

            if valid < max_tokens:
                self.assertTrue(
                    bool((actual[expert, valid:].view(torch.uint8) == 0x7F).all())
                )
                self.assertTrue(bool(actual_scale[expert, valid:].isnan().all()))


if __name__ == "__main__":
    unittest.main()

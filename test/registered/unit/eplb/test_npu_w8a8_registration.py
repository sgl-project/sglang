"""Check NPU W8A8 post-load parameters exposed to EPLB, using CPU tensors."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuW8A8ExpertRegistration(unittest.TestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        # Import the model in normal framework order. Hardware probing and NZ
        # conversion are outside this CPU test's scope.
        arch_patch = patch(
            "sglang.srt.hardware_backend.npu.utils.is_npu_arch35", return_value=False
        )
        arch_patch.start()
        self.addCleanup(arch_patch.stop)
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        self.get_weights = DeepseekV2MoE.get_moe_weights
        from sglang.srt.hardware_backend.npu.quantization import moe_methods

        self.methods = moe_methods

    def experts(self, with_offsets=True):
        experts = torch.nn.Module()
        experts.num_local_experts = 5
        raw = {}
        for prefix in ("w13", "w2"):
            raw[prefix] = torch.arange(5 * 32 * 64).reshape(5, 32, 64).to(torch.int8)
            experts.register_parameter(
                prefix + "_weight",
                torch.nn.Parameter(raw[prefix].clone(), requires_grad=False),
            )
            experts.register_parameter(
                prefix + "_weight_scale",
                torch.nn.Parameter(
                    torch.arange(5 * 32).reshape(5, 32, 1).float() / 32,
                    requires_grad=False,
                ),
            )
            if with_offsets:
                experts.register_parameter(
                    prefix + "_weight_offset",
                    torch.nn.Parameter(
                        torch.arange(5 * 32).reshape(5, 32, 1).float() + 1,
                        requires_grad=False,
                    ),
                )
        experts.correction_bias = torch.nn.Parameter(torch.ones(5), requires_grad=False)
        experts.global_scale = torch.nn.Parameter(torch.ones(5), requires_grad=False)
        experts.global_scale._sglang_require_global_experts = True
        # Post-load processing uses only static helpers. Skip the constructor's
        # activation-quantizer binding so CPU CI needs no registered NPU ops.
        method = object.__new__(self.methods.NPUW8A8Int8MoEMethod)
        with (
            patch.object(
                self.methods, "npu_format_cast", side_effect=lambda tensor: tensor
            ),
            patch.object(torch.ops, "npu", SimpleNamespace()),
        ):
            for prefix in ("w13", "w2"):
                method.process_weights_after_loading(experts, prefix)
        return experts, raw

    def test_processed_weights_scales_offsets_are_routed_views(self):
        experts, raw = self.experts()
        moe = SimpleNamespace(experts=experts, num_fused_shared_experts=1)
        weights = self.get_weights(moe)
        expected_names = [
            prefix + suffix
            for prefix in ("w13", "w2")
            for suffix in ("_weight", "_weight_scale", "_weight_offset")
        ]
        self.assertEqual(len(weights), 6)
        for name, view in zip(expected_names, weights, strict=True):
            full = getattr(experts, name)
            self.assertEqual(view.shape[0], 4)
            self.assertEqual(view.data_ptr(), full.data_ptr())
            shared_before = full[-1].clone()
            if name.endswith("_weight"):
                prefix = name.removesuffix("_weight")
                torch.testing.assert_close(full, raw[prefix].transpose(1, 2))
                self.assertEqual(view.dtype, torch.int8)
            elif name.endswith("_scale"):
                self.assertEqual(view.dtype, torch.bfloat16)
                self.assertEqual(view.ndim, 2)
                torch.testing.assert_close(
                    full, (torch.arange(5 * 32).reshape(5, 32).float() / 32).bfloat16()
                )
            else:
                self.assertEqual(view.dtype, torch.float32)
                self.assertEqual(view.ndim, 2)
                torch.testing.assert_close(
                    full, torch.arange(5 * 32).reshape(5, 32).float() + 1
                )
            view.zero_()
            torch.testing.assert_close(full[-1], shared_before)
        torch.testing.assert_close(experts.correction_bias, torch.ones(5))
        torch.testing.assert_close(experts.global_scale, torch.ones(5))

    def test_optional_offsets_and_unfused_experts(self):
        experts, _ = self.experts(with_offsets=False)
        weights = self.get_weights(
            SimpleNamespace(experts=experts, num_fused_shared_experts=0)
        )
        self.assertEqual(len(weights), 4)
        self.assertTrue(all(view.shape[0] == 5 for view in weights))


if __name__ == "__main__":
    unittest.main()

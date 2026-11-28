"""Dispatch contracts for clipped ARCH22 W4A8 GMM1 fusion."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="extra-a-test-npu")

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu import utils
from sglang.srt.hardware_backend.npu.quantization import moe_methods
from sglang.srt.layers.moe.moe_runner import ascend


class TestW4A8Int8MoEFusion(unittest.TestCase):
    def test_loader_requires_w4a8_binding(self):
        def w8a8_op(x, weight, weight_scale, x_scale, group_list, swiglu_limit):
            pass

        def w4a8_op(*args, weight_assist_matrix=None):
            pass

        op_name = "sgl_grouped_matmul_swiglu_quant"
        package = SimpleNamespace()
        operators = SimpleNamespace()
        loader = utils.get_grouped_matmul_swiglu_quant_w4a8
        self.addCleanup(loader.cache_clear)
        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel_npu": package,
                    "sgl_kernel_npu.operators.moe": operators,
                },
            ),
            patch.object(utils, "is_npu_arch22") as arch22,
        ):
            for arch, ops, op, expected in (
                (False, [op_name], w4a8_op, None),
                (True, [], w4a8_op, None),
                (True, [op_name], w8a8_op, None),
                (True, [op_name], w4a8_op, w4a8_op),
            ):
                with self.subTest(arch=arch, ops=ops, op=op):
                    loader.cache_clear()
                    arch22.return_value = arch
                    operators.grouped_matmul_swiglu_quant = op
                    package._manifest = {"third_party": [{"ops": ops}]}
                    self.assertIs(loader(), expected)

    def setUp(self):
        kernel = moe_methods.NPUW4A8Int8MoEMethod
        self.w13 = kernel.__new__(kernel)
        self.w13.apply = Mock(return_value=torch.zeros(3, 64, dtype=torch.bfloat16))
        self.w2 = kernel.__new__(kernel)
        self.w2.apply = Mock(return_value=torch.zeros(3, 32, dtype=torch.bfloat16))
        self.x = torch.zeros(3, 32, dtype=torch.int8)
        self.scale = torch.ones(3)
        self.w13.hidden_states_quantizer = Mock(return_value=(self.x, self.scale))
        self.info = ascend.AscendQuantInfo(
            w13_weight=torch.empty(3, 32, 8, dtype=torch.int32),
            w2_weight=torch.empty(0),
            w13_weight_scale=torch.empty(3, 2, 64, dtype=torch.int64),
            w13_weight_bias=torch.zeros(3, 64),
            w13_scale_bias=torch.ones(3, 64),
        )
        self.qact = torch.empty(3, 32, dtype=torch.int8)
        self.qscale = torch.full((3,), 0.5)
        self.fused = Mock(return_value=(self.qact, self.qscale))
        self.runner = ascend.AscendRunnerCore.__new__(ascend.AscendRunnerCore)
        self.runner.config = SimpleNamespace(
            layer=SimpleNamespace(w13_kernel=self.w13, w2_kernel=self.w2),
            activation="silu",
            is_gated=True,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            use_tp_all_gather_activation=False,
            swiglu_limit=10.0,
        )
        self.runner.activation = Mock()
        self.runner.activation._apply_activation.return_value = self.qact, self.qscale
        self.loader = self.enterContext(
            patch.object(
                moe_methods,
                "get_grouped_matmul_swiglu_quant_w4a8",
                return_value=self.fused,
            )
        )
        self.npu_format = self.enterContext(
            patch.object(torch.ops.npu, "get_npu_format", return_value=29, create=True)
        )

    def test_fused_runner_passes_compensation_and_preserves_gmm2_metadata(self):
        for kind, groups, x, scale in (
            (1, [2, 0, 1], self.x, self.scale.reshape(3, 1)),
            (0, [2, 2, 3], self.x.to(torch.bfloat16), None),
        ):
            with self.subTest(group_list_type=kind):
                self.w13.hidden_states_quantizer.reset_mock()
                group_list = torch.tensor(groups, dtype=torch.int64)
                runner_input = ascend.AscendRunnerInput(x, scale, group_list, kind)
                out = self.runner.run(runner_input, self.info, {})
                args, kwargs = self.fused.call_args
                self.assertIs(args[0], self.x)
                self.assertIs(args[1], self.info.w13_weight)
                self.assertIs(args[2], self.info.w13_weight_scale)
                torch.testing.assert_close(args[3], self.scale)
                torch.testing.assert_close(args[4], torch.tensor([2, 2, 3]))
                self.assertEqual(args[5], 10.0)
                self.assertIs(kwargs["weight_assist_matrix"], self.info.w13_scale_bias)
                if scale is None:
                    self.w13.hidden_states_quantizer.assert_called_once_with(x)
                else:
                    self.w13.hidden_states_quantizer.assert_not_called()
                args, kwargs = self.w2.apply.call_args
                self.assertIs(args[1], self.qact)
                self.assertIs(args[2], group_list)
                self.assertIs(kwargs["pertoken_scale"], self.qscale)
                self.assertEqual(kwargs["group_list_type"], kind)
                self.assertIs(out.hidden_states, self.w2.apply.return_value)
        self.w13.apply.assert_not_called()
        self.runner.activation._apply_activation.assert_not_called()

    def test_runner_preserves_unfused_path_for_incompatible_inputs(self):
        runner_input = ascend.AscendRunnerInput(
            self.x, self.scale, torch.tensor([2, 0, 1]), 1
        )
        for context in (
            patch.object(self.loader, "return_value", None),
            patch.object(self.npu_format, "return_value", 2),
            patch.object(self.info, "w13_weight_scale", None),
            patch.object(runner_input, "hidden_states", self.x[:0]),
            patch.object(self.runner.config, "use_tp_all_gather_activation", True),
            patch.object(self.runner.config, "swiglu_limit", None),
        ):
            with self.subTest(context=context.attribute), context:
                self.w13.apply.reset_mock()
                self.runner.activation._apply_activation.reset_mock()
                self.runner.run(runner_input, self.info, {})
                self.w13.apply.assert_called_once()
                self.runner.activation._apply_activation.assert_called_once_with(
                    self.w13.apply.return_value
                )
        self.fused.assert_not_called()


if __name__ == "__main__":
    unittest.main()

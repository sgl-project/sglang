"""Selection and argument contracts for the optional arch22 GMM package."""

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
from sglang.test.test_utils import CustomTestCase


class TestA3GroupedMatmulSwigluQuant(CustomTestCase):
    def setUp(self):
        utils.get_grouped_matmul_swiglu_quant.cache_clear()
        self.addCleanup(utils.get_grouped_matmul_swiglu_quant.cache_clear)

    def test_loader_requires_arch22_and_a_bundled_kernel(self):
        package = SimpleNamespace(_manifest={})
        fused = Mock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "sgl_kernel_npu": package,
                    "sgl_kernel_npu.operators.moe": SimpleNamespace(
                        grouped_matmul_swiglu_quant=fused
                    ),
                },
            ),
            patch.object(utils, "is_npu_arch22") as arch22,
        ):
            for supported, ops, expected in (
                (False, ["sgl_grouped_matmul_swiglu_quant"], None),
                (True, [], None),
                (True, ["sgl_grouped_matmul_swiglu_quant"], fused),
            ):
                with self.subTest(arch22=supported, ops=ops):
                    utils.get_grouped_matmul_swiglu_quant.cache_clear()
                    arch22.return_value = supported
                    package._manifest = {"third_party": [{"ops": ops}]}
                    self.assertIs(utils.get_grouped_matmul_swiglu_quant(), expected)

    def test_gmm_gate_preserves_unsupported_paths(self):
        kernel = moe_methods.NPUW8A8Int8MoEMethod.__new__(
            moe_methods.NPUW8A8Int8MoEMethod
        )
        config = SimpleNamespace(
            activation="silu",
            is_gated=True,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
            use_tp_all_gather_activation=False,
            swiglu_limit=10.0,
        )
        info = SimpleNamespace(w13_weight=object())
        x = torch.zeros(8, 16, dtype=torch.int8)
        with (
            patch.object(
                ascend, "get_grouped_matmul_swiglu_quant", return_value=None
            ) as fused,
            patch.object(
                torch.ops.npu, "get_npu_format", return_value=29, create=True
            ) as storage_format,
        ):
            self.assertFalse(ascend._uses_fused_int8_gmm1(kernel, config, info, x, 0))
            fused.return_value = Mock()
            self.assertTrue(ascend._uses_fused_int8_gmm1(kernel, config, info, x, 0))
            storage_format.return_value = 2
            self.assertFalse(ascend._uses_fused_int8_gmm1(kernel, config, info, x, 0))
            storage_format.return_value = 29
            w4 = moe_methods.NPUW4A8Int8MoEMethod.__new__(
                moe_methods.NPUW4A8Int8MoEMethod
            )
            self.assertFalse(ascend._uses_fused_int8_gmm1(w4, config, info, x, 0))

    def test_gmm_converts_counts_and_reuses_existing_quantization(self):
        kernel = moe_methods.NPUW8A8Int8MoEMethod.__new__(
            moe_methods.NPUW8A8Int8MoEMethod
        )
        x = torch.zeros(3, 16, dtype=torch.int8)
        scale = torch.ones(3)
        kernel.hidden_states_quantizer = Mock(return_value=(x, scale))
        info = SimpleNamespace(w13_weight=object(), w13_weight_scale=object())
        fused = Mock(return_value=(x, scale))
        with patch.object(
            moe_methods,
            "get_grouped_matmul_swiglu_quant",
            return_value=fused,
        ):
            for kind, counts in ((1, [2, 0, 1]), (0, [2, 2, 3])):
                kernel.apply_fused_gmm1_swiglu(
                    info, x, torch.tensor(counts), scale, kind, 10.0
                )
                args = fused.call_args.args
                self.assertIs(args[0], x)
                self.assertIs(args[3], scale)
                torch.testing.assert_close(args[4], torch.tensor([2, 2, 3]))
                self.assertEqual(args[5], 10.0)
            kernel.hidden_states_quantizer.assert_not_called()
            bf16 = x.to(torch.bfloat16)
            kernel.apply_fused_gmm1_swiglu(
                info, bf16, torch.tensor([2, 0, 1]), None, 1, None
            )
            kernel.hidden_states_quantizer.assert_called_once_with(bf16)
            self.assertEqual(fused.call_args.args[5], 0.0)


if __name__ == "__main__":
    unittest.main()

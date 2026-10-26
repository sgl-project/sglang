"""Check online W4A8 weight policy without changing MXFP8 activation quantization."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import linear_method_npu, moe_methods
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestOnlineMXFP4WeightPolicy(unittest.TestCase):
    # torch_npu's scalar-type enum, not torch.float4_e2m1fn_x2.
    FP4_DTYPE = 296

    def setUp(self):
        patcher = patch.object(
            torch.Tensor, "is_npu", property(lambda _: True), create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_weight_policy(self, quantize):
        quantize.assert_called_once_with(
            ANY,
            dst_type=self.FP4_DTYPE,
            round_mode="rint",
            scale_alg=2,
            dst_type_max=7.25,
        )

    def test_dense_weights_use_policy_and_keep_packed_layout(self):
        for dtype in (torch.bfloat16, torch.float16):
            for flat_scale in (False, True):
                with self.subTest(dtype=dtype, flat_scale=flat_scale):
                    layer = torch.nn.Module()
                    layer.weight = torch.nn.Parameter(
                        torch.ones(16, 128, dtype=dtype), requires_grad=False
                    )
                    packed = torch.arange(16 * 64).to(torch.uint8).reshape(16, 64)
                    scale = torch.arange(16 * 4).to(torch.uint8).reshape(16, 2, 2)
                    quantize = Mock(
                        return_value=(
                            packed,
                            scale.flatten(-2) if flat_scale else scale,
                        )
                    )
                    with (
                        patch.object(
                            torch.ops,
                            "npu",
                            SimpleNamespace(npu_dynamic_mx_quant=quantize),
                        ),
                        patch.object(
                            linear_method_npu,
                            "_get_float4_e2m1fn_x2_dtype",
                            return_value=self.FP4_DTYPE,
                        ),
                        patch.object(
                            linear_method_npu,
                            "npu_format_cast",
                            side_effect=lambda weight, *args, **kwargs: weight,
                        ),
                    ):
                        linear_method_npu.NPUMXFP4W4A8LinearMethod()._quantize_weights(
                            layer
                        )
                    self.assert_weight_policy(quantize)
                    self.assertEqual(quantize.call_args.args[0].dtype, dtype)
                    torch.testing.assert_close(layer.weight, packed.transpose(-1, -2))
                    torch.testing.assert_close(
                        layer.weight_scale, scale.transpose(0, 1)
                    )

    def test_moe_weights_use_same_policy_for_both_projections(self):
        for prefix in ("w13", "w2"):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(prefix=prefix, dtype=dtype):
                    source = torch.ones(2, 16, 128, dtype=dtype)
                    result = (
                        torch.zeros(2, 16, 64, dtype=torch.uint8),
                        torch.zeros(2, 16, 2, 2, dtype=torch.uint8),
                    )
                    quantize = Mock(return_value=result)
                    with patch.object(
                        torch.ops, "npu", SimpleNamespace(npu_dynamic_mx_quant=quantize)
                    ):
                        actual = (
                            moe_methods.NPUW4A8MXFP4MoEMethod._quantize_weight_online(
                                source, prefix, self.FP4_DTYPE
                            )
                        )
                    self.assertIs(actual, result)
                    self.assert_weight_policy(quantize)
                    self.assertIs(quantize.call_args.args[0], source)

    def test_dense_mxfp8_activation_policy_is_unchanged(self):
        layer = SimpleNamespace(
            weight=torch.zeros(32, 16, dtype=torch.uint8),
            weight_scale=torch.zeros(1, 16, 2, dtype=torch.uint8),
        )
        quantize = Mock(
            return_value=(
                torch.zeros(3, 64, dtype=torch.float8_e4m3fn),
                torch.zeros(3, 1, 2, dtype=torch.uint8),
            )
        )
        matmul = Mock(return_value=torch.zeros(3, 16, dtype=torch.bfloat16))
        with (
            patch.object(
                torch.ops,
                "npu",
                SimpleNamespace(npu_dynamic_mx_quant=quantize, npu_quant_matmul=matmul),
            ),
            patch.object(
                linear_method_npu,
                "_get_float4_e2m1fn_x2_dtype",
                return_value=self.FP4_DTYPE,
            ),
            patch.object(
                linear_method_npu, "_get_float8_e8m0fnu_dtype", return_value=torch.uint8
            ),
        ):
            output = linear_method_npu.NPUMXFP4W4A8LinearMethod().apply(
                layer, torch.ones(1, 3, 64, dtype=torch.bfloat16)
            )
        quantize.assert_called_once_with(ANY, dst_type=torch.float8_e4m3fn)
        self.assertEqual(output.shape, (1, 3, 16))
        self.assertEqual(matmul.call_args.kwargs["x2_dtype"], self.FP4_DTYPE)

    def test_moe_mxfp8_activation_policy_is_unchanged(self):
        info = SimpleNamespace(
            w2_weight=torch.zeros(2, 32, 16, dtype=torch.uint8),
            w2_weight_scale=torch.zeros(2, 1, 16, 2, dtype=torch.uint8),
        )
        method = moe_methods.NPUW4A8MXFP4MoEMethod()
        with patch.object(moe_methods, "w4a8_mxfp_gmm") as gmm:
            method.apply(
                info,
                torch.ones(3, 64, dtype=torch.bfloat16),
                torch.tensor([1, 3]),
                None,
                torch.bfloat16,
                "w2",
                0,
            )
        self.assertEqual(
            gmm.call_args.kwargs["dynamic_quant_kwargs"],
            {"dst_type": torch.float8_e4m3fn},
        )


if __name__ == "__main__":
    unittest.main()

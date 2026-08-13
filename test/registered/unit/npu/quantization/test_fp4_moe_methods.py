import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.quantization import fp4_moe_methods
from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
    _pair_pack_mxfp_act_scale,
    _set_fp4_dispatcher_output_dtype,
    _swiglu_limit_mx_quant,
    _w4a8_mxfp_gmm,
    npu_apply_without_routing_weights_w4a4_mxfp,
)
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher


class TestFp4DispatcherOutputDtype(unittest.TestCase):
    @staticmethod
    def _ascend_tp_layer():
        dispatcher = AscendTPDispatcher.__new__(AscendTPDispatcher)
        dispatcher.set_quant_config = MagicMock()
        return SimpleNamespace(dispatcher=dispatcher)

    def test_a5_ascend_tp_uses_mxfp8(self):
        layer = self._ascend_tp_layer()
        with patch.object(
            fp4_moe_methods, "is_npu_before_atlas_a5", return_value=False
        ):
            _set_fp4_dispatcher_output_dtype(layer)

        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "mxfp8"}
        )

    def test_deepep_stays_bf16(self):
        layer = SimpleNamespace(dispatcher=MagicMock())
        with patch.object(
            fp4_moe_methods, "is_npu_before_atlas_a5", return_value=False
        ):
            _set_fp4_dispatcher_output_dtype(layer)

        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )

    def test_pre_a5_ascend_tp_stays_bf16(self):
        layer = self._ascend_tp_layer()
        with patch.object(fp4_moe_methods, "is_npu_before_atlas_a5", return_value=True):
            _set_fp4_dispatcher_output_dtype(layer)

        layer.dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )


class TestW4A8MxfpGmmInputScale(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(2, 64)
        self.input_scale = torch.ones(2, 1, 2)
        self.weight = torch.empty(2, 64, 32, dtype=torch.uint8)
        self.weight_scale = torch.ones(2, 1, 32, 2, dtype=torch.uint8)
        self.group_list = torch.tensor([1, 1], dtype=torch.int64)

    def _call_gmm(self, input_scale):
        return _w4a8_mxfp_gmm(
            input=self.input,
            input_scale=input_scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            group_list_type=1,
            group_list=self.group_list,
            output_dtype=torch.bfloat16,
        )

    def test_supplied_scale_skips_dynamic_quant(self):
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu, "npu_dynamic_mx_quant", create=True
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(self.input_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        call_kwargs = grouped_matmul.call_args.kwargs
        self.assertIs(call_kwargs["per_token_scale"][0], self.input_scale)
        self.assertIs(call_kwargs["group_list"], self.group_list)

    def test_missing_scale_uses_dynamic_quant(self):
        quantized = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(quantized, quantized_scale),
                create=True,
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._call_gmm(None)

        dynamic_quant.assert_called_once()
        self.assertIs(output, expected)
        self.assertIs(
            grouped_matmul.call_args.kwargs["per_token_scale"][0], quantized_scale
        )

    def test_flat_dispatch_scale_is_pair_packed_as_a_view(self):
        flat_scale = torch.arange(8).view(2, 4)
        packed_scale = _pair_pack_mxfp_act_scale(flat_scale)

        self.assertEqual(packed_scale.shape, (2, 2, 2))
        self.assertEqual(packed_scale.data_ptr(), flat_scale.data_ptr())


class TestW4A8MxfpGmmChain(unittest.TestCase):
    def test_gmm2_reuses_swiglu_quant_scale(self):
        gate_up = torch.randn(2, 64)
        quantized = torch.empty(2, 32, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        expected = torch.randn(2, 64)
        layer = SimpleNamespace(
            w13_weight=MagicMock(),
            w13_weight_scale_inv=MagicMock(),
            w2_weight=MagicMock(),
            w2_weight_scale_inv=MagicMock(),
            moe_runner_config=SimpleNamespace(swiglu_limit=7.0),
        )

        with (
            patch.object(
                fp4_moe_methods,
                "w4a4_mxfp_gmm_npu",
                side_effect=[gate_up, expected],
            ) as gmm,
            patch.object(
                fp4_moe_methods,
                "_swiglu_limit_mx_quant",
                return_value=(quantized, quantized_scale),
            ),
        ):
            output = npu_apply_without_routing_weights_w4a4_mxfp(
                layer,
                torch.randn(2, 32),
                torch.ones(2, 1, 2),
                group_list_type=1,
                group_list=torch.tensor([1, 1], dtype=torch.int64),
                output_dtype=torch.bfloat16,
            )

        self.assertIs(output, expected)
        self.assertIs(gmm.call_args_list[1].kwargs["input"], quantized)
        self.assertIs(gmm.call_args_list[1].kwargs["input_scale"], quantized_scale)

    def test_swiglu_quant_fallback_preserves_asymmetric_clamp(self):
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        activated = torch.randn(1, 2)
        quantized = torch.empty(1, 2, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(1, 1, 2)

        with (
            patch.object(
                fp4_moe_methods,
                "_get_swiglu_group_quant_op",
                return_value=None,
            ),
            patch.object(
                torch.ops.npu,
                "npu_swiglu",
                return_value=activated,
                create=True,
            ) as swiglu,
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(quantized, quantized_scale),
                create=True,
            ) as dynamic_quant,
        ):
            output = _swiglu_limit_mx_quant(gate_up, 7.0)

        self.assertTrue(
            torch.equal(
                swiglu.call_args.args[0], torch.tensor([[7.0, -9.0, 7.0, -7.0]])
            )
        )
        self.assertIs(dynamic_quant.call_args.args[0], activated)
        self.assertEqual(output, (quantized, quantized_scale))

    def test_swiglu_group_quant_fuses_middle_path(self):
        gate_up = torch.randn(2, 128)
        quantized = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        unused_origin = torch.empty(2, 64, dtype=torch.bfloat16)
        fused_op = MagicMock(return_value=(quantized, quantized_scale, unused_origin))

        with (
            patch.object(
                fp4_moe_methods,
                "_get_swiglu_group_quant_op",
                return_value=fused_op,
            ),
            patch.object(torch.ops.npu, "npu_swiglu", create=True) as separate_swiglu,
            patch.object(
                torch.ops.npu, "npu_dynamic_mx_quant", create=True
            ) as separate_quant,
        ):
            output = _swiglu_limit_mx_quant(gate_up, 7.0)

        self.assertEqual(output, (quantized, quantized_scale))
        separate_swiglu.assert_not_called()
        separate_quant.assert_not_called()
        call_kwargs = fused_op.call_args.kwargs
        self.assertIs(fused_op.call_args.args[0], gate_up)
        self.assertEqual(call_kwargs["dst_type"], torch.float8_e4m3fn)
        self.assertEqual(call_kwargs["quant_mode"], 2)
        self.assertEqual(call_kwargs["clamp_value"], 7.0)


if __name__ == "__main__":
    unittest.main()

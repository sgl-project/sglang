"""CPU-safe tests for the Ascend fused norm+quant capability selection, the
--online-quantization grammar, and the quantized router-gate schemes.

No torch_npu install or NPU device is required: operator probes are mocked
and only pure selection logic runs.
"""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401  (canonical import entry)
from sglang.srt.hardware_backend.npu.layernorm_quant import (
    NormQuantizedReadout,
    can_fuse_norm_quant,
    parse_online_quantization,
)
from sglang.srt.hardware_backend.npu.moe.contracts import INT8, MXFP8
from sglang.srt.hardware_backend.npu.moe.norm_gate import (
    GATE_SCHEME_DECLINED,
    QuantizedMoEGateMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOnlineQuantizationGrammar(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_online_quantization(None), {})
        self.assertEqual(parse_online_quantization(""), {})

    def test_single_pair(self):
        self.assertEqual(
            parse_online_quantization("gate:w8a8_mxfp8"), {"gate": "W8A8_MXFP8"}
        )
        self.assertEqual(
            parse_online_quantization("gate:w4a8_mxfp"), {"gate": "W4A8_MXFP"}
        )
        self.assertEqual(
            parse_online_quantization("GATE:W8A8_DYNAMIC"), {"gate": "W8A8_DYNAMIC"}
        )

    def test_multiple_pairs(self):
        parsed = parse_online_quantization(
            "gate:w8a8_mxfp8, moe:w4a8_mxfp ,qkv:w4a4_mxfp4"
        )
        self.assertEqual(
            parsed,
            {
                "gate": "W8A8_MXFP8",
                "moe": "W4A8_MXFP",
                "qkv": "W4A4_MXFP4",
            },
        )

    def test_all_expands(self):
        parsed = parse_online_quantization("all:w8a8_dynamic")
        self.assertEqual(parsed["gate"], "W8A8_DYNAMIC")
        self.assertEqual(parsed["dense"], "W8A8_DYNAMIC")
        self.assertNotIn("all", parsed)

    def test_rejects_unknown_target(self):
        with self.assertRaises(ValueError):
            parse_online_quantization("router:w8a8_mxfp8")

    def test_rejects_missing_scheme(self):
        with self.assertRaises(ValueError):
            parse_online_quantization("gate")

    def test_rejects_unknown_scheme(self):
        with self.assertRaises(ValueError):
            parse_online_quantization("gate:int4")


class TestNormQuantSelection(unittest.TestCase):
    def test_no_consumers(self):
        contract, reason = can_fuse_norm_quant([])
        self.assertIsNone(contract)
        self.assertIn("no quantized consumer", reason)

    def test_float_consumer_declines(self):
        contract, reason = can_fuse_norm_quant([MXFP8, None])
        self.assertIsNone(contract)
        self.assertIn("floating-point", reason)

    def test_disagreeing_consumers_decline(self):
        contract, reason = can_fuse_norm_quant([MXFP8, INT8])
        self.assertIsNone(contract)
        self.assertIn("disagree", reason)

    def test_probe_gate(self):
        # All consumers agree on MXFP8 but the device probe fails closed on
        # CPU (no NPU, no fused ABI): the selection must decline, not guess.
        contract, reason = can_fuse_norm_quant([MXFP8, MXFP8])
        self.assertIsNone(contract)
        self.assertIn("cannot emit", reason)

    def test_readout_dataclass_is_frozen(self):
        operand = torch.zeros(2, 64, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 1, 2, dtype=torch.uint8)
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand

        q = QuantizedOperand(operand, scale, MXFP8, (2, 64))
        residual = torch.zeros(2, 64)
        readout = NormQuantizedReadout(q, residual)
        self.assertIs(readout.normalized_float, None)
        self.assertIs(readout.operand, q)
        self.assertIs(readout.updated_residual, residual)


class TestGateSchemes(unittest.TestCase):
    def test_supported_schemes(self):
        method = QuantizedMoEGateMethod("W8A8_MXFP8")
        self.assertEqual(method.contract, MXFP8)
        self.assertEqual(method.output_dtype, torch.float32)

    def test_int8_scheme_uses_bf16_output(self):
        method = QuantizedMoEGateMethod("W8A8_DYNAMIC")
        self.assertEqual(method.contract, INT8)
        self.assertEqual(method.output_dtype, torch.bfloat16)

    def test_declined_schemes_fail_closed(self):
        for scheme in ("W4A8_MXFP", "W4A4_MXFP4", "W4A4_DYNAMIC"):
            self.assertIn(scheme, GATE_SCHEME_DECLINED)
            with self.assertRaises(ValueError):
                QuantizedMoEGateMethod(scheme)

    def test_unknown_scheme_rejected(self):
        with self.assertRaises(ValueError):
            QuantizedMoEGateMethod("FP16")

    def test_create_weights_is_noop(self):
        method = QuantizedMoEGateMethod("W8A8_MXFP8")
        self.assertIsNone(method.create_weights(object()))


class TestNpuNormQuantKernelDispatch(unittest.TestCase):
    def test_npu_format_selects_npu_kernel(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _norm_quant_kernel,
            _update_and_read_residual_npu,
        )

        for fmt in ("npu_mxfp8", "npu_mxfp4", "npu_int8", "npu_int4"):
            kernel = _norm_quant_kernel(fmt)
            self.assertIs(kernel.func, _update_and_read_residual_npu)
            self.assertEqual(kernel.keywords, {"quant_format": fmt})

    def test_non_npu_format_unchanged(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _norm_quant_kernel,
            _update_and_read_residual_plain,
        )

        self.assertIs(_norm_quant_kernel(""), _update_and_read_residual_plain)


class TestBoundaryOperandMoeForward(unittest.TestCase):
    """The GLM MoE block routes a boundary QuantizedOperand to the fused path
    without touching the tensor paths (CPU, dispatcher mocked)."""

    def _make_block(self):
        from sglang.srt.models import glm4_moe

        block = glm4_moe.Glm4MoeSparseMoeBlock.__new__(glm4_moe.Glm4MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block._enable_a2a_moe = False
        block._fuse_npu_tp_shared = False
        block._fuse_npu_tp_routed_scale = False
        block.routed_scaling_factor = 2.5
        block.gate = object()
        block.topk = None
        block.experts = None
        return block

    def test_operand_routes_to_quantized_path(self):
        block = self._make_block()
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand
        from sglang.srt.models import glm4_moe

        operand = QuantizedOperand(
            torch.zeros(2, 64, dtype=torch.float8_e4m3fn),
            torch.zeros(2, 1, 2, dtype=torch.uint8),
            MXFP8,
            (2, 64),
        )
        with (
            patch.object(glm4_moe, "_is_npu", True),
            patch.object(
                block, "forward_normal_quantized", return_value="out"
            ) as fused,
        ):
            result = block.forward(operand)
        fused.assert_called_once_with(operand)
        self.assertEqual(result, "out")


class TestFusedNormABI(unittest.TestCase):
    def test_integer_five_outputs_and_int4_dtype(self):
        from sglang.srt.hardware_backend.npu import layernorm_quant as nq
        from sglang.srt.hardware_backend.npu.moe.contracts import INT4

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        q = torch.zeros(2, 8, dtype=torch.int32)
        scale = torch.ones(2, dtype=torch.float32)
        with patch.object(nq, "native_int_norm_op") as factory:
            factory.return_value.return_value = (q, None, residual, scale, None)
            readout = nq.native_add_rmsnorm_quant(x, residual, x[0], 1e-6, INT4)
        self.assertIs(readout.operand.data, q)
        self.assertIs(readout.updated_residual, residual)
        self.assertEqual(
            factory.return_value.call_args.kwargs["y_dtype"], torch.quint4x2
        )
        self.assertEqual(
            factory.return_value.call_args.kwargs["output_mask"], [True, False]
        )

    def test_residual_free_mx_uses_three_output_operator(self):
        from sglang.srt.hardware_backend.npu import layernorm_quant as nq

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        q = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(2, 1, 2, dtype=torch.uint8)
        with patch.object(nq, "native_norm_no_residual_op") as factory:
            factory.return_value.return_value = (q, scale, None)
            readout = nq.native_add_rmsnorm_quant(x, None, x[0], 1e-6, MXFP8)
        self.assertIs(readout.operand.data, q)
        self.assertIsNone(readout.updated_residual)

    def test_boundary_keeps_type_effective_gamma_and_add_order(self):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.npu import layernorm_quant as nq
        from sglang.srt.hardware_backend.npu.moe.contracts import QuantizedOperand
        from sglang.srt.layers.layer_boundary.residual.add_norm import (
            _update_and_read_residual_npu,
        )

        x = torch.zeros(2, 64, dtype=torch.bfloat16)
        residual = torch.ones_like(x)
        post = torch.full_like(x, 2)
        norm = SimpleNamespace(
            weight=x[0], gemma_weight=residual[0], variance_epsilon=1e-6
        )
        operand = QuantizedOperand(
            torch.zeros_like(x, dtype=torch.int8), torch.ones(2), INT8, (2, 64)
        )
        with patch.object(
            nq,
            "native_add_rmsnorm_quant",
            return_value=nq.NormQuantizedReadout(operand, post),
        ) as fused:
            output, updated = _update_and_read_residual_npu(
                "npu_int8", norm, x, residual, post
            )
        self.assertIs(output, operand)
        self.assertIs(updated, post)
        torch.testing.assert_close(fused.call_args.args[2], norm.gemma_weight)
        torch.testing.assert_close(fused.call_args.args[1], residual + post)

    def test_packed_routing_storage_is_integer_not_fp4(self):
        from sglang.srt.hardware_backend.npu.moe.contracts import INT4
        from sglang.srt.hardware_backend.npu.moe.init_routing import _packed_payload

        bytes_ = torch.arange(64, dtype=torch.uint8).reshape(2, 32)
        packed, scale = _packed_payload(bytes_, torch.ones(2), (2, 64), INT4)
        self.assertEqual(packed.dtype, torch.int32)
        self.assertEqual(packed.shape, (2, 8))
        self.assertTrue(torch.equal(packed.view(torch.uint8), bytes_))

    def test_explicit_online_targets_override_all(self):
        self.assertEqual(
            parse_online_quantization("all:w8a8_mxfp8,gate:w8a8_dynamic")["gate"],
            "W8A8_DYNAMIC",
        )
        with self.assertRaises(ValueError):
            parse_online_quantization("gate:w8a8_mxfp8,gate:w8a8_dynamic")


class TestOfflineRouterLoading(unittest.TestCase):
    def test_custom_gate_uses_checkpoint_scheme_and_preserves_bias(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from sglang.srt.hardware_backend.npu.moe.norm_gate import (
            configure_online_gate,
            gate_input_contract,
        )
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUW8A8Int8DynamicLinearMethod,
        )
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimLinearMethod,
        )

        gate = torch.nn.Module()
        gate.weight = torch.nn.Parameter(torch.empty(4, 64))
        gate.e_score_correction_bias = torch.nn.Parameter(
            torch.arange(4, dtype=torch.float32)
        )
        correction = gate.e_score_correction_bias
        config = Mock()
        config.get_name.return_value = "modelslim"
        config.resolve_projection_scheme.return_value = "W8A8_DYNAMIC"
        config.linear_scheme_types.return_value = {"W8A8_DYNAMIC": object}
        config.get_linear_scheme.return_value = SimpleNamespace(
            kernel=NPUW8A8Int8DynamicLinearMethod()
        )
        with patch.object(ModelSlimLinearMethod, "create_weights") as create:
            self.assertFalse(
                configure_online_gate(gate, "model.layers.0.mlp.gate", config)
            )
        create.assert_called_once()
        self.assertIs(gate.e_score_correction_bias, correction)
        self.assertEqual(gate_input_contract(gate), INT8)

    def test_descriptor_resolution_is_per_layer(self):
        from unittest.mock import Mock

        from sglang.srt.hardware_backend.npu.moe.norm_gate import offline_gate_config

        config = Mock()
        config.get_name.return_value = "modelslim"
        config.resolve_projection_scheme.side_effect = lambda prefix: (
            "FLOAT" if ".0." in prefix else "W8A8_MXFP8"
        )
        config.linear_scheme_types.return_value = {"W8A8_MXFP8": object}
        self.assertIsNone(offline_gate_config(config, "model.layers.0.mlp.gate"))
        self.assertIs(offline_gate_config(config, "model.layers.1.mlp.gate"), config)


class TestOnlineSharedFallback(unittest.TestCase):
    def test_online_mxfp8_weight_buffers_do_not_use_offline_views(self):
        from types import SimpleNamespace

        from sglang.srt.hardware_backend.npu.moe.norm_gate import OnlineLinearMethod
        from sglang.srt.hardware_backend.npu.moe.tp_fusion import shared_gmm1_mode
        from sglang.srt.layers.activation import SiluAndMul

        linear = SimpleNamespace(
            quant_method=OnlineLinearMethod("W8A8_MXFP8", torch.bfloat16), bias=None
        )
        mlp = SimpleNamespace(
            gate_up_proj=linear, down_proj=linear, act_fn=SiluAndMul()
        )
        self.assertEqual(shared_gmm1_mode(mlp), "baseline")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe import tp_fusion
from sglang.srt.hardware_backend.npu.moe.contracts import (
    INT8,
    MXFP4,
    MXFP8,
    QuantizedOperand,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUUnquantMoEMethod,
    NPUW4A4Int4MoEMethod,
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8Int8MoEMethod,
    NPUW8A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore, AscendRunnerInput
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
    AscendTPCombineInput,
    AscendTPDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendTPOptimizations(CustomTestCase):
    def test_mxfp4_dense_and_moe_agree_at_rounding_ties(self):
        from sglang.srt.hardware_backend.npu.quantization import linear_method_npu

        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        expected = torch.tensor([0, 1, 1, 2, 2, 4, 4])
        x = torch.cat((values, -values)).repeat(5)[:64].reshape(1, 64).bfloat16()
        expected = (
            torch.cat((expected, -expected)).repeat(5)[:64].reshape(1, 64).bfloat16()
        )

        def quantize(value, *, round_mode="rint", **kwargs):
            distances = (value.float().abs().unsqueeze(-1) - levels).abs()
            codes = torch.arange(8)
            tie_priority = (codes % 2) if round_mode == "rint" else -codes
            encoded = (distances + tie_priority * 1e-4).argmin(-1).to(torch.uint8)
            encoded |= (value < 0).to(torch.uint8) * 8
            packed = encoded[:, ::2] | (encoded[:, 1::2] << 4)
            return packed, torch.full((1, 1, 2), 127, dtype=torch.uint8)

        def decode(data):
            codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
            return (
                levels[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
            ).bfloat16()

        with (
            patch.object(torch.ops.npu, "npu_dynamic_mx_quant", quantize, create=True),
            patch.object(
                torch.ops.npu,
                "npu_quant_matmul",
                lambda data, *a, **kw: decode(data),
                create=True,
            ),
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                lambda *, x, **kw: [decode(x[0])],
                create=True,
            ),
            patch.object(
                linear_method_npu, "_get_float4_e2m1fn_x2_dtype", return_value=296
            ),
            patch.object(
                linear_method_npu, "_get_float8_e8m0fnu_dtype", return_value=torch.uint8
            ),
            patch(
                "sglang.srt.hardware_backend.npu.quantization.moe_methods._get_float4_e2m1fn_x2_dtype",
                return_value=296,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.quantization.moe_methods._require_e8m0_dtype",
                return_value=torch.uint8,
            ),
        ):
            layer = SimpleNamespace(weight=torch.empty(1), weight_scale=torch.empty(1))
            dense = linear_method_npu.NPUSingleLevelMXFP4OfflineLinearMethod().apply(
                layer, x
            )
            kernel = NPUW4A4MXFP4MoEMethod()
            info = SimpleNamespace(
                w13_weight=torch.empty(1, 1, 1), w13_weight_scale=torch.empty(1)
            )
            routed = kernel.apply(
                info, x, torch.tensor([1]), None, torch.bfloat16, "w13", 1
            )
        torch.testing.assert_close(dense, expected, atol=0, rtol=0)
        torch.testing.assert_close(routed, expected, atol=0, rtol=0)
        self.assertEqual(kernel.hidden_states_quantizer.contract, MXFP4)

    def test_unquantized_w13_mxfp8_w2_keeps_activation_and_quantizes_w2_input(self):
        layer = SimpleNamespace(
            w13_kernel=NPUUnquantMoEMethod(), w2_kernel=NPUMXFP8MoEMethod("w2")
        )
        runner = AscendRunnerCore(MoeRunnerConfig(layer=layer))
        x = torch.linspace(-1, 1, 128, dtype=torch.bfloat16).view(2, 64)
        identity = torch.eye(64, dtype=torch.bfloat16)
        info = SimpleNamespace(
            w13_weight=torch.cat((identity, identity), dim=0).unsqueeze(0),
            w2_weight=identity.unsqueeze(0),
            w2_weight_scale=torch.ones((1, 1, 64, 2), dtype=torch.uint8),
        )
        quantizations = []

        def quantize(value, **kwargs):
            quantizations.append(value.clone())
            return value, torch.ones((2, 1, 2), dtype=torch.uint8)

        def grouped_matmul(*, x, weight, per_token_scale=None, **kwargs):
            if weight[0].shape[-1] == 64:
                self.assertIsNotNone(per_token_scale[0])
            return [x[0] @ weight[0][0]]

        def swiglu(value):
            gate, up = value.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

        inputs = AscendRunnerInput(x, None, torch.tensor([2]), 1)
        with (
            patch.object(torch.ops.npu, "npu_dynamic_mx_quant", quantize, create=True),
            patch.object(
                torch.ops.npu, "npu_grouped_matmul", grouped_matmul, create=True
            ),
            patch.object(torch.ops.npu, "npu_swiglu", swiglu, create=True),
        ):
            result = runner.run(inputs, info, {})
        expected = torch.nn.functional.silu(x) * x
        torch.testing.assert_close(result.hidden_states, expected)
        self.assertEqual(len(quantizations), 1)
        torch.testing.assert_close(quantizations[0], expected)

    def test_w2_compensation_matches_rank_local_k_partition(self):
        bias = torch.arange(2 * 3 * 16, dtype=torch.float32).square().view(2, 3, 16)
        for tp_size in (1, 2, 4, 8, 16):
            reduced = []
            for rank, shard in enumerate(bias.chunk(tp_size, dim=-1)):
                layer = torch.nn.Module()
                layer.moe_tp_size, layer.moe_tp_rank = tp_size, rank
                layer.w2_scale_bias = torch.nn.Parameter(
                    bias.clone(), requires_grad=False
                )
                parameter = layer.w2_scale_bias
                NPUW4A8Int8MoEMethod._update_bias(layer, "w2")
                torch.testing.assert_close(
                    layer.w2_scale_bias, shard.sum(-1), atol=0, rtol=0
                )
                self.assertIs(layer.w2_scale_bias, parameter)
                reduced.append(layer.w2_scale_bias)
            torch.testing.assert_close(
                torch.stack(reduced).sum(0), bias.sum(-1), atol=0, rtol=0
            )
        layer.moe_tp_size = 3
        layer.w2_scale_bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "divisible"):
            NPUW4A8Int8MoEMethod._update_bias(layer, "w2")

    def test_shared_eligibility_compares_each_projection_and_expert(self):
        config = SimpleNamespace(
            n_shared_experts=1, n_routed_experts=2, moe_intermediate_size=64
        )
        description = {}
        for proj, scheme in (
            ("gate_proj", "W4A8_MXFP"),
            ("up_proj", "W4A8_MXFP"),
            ("down_proj", "W8A8_DYNAMIC"),
        ):
            for path in ("shared_experts", "experts.0", "experts.1"):
                description[f"model.layers.0.mlp.{path}.{proj}.weight"] = scheme
        quant = ModelSlimConfig(description)
        description = quant.quant_description
        parallel = SimpleNamespace(
            moe_ep_size=1,
            moe_dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            enable_prefill_cp=False,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch("sglang.srt.runtime_context.get_parallel", return_value=parallel)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.runtime_context.get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(enable_eplb=False, enable_waterfill=False)
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.moe.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                )
            )
            stack.enter_context(
                patch("sglang.srt.layers.moe.utils.is_sbo_enabled", return_value=False)
            )
            stack.enter_context(
                patch("sglang.srt.layers.moe.utils.is_tbo_enabled", return_value=False)
            )
            self.assertIsNone(tp_fusion.tp_fused_shared_expert_reason(config, quant))
            for scheme in ("W4A4_DYNAMIC", "W4A8_DYNAMIC", "W8A8_DYNAMIC"):
                for key in description:
                    description[key] = scheme
                self.assertIsNone(
                    tp_fusion.tp_fused_shared_expert_reason(config, quant)
                )
            description["model.layers.0.mlp.experts.1.down_proj.weight"] = "W4A4"
            self.assertIsNotNone(tp_fusion.tp_fused_shared_expert_reason(config, quant))
            parallel.moe_ep_size = 2
            self.assertIsNotNone(tp_fusion.tp_fused_shared_expert_reason(config, None))
        self.assertEqual(tp_fusion.shared_gmm1_mode(), "baseline")

    def test_projection_aliases_preserve_mixed_precision(self):
        description = {
            f"language_model.model.layers.0.block_sparse_moe.{path}.{proj}.weight": scheme
            for path in ("shared_expert", "experts.0")
            for proj, scheme in (
                ("w1", "W4A4_DYNAMIC"),
                ("w3", "W4A4_DYNAMIC"),
                ("w2", "W8A8_DYNAMIC"),
            )
        }
        quant = ModelSlimConfig(description)
        prefix = "model.layers.0.mlp.shared_expert"
        self.assertEqual(
            quant.resolve_projection_scheme(prefix + ".gate_proj"), "W4A4_DYNAMIC"
        )
        self.assertEqual(
            quant.resolve_projection_scheme(prefix + ".down_proj"), "W8A8_DYNAMIC"
        )

    def test_modelslim_mixed_float_and_integer_groups_allocate_independently(self):
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimFusedMoEMethod,
        )

        for w13, w2 in (
            ("FLOAT", "W8A8_DYNAMIC"),
            ("W8A8_DYNAMIC", "FLOAT"),
            ("W4A4_DYNAMIC", "W8A8_DYNAMIC"),
        ):
            description = {
                f"model.layers.0.mlp.experts.0.{proj}.weight": scheme
                for proj, scheme in (
                    ("gate_proj", w13),
                    ("up_proj", w13),
                    ("down_proj", w2),
                )
            }
            quant = ModelSlimConfig(description)
            layer = torch.nn.Module()
            layer.moe_runner_config = MoeRunnerConfig(params_dtype=torch.bfloat16)
            with patch.object(torch.ops.npu, "npu_dynamic_quant", create=True):
                layer.w13_scheme, layer.w2_scheme = quant.get_moe_scheme(
                    layer, "model.layers.0.mlp.experts"
                )
            ModelSlimFusedMoEMethod(quant).create_weights(
                layer, 3, 64, 32, torch.bfloat16
            )
            self.assertEqual(
                layer.w13_weight.dtype, torch.bfloat16 if w13 == "FLOAT" else torch.int8
            )
            self.assertEqual(
                layer.w2_weight.dtype, torch.bfloat16 if w2 == "FLOAT" else torch.int8
            )
            self.assertEqual(layer.w13_weight.shape[0], 3)
            self.assertEqual(layer.w2_weight.shape[0], 3)
            self.assertEqual(layer.w13_weight_scale is None, w13 == "FLOAT")
            self.assertEqual(layer.w2_weight_scale is None, w2 == "FLOAT")

    def test_unsupported_integer_shared_linear_cannot_load_as_bf16(self):
        from sglang.srt.layers.linear import LinearBase

        prefix = "model.layers.0.mlp.shared_expert.down_proj"
        quant = ModelSlimConfig({prefix + ".weight": "W4A8_DYNAMIC"})
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        with (
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(moe_ep_size=1),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "has no dense NPU method"):
                quant.get_quant_method(layer, prefix)

    def test_routing_rejects_incompatible_operand_before_launch(self):
        dispatcher = AscendTPDispatcher(MoeRunnerConfig(num_experts=2, top_k=1))
        x = torch.ones((2, 64), dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), torch.empty(2, 2)
        )
        operand = QuantizedOperand(
            x.to(torch.int8), torch.ones(2), INT8, tuple(x.shape)
        )
        with self.assertRaisesRegex(ValueError, "Incompatible NPU operand"):
            dispatcher.dispatch(x, topk, pre_quant_input=operand)
        with self.assertRaisesRegex(ValueError, "Incompatible NPU operand"):
            operand.validate(MXFP8)

    def test_missing_shared_side_tensor_load_fails_before_prepack(self):
        layer = torch.nn.Module()
        layer.dispatcher = AscendTPDispatcher(MoeRunnerConfig(num_experts=3, top_k=2))
        layer.moe_ep_size = 1
        layer.num_fused_shared_experts = 1
        layer.num_local_experts = 3
        for prefix in ("w13", "w2"):
            weight = torch.nn.Parameter(torch.ones(3, 4, 4), requires_grad=False)
            weight._npu_shared_loaded_shards = {"w1", "w3", "w2"}
            layer.register_parameter(prefix + "_weight", weight)
            scale = torch.nn.Parameter(torch.ones(3, 4), requires_grad=False)
            layer.register_parameter(prefix + "_weight_scale", scale)
        with self.assertRaisesRegex(ValueError, "w13_weight_scale.*missing checkpoint"):
            tp_fusion.validate_tp_expert_weights(layer, prepared=False)
        for _, param in layer.named_parameters():
            param._npu_shared_loaded_shards = {"w1", "w3", "w2"}
        tp_fusion.validate_tp_expert_weights(layer, prepared=False)

    def test_int8_reuse_routes_scales_without_mxfp_reinterpretation(self):
        from sglang.srt.hardware_backend.npu.moe.init_routing import (
            NPUMoEInitRouting_v2,
        )

        x = torch.arange(12, dtype=torch.int8).view(3, 4)
        scale = torch.tensor([0.125, 2.0, 0.5])
        ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32)

        def route(data, expert_ids, *, scale, quant_mode, **kwargs):
            self.assertEqual(quant_mode, -1)
            order = expert_ids.flatten().argsort(stable=True)
            inverse = order.argsort().to(torch.int32)
            return (
                data.repeat_interleave(2, 0)[order],
                inverse,
                torch.tensor([3, 3]),
                scale.repeat_interleave(2)[order],
            )

        with patch.object(torch.ops.npu, "npu_moe_init_routing_v2", route, create=True):
            routed, inverse, _, routed_scale = NPUMoEInitRouting_v2(1)._init_routing(
                x, ids, 2, 2, input_scale=scale
            )
        expected = (x.float() * scale[:, None]).repeat_interleave(2, 0)
        torch.testing.assert_close(
            (routed.float() * routed_scale[:, None])[inverse.long()], expected
        )

    def test_integer_w2_keeps_gelu_and_clamp_semantics(self):
        from sglang.srt.hardware_backend.npu.moe.activation import (
            NPUGeluAndMul,
            NPUSwigluStepAndMul,
        )

        with patch.object(torch.ops.npu, "npu_dynamic_quant", create=True):
            for cls in (NPUW4A8Int8MoEMethod, NPUW8A8Int8MoEMethod):
                layer = SimpleNamespace(
                    w13_kernel=NPUUnquantMoEMethod(), w2_kernel=cls()
                )
                runner = AscendRunnerCore(
                    MoeRunnerConfig(layer=layer, activation="gelu")
                )
                self.assertIsInstance(runner.activation, NPUGeluAndMul)
                runner = AscendRunnerCore(
                    MoeRunnerConfig(layer=layer, gemm1_clamp_limit=2.0)
                )
                self.assertIsInstance(runner.activation, NPUSwigluStepAndMul)
                value = torch.tensor([[10.0, -4.0, 3.0, -5.0]])
                result, scale = runner.activation._apply_activation(value)
                torch.testing.assert_close(
                    result,
                    torch.nn.functional.silu(value[:, :2]).clamp(max=2)
                    * value[:, 2:].clamp(-2, 2),
                )
                self.assertIsNone(scale)
                runner = AscendRunnerCore(
                    MoeRunnerConfig(layer=layer, swiglu_limit=2.0)
                )
                result, scale = runner.activation._apply_activation(value)
                expected = torch.nn.functional.silu(value[:, :2].clamp(max=2)) * value[
                    :, 2:
                ].clamp(-2, 2)
                torch.testing.assert_close(result, expected)
                self.assertIsNone(scale)

    def test_integer_w4a4_packing_variants_preserve_signed_nibbles(self):
        from sglang.srt.environ import envs

        kernel = NPUW4A4Int4MoEMethod.__new__(NPUW4A4Int4MoEMethod)
        original = (torch.arange(256) % 16 - 8).to(torch.int8).view(2, 16, 8)
        packed = kernel._w4a4_pack_int4(original)
        results = []
        for new_packing in (False, True):
            layer = torch.nn.Module()
            layer.w2_weight = torch.nn.Parameter(
                (packed if new_packing else original).clone(), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.ones(2, 16, 1), requires_grad=False
            )
            with (
                patch.object(
                    envs.SGLANG_NPU_W4A4_NEW_PACKING, "get", return_value=new_packing
                ),
                patch(
                    "sglang.srt.hardware_backend.npu.quantization.moe_methods.npu_format_cast",
                    side_effect=lambda x: x,
                ),
                patch.object(torch.Tensor, "npu", lambda x: x, create=True),
            ):
                kernel.process_weights_after_loading(layer, "w2")
            results.append(layer.w2_weight)
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
        storage = results[0].view(torch.int8)
        low, high = storage & 15, (storage >> 4) & 15
        decoded = torch.stack((low, high), dim=-1).reshape(2, 8, 16).transpose(1, 2)
        decoded = torch.where(decoded >= 8, decoded - 16, decoded)
        torch.testing.assert_close(decoded, original, atol=0, rtol=0)

    def test_topk_scales_only_routed_weights_before_appending_shared(self):
        path = Path(tp_fusion.__file__).with_name("topk.py")
        spec = importlib.util.spec_from_file_location("_test_npu_topk", path)
        module = importlib.util.module_from_spec(spec)
        # The optional wheel owns only the normalization operation here.
        l1 = SimpleNamespace(l1_norm=lambda x: x / x.sum(-1, keepdim=True))
        with patch.dict(sys.modules, {"sgl_kernel_npu.norm.l1_norm": l1}):
            spec.loader.exec_module(module)

        def topk_op(logits, k):
            weights, ids = torch.softmax(logits, dim=-1).topk(k, dim=-1)
            return weights, ids, None

        logits = torch.tensor([[1.0, -2.0, 3.0, 0.5], [0.0, 1.0, 2.0, 3.0]])
        config = TopKConfig(
            top_k=3,
            num_fused_shared_experts=1,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=True,
            fused_shared_experts_scaling_factor=0.75,
        )
        with (
            patch.object(
                torch.ops.npu, "npu_moe_gating_top_k_softmax", topk_op, create=True
            ),
            patch.object(
                module,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(on_select_experts=lambda **kwargs: None),
            ),
            patch.object(module, "capture_routed_experts_if_allowed"),
        ):
            output = module.fused_topk_npu(torch.zeros(2, 64), logits, config)
        selected, ids = torch.softmax(logits, -1).topk(2, dim=-1)
        expected = torch.cat(
            (selected / selected.sum(-1, keepdim=True) * 2.5, torch.full((2, 1), 0.75)),
            dim=-1,
        )
        torch.testing.assert_close(output.topk_weights, expected)
        torch.testing.assert_close(
            output.topk_ids[:, :2], ids.to(output.topk_ids.dtype)
        )
        self.assertTrue(torch.all(output.topk_ids[:, 2] == 4))

    def test_shared_output_is_finalized_once_and_dispatch_state_is_cleared(self):
        dispatcher = AscendTPDispatcher(MoeRunnerConfig(num_experts=2, top_k=1))
        hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
        weights = torch.tensor([[0.25], [0.75]])
        ids = torch.tensor([[0], [1]], dtype=torch.int32)
        routing = (
            hidden,
            torch.arange(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int64),
            None,
        )
        shared = torch.full_like(hidden, 3)
        with patch.object(dispatcher.init, "_init_routing", return_value=routing):
            dispatcher.dispatch(
                hidden, StandardTopKOutput(weights, ids, torch.empty(2, 2))
            )

        def finalize(x, *, scales, skip1, **kwargs):
            return x * scales + skip1

        with patch.object(
            torch.ops.npu, "npu_moe_finalize_routing", finalize, create=True
        ):
            result = dispatcher.combine(AscendTPCombineInput(hidden, shared))
        torch.testing.assert_close(result, hidden * weights + shared)
        with self.assertRaisesRegex(RuntimeError, "before dispatch"):
            dispatcher.combine(AscendTPCombineInput(hidden, shared))


if __name__ == "__main__":
    unittest.main()

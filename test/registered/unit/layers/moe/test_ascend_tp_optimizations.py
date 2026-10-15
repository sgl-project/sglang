import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe import tp_fusion
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUUnquantMoEMethod,
    NPUW4A8Int8MoEMethod,
)
from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore, AscendRunnerInput
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.ascend_tp import (
    AscendTPCombineInput,
    AscendTPDispatcher,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendTPOptimizations(CustomTestCase):
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
        for tp_size in (1, 2, 4, 8):
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
            ("down_proj", "W8A8"),
        ):
            for path in ("shared_experts", "experts.0", "experts.1"):
                description[f"model.layers.0.mlp.{path}.{proj}.weight"] = scheme
        quant = SimpleNamespace(
            get_name=lambda: "modelslim", quant_description=description
        )
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
            description["model.layers.0.mlp.experts.1.down_proj.weight"] = "W4A4"
            self.assertIsNotNone(tp_fusion.tp_fused_shared_expert_reason(config, quant))
            parallel.moe_ep_size = 2
            self.assertIsNotNone(tp_fusion.tp_fused_shared_expert_reason(config, None))
        self.assertEqual(tp_fusion.shared_gmm1_mode(), "baseline")

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

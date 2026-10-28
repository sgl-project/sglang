"""Model routing policies retain equivalent native selectors and materialized fallbacks."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.moe.topk as topk_module
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestNativeMoeRoutingPolicy(CustomTestCase):
    def setUp(self):
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                tp_group=None,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_tp_group=None,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            )
        )
        self.enterContext(get_model().override(shared_experts=SharedExpertsConfig()))
        self.enterContext(patch.object(Fp8MoEMethod, "create_weights"))
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_device",
                return_value="cpu",
            )
        )
        self.quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        self.logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        self.bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])

    def _mimo(self, *, bias=True, renormalize=True):
        from sglang.srt.models.mimo_v2 import MiMoV2MoE

        config = SimpleNamespace(
            n_routed_experts=8,
            num_experts_per_tok=2,
            hidden_size=4,
            moe_intermediate_size=128,
            hidden_act="silu",
            topk_method="noaux_tc" if bias else "group_limited_greedy",
            norm_topk_prob=renormalize,
            n_group=2,
            topk_group=1,
            scoring_func="sigmoid" if bias else "softmax",
        )
        block = MiMoV2MoE(config, layer_id=0, quant_config=self.quant_config)
        if bias:
            block.gate.e_score_correction_bias.data.copy_(self.bias)
        return block

    def _bailing(self, *, grouped, bias, renormalize=True):
        from sglang.srt.models.bailing_moe import BailingMoESparseMoeBlock

        config = SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=None,
            hidden_size=4,
            moe_intermediate_size=128,
            hidden_act="silu",
            n_group=2 if grouped else 0,
            topk_group=1 if grouped else 0,
            norm_topk_prob=renormalize,
            routed_scaling_factor=2.5,
            moe_router_enable_expert_bias=bias,
        )
        block = BailingMoESparseMoeBlock(
            config=config, layer_id=0, quant_config=self.quant_config
        )
        if bias:
            block.gate.expert_bias.data.copy_(self.bias)
        return block

    def _cohere(self, *, renormalize):
        from sglang.srt.models.cohere2_moe import Cohere2MoeSparseMoeBlock

        return Cohere2MoeSparseMoeBlock(
            config=SimpleNamespace(
                num_experts=8,
                num_experts_per_tok=2,
                hidden_size=4,
                intermediate_size=128,
                expert_selection_fn="sigmoid",
                norm_topk_prob=renormalize,
            ),
            layer_id=0,
            quant_config=self.quant_config,
        )

    @staticmethod
    def _weighted_experts(ids, weights, hidden_states):
        values = ids.float() + 1
        return (values * weights).sum(dim=-1, keepdim=True).expand_as(hidden_states)

    def _native_output(self, block):
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

        hidden_states = torch.ones((1, 4))
        self.assertEqual(
            block.topk.topk_config.output_format, TopKOutputFormat.BYPASSED
        )
        routing = block.topk.forward_cuda(hidden_states, self.logits)
        policy = block.experts.routing_method_type
        self.assertEqual(block.experts.moe_runner_config.routing_method_type, policy)

        def native_kernel(*, routing_logits, routing_method_type, output, **kwargs):
            self.assertEqual(routing_method_type, policy)
            if routing_method_type == RoutingMethodType.DeepSeekV3:
                scores = routing_logits.sigmoid()
                choices = scores + kwargs["routing_bias"]
                group_scores = choices.reshape(1, 2, 4).topk(2, dim=-1).values.sum(-1)
                selected = group_scores.argmax(-1).item()
                ids = choices[:, selected * 4 : (selected + 1) * 4].topk(2).indices
                ids = ids + selected * 4
                weights = scores.gather(1, ids)
            elif routing_method_type == RoutingMethodType.SigmoidRenorm:
                weights, ids = routing_logits.sigmoid().topk(2)
            else:
                self.assertEqual(
                    routing_method_type, RoutingMethodType.RenormalizeNaive
                )
                weights, ids = routing_logits.softmax(-1).topk(2)
            weights = weights / weights.sum(-1, keepdim=True)
            weights = weights * kwargs["routed_scaling_factor"]
            output.copy_(self._weighted_experts(ids, weights, hidden_states))

        quant_info = flashinfer_trtllm.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty(8, 256, 4),
            w2_weight=torch.empty(8, 4, 128),
            global_num_experts=8,
            local_expert_offset=0,
            local_num_experts=8,
            intermediate_size=128,
            routing_method_type=policy,
            block_quant=True,
            weight_block_k=128,
            w13_weight_scale_inv=torch.ones(8, 2, 1),
            w2_weight_scale_inv=torch.ones(8, 1, 1),
        )
        provider = ModuleType("flashinfer.fused_moe")
        provider.Fp8QuantizationType = SimpleNamespace(MxFp8=1, DeepSeekFp8=0)
        with (
            patch.dict(sys.modules, {"flashinfer.fused_moe": provider}),
            patch.object(
                flashinfer_trtllm,
                "per_token_group_quant_fp8",
                return_value=(hidden_states, torch.ones((1, 1))),
            ),
            patch.object(
                flashinfer_trtllm,
                "trtllm_fp8_block_scale_moe_out_wrapper",
                side_effect=native_kernel,
            ),
        ):
            return flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp8(
                StandardDispatchOutput(hidden_states, None, routing),
                quant_info,
                block.experts.moe_runner_config,
            ).hidden_states

    def test_mimo_grouped_bias_retains_native_selection_and_weights(self):
        """Equivalent grouped routing must not incur materialized dispatch."""
        block = self._mimo()
        output = self._native_output(block)
        weights = self.logits[:, 4:6].sigmoid()
        weights = weights / weights.sum(-1, keepdim=True)
        expected = self._weighted_experts(torch.tensor([[4, 5]]), weights, output)
        torch.testing.assert_close(output, expected)
        for bias, renormalize in ((False, True), (True, False)):
            with self.subTest(bias=bias, renormalize=renormalize):
                fallback = self._mimo(bias=bias, renormalize=renormalize)
                self.assertEqual(
                    fallback.topk.topk_config.output_format, TopKOutputFormat.STANDARD
                )

    def test_bailing_native_subsets_preserve_nonunit_scale(self):
        """Native declarations must match the configured router, including its factor."""
        for grouped, bias, ids in ((True, True, [4, 5]), (False, False, [0, 1])):
            with self.subTest(grouped=grouped):
                block = self._bailing(grouped=grouped, bias=bias)
                output = self._native_output(block)
                scores = self.logits.sigmoid() if bias else self.logits.softmax(-1)
                indices = torch.tensor([ids])
                weights = scores.gather(1, indices)
                weights = weights / weights.sum(-1, keepdim=True) * 2.5
                torch.testing.assert_close(
                    output, self._weighted_experts(indices, weights, output)
                )
        for grouped, bias, renormalize in (
            (True, False, True),
            (False, True, True),
            (False, False, False),
        ):
            with self.subTest(grouped=grouped, bias=bias, renormalize=renormalize):
                fallback = self._bailing(
                    grouped=grouped, bias=bias, renormalize=renormalize
                )
                self.assertEqual(
                    fallback.topk.topk_config.output_format, TopKOutputFormat.STANDARD
                )

    def test_cohere_normalized_sigmoid_keeps_native_and_finite_fallback(self):
        """Sigmoid normalization keeps the native policy and finite zero-score fallback."""
        block = self._cohere(renormalize=True)
        output = self._native_output(block)
        weights = self.logits[:, :2].sigmoid()
        weights = weights / weights.sum(-1, keepdim=True)
        torch.testing.assert_close(
            output, self._weighted_experts(torch.tensor([[0, 1]]), weights, output)
        )
        with (
            get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON),
            patch.object(
                topk_module,
                "fused_topk_native",
                side_effect=topk_module.fused_topk_torch_native,
            ),
        ):
            fallback = self._cohere(renormalize=True)
            routing = fallback.topk.forward_native(
                torch.ones((1, 4)), torch.full((1, 8), -1000.0)
            )
        self.assertTrue(torch.isfinite(routing.topk_weights).all().item())
        torch.testing.assert_close(routing.topk_weights, torch.zeros((1, 2)))
        fallback = self._cohere(renormalize=False)
        self.assertEqual(
            fallback.topk.topk_config.output_format, TopKOutputFormat.STANDARD
        )
        routing = fallback.topk.forward_native(torch.ones((1, 4)), self.logits)
        expected = self.logits.sigmoid().topk(2).values.sort(-1).values
        torch.testing.assert_close(routing.topk_weights.sort(-1).values, expected)

    def test_deepseek_dual_stream_graph_rejects_materialized_routing(self):
        """The logits-only graph cannot bypass a router requiring materialization."""
        from sglang.srt.models import deepseek_v2

        block = object.__new__(deepseek_v2.DeepseekV2MoE)
        torch.nn.Module.__init__(block)
        block.alt_stream = object()
        block.num_fused_shared_experts = 0
        block.shared_experts = torch.nn.Identity()
        block.experts = SimpleNamespace(
            use_flashinfer_trtllm_moe=True,
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
        )
        block._enable_a2a_moe = False
        block._fuse_shared_experts_inside_sbo = False
        block.topk = SimpleNamespace(
            topk_config=SimpleNamespace(output_format=TopKOutputFormat.BYPASSED)
        )
        with (
            patch.object(deepseek_v2, "_enable_pcg_dsv2_dual_stream", True),
            patch.object(
                deepseek_v2, "is_in_tc_piecewise_cuda_graph", return_value=True
            ),
        ):
            self.assertTrue(block._can_dual_stream_graph(torch.ones((1, 4))))
            block.topk.topk_config.output_format = TopKOutputFormat.STANDARD
            self.assertFalse(block._can_dual_stream_graph(torch.ones((1, 4))))


if __name__ == "__main__":
    unittest.main()

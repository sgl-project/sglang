"""Model routing policies retain equivalent native selectors and materialized fallbacks."""

import itertools
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

    def _implicit_bailing_policy(
        self, kind, *, grouped=True, bias=True, renormalize=True, block_routing=False
    ):
        from sglang.srt.models.bailing_moe_linear import BailingMoE as LinearBailingMoE
        from sglang.srt.models.bailing_moe_v3 import BailingMoE
        from sglang.srt.models.llada2 import LLaDA2MoeSparseMoeBlock

        factory = {
            "v3": BailingMoE,
            "linear": LinearBailingMoE,
            "llada2": LLaDA2MoeSparseMoeBlock,
        }[kind]
        config = SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=None if kind == "llada2" else 0,
            hidden_size=4,
            moe_intermediate_size=128,
            hidden_act="silu",
            n_group=2 if grouped else 0,
            topk_group=1 if grouped else 0,
            norm_topk_prob=renormalize,
            routed_scaling_factor=2.5,
            moe_router_enable_expert_bias=bias,
            score_function="sigmoid" if bias else "softmax",
            expert_capacity=4 if block_routing else None,
            block_size=2,
        )
        result = factory(config=config, layer_id=0, quant_config=self.quant_config)
        if bias:
            result.gate.expert_bias.data.copy_(self.bias)
        return result

    def test_implicit_fp8_policies_reach_the_actual_model_bindings(self):
        """Existing FP8 defaults remain native only for equivalent model routers."""
        for kind in ("v3", "linear", "llada2"):
            with self.subTest(kind=kind):
                block = self._implicit_bailing_policy(kind)
                output = self._native_output(block)
                scores = self.logits[:, 4:6].sigmoid()
                weights = scores / scores.sum(-1, keepdim=True) * 2.5
                torch.testing.assert_close(
                    output,
                    self._weighted_experts(torch.tensor([[4, 5]]), weights, output),
                )
            for grouped, bias, renormalize in (
                (False, False, True),
                (True, False, True),
                (True, True, False),
            ):
                with self.subTest(
                    kind=kind, grouped=grouped, bias=bias, renormalize=renormalize
                ):
                    block = self._implicit_bailing_policy(
                        kind, grouped=grouped, bias=bias, renormalize=renormalize
                    )
                    self.assertEqual(
                        block.topk.topk_config.output_format, TopKOutputFormat.STANDARD
                    )
        with self.assertRaisesRegex(
            ValueError, "block routing does not support.*flashinfer_trtllm"
        ):
            self._implicit_bailing_policy("llada2", block_routing=True)

    @staticmethod
    def _weighted_experts(ids, weights, hidden_states):
        values = ids.float() + 1
        return (values * weights).sum(dim=-1, keepdim=True).expand_as(hidden_states)

    def _native_output(self, block, *, block_quant=True):
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

        hidden_states = torch.ones((1, 4))
        self.assertEqual(
            block.topk.topk_config.output_format, TopKOutputFormat.BYPASSED
        )
        routing = block.topk.forward_cuda(hidden_states, self.logits)
        policy = block.experts.routing_method_type
        self.assertEqual(block.experts.moe_runner_config.routing_method_type, policy)

        def native_kernel(
            *, routing_logits, routing_method_type, output=None, **kwargs
        ):
            self.assertEqual(routing_method_type, policy)
            if not block_quant:
                self.assertEqual(routing_logits.dtype, torch.bfloat16)
                self.assertFalse(kwargs["use_routing_scales_on_input"])
            if routing_method_type == RoutingMethodType.DeepSeekV3:
                scores = routing_logits.float().sigmoid()
                choices = scores + kwargs["routing_bias"]
                group_scores = choices.reshape(1, 2, 4).topk(2, dim=-1).values.sum(-1)
                selected = group_scores.argmax(-1).item()
                ids = choices[:, selected * 4 : (selected + 1) * 4].topk(2).indices
                ids = ids + selected * 4
                weights = scores.gather(1, ids)
            elif routing_method_type == RoutingMethodType.SigmoidRenorm:
                weights, ids = routing_logits.float().sigmoid().topk(2)
            else:
                self.assertEqual(
                    routing_method_type, RoutingMethodType.RenormalizeNaive
                )
                weights, ids = routing_logits.float().softmax(-1).topk(2)
            weights = weights / weights.sum(-1, keepdim=True)
            weights = weights * kwargs["routed_scaling_factor"]
            result = self._weighted_experts(ids, weights, hidden_states)
            if output is None:
                return result.to(torch.bfloat16)
            output.copy_(result)

        quant_info = flashinfer_trtllm.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty(8, 256, 4),
            w2_weight=torch.empty(8, 4, 128),
            global_num_experts=8,
            local_expert_offset=0,
            local_num_experts=8,
            intermediate_size=128,
            routing_method_type=policy,
            block_quant=block_quant,
            weight_block_k=128,
            w13_weight_scale_inv=torch.ones(8, 2, 1),
            w2_weight_scale_inv=torch.ones(8, 1, 1),
            w13_input_scale=torch.ones(1),
            output1_scales_scalar=torch.ones(8),
            output1_scales_gate_scalar=torch.ones(8),
            output2_scales_scalar=torch.ones(8),
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
            patch.object(
                flashinfer_trtllm,
                "scaled_fp8_quant",
                return_value=(hidden_states, torch.ones(1)),
            ),
            patch.object(
                flashinfer_trtllm,
                "trtllm_fp8_per_tensor_scale_moe_wrapper",
                side_effect=native_kernel,
            ),
        ):
            return flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp8(
                StandardDispatchOutput(hidden_states, None, routing),
                quant_info,
                block.experts.moe_runner_config,
            ).hidden_states

    def test_unpadded_per_tensor_fp8_keeps_the_logits_policy_and_scale(self):
        """The routed adapter must not alter an admitted logits-only invocation."""
        block = self._implicit_bailing_policy("v3")
        output = self._native_output(block, block_quant=False)
        scores = self.logits[:, 4:6].sigmoid()
        weights = scores / scores.sum(-1, keepdim=True) * 2.5
        expected = self._weighted_experts(torch.tensor([[4, 5]]), weights, output).to(
            torch.bfloat16
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

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


def _custom_router(hidden_states, gating_output, topk, renormalize):
    raise AssertionError("never routed in these tests")


def _deepseek_v3_route(
    logits, bias, *, top_k, num_groups, topk_group, renormalize=True
):
    """TRT-LLM DeepSeekV3 routing: groups ranked by their top-2 biased sigmoid
    scores, top-k inside the kept groups, unbiased scores renormalized unless
    ``renormalize`` (norm_topk_prob) is False."""
    scores = logits.float().sigmoid()
    choices = scores + bias
    group_scores = (
        choices.view(len(logits), num_groups, -1).topk(2, dim=-1).values.sum(-1)
    )
    kept = group_scores.topk(topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, kept, 1.0)
    expert_mask = group_mask.repeat_interleave(logits.shape[1] // num_groups, dim=1)
    ids = choices.masked_fill(expert_mask == 0, float("-inf")).topk(top_k).indices
    weights = scores.gather(1, ids)
    if not renormalize:
        return weights, ids
    return weights / weights.sum(-1, keepdim=True), ids


class TestSimulatedRoutingLogitEncoding(CustomTestCase):
    """SGLANG_SIMULATE_* routing on in-kernel routers."""

    NUM_EXPERTS = 8

    def setUp(self):
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(get_model().override(shared_experts=SharedExpertsConfig()))
        self.enterContext(
            get_parallel().override(
                attn_dp_size=1, attn_dp_rank=0, moe_ep_size=1, moe_ep_rank=0
            )
        )

    def _topk(self, backend, *, routing_method_type=None, simulate=True, **kwargs):
        from sglang.srt.environ import envs

        with envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(simulate):
            topk = topk_module.TopK(layer_id=3, **kwargs)
            topk.bind_runner_backend(backend, routing_method_type=routing_method_type)
        return topk

    def _forward(self, topk, num_tokens=3):
        """forward_cuda with the GPU encoder stubbed; returns the output and the
        logits each in-kernel router received."""
        encoded = torch.full((num_tokens, self.NUM_EXPERTS), 7.0)
        routed = []
        with (
            patch.object(
                topk_module, "is_expert_remapping_enabled", return_value=False
            ),
            patch.object(
                topk_module, "encode_simulated_router_logits", return_value=encoded
            ),
        ):
            output = topk.forward_cuda(
                torch.zeros((num_tokens, 4)),
                torch.zeros((num_tokens, self.NUM_EXPERTS)),
            )
        if topk.topk_config.output_format == TopKOutputFormat.BYPASSED:
            routed.append(output.raw_router_logits)
        return output, routed, encoded

    def _trtllm_configs(self):
        """Every TopK construction the TRT-LLM bypass predicate is asked about."""
        bias = torch.arange(self.NUM_EXPERTS, dtype=torch.float32)
        for method in RoutingMethodType:
            for (
                grouped,
                has_bias,
                renormalize,
                scoring_func,
                top_k,
                custom,
            ) in itertools.product(
                (False, True),
                (False, True),
                (False, True),
                ("softmax", "sigmoid"),
                (1, 2),
                (None, _custom_router),
            ):
                yield (
                    method,
                    dict(
                        top_k=top_k,
                        use_grouped_topk=grouped,
                        num_expert_group=4 if grouped else None,
                        topk_group=2 if grouped else None,
                        correction_bias=bias if has_bias else None,
                        renormalize=renormalize,
                        scoring_func=scoring_func,
                        custom_routing_function=custom,
                    ),
                )

    def test_every_in_kernel_router_receives_encoded_logits(self):
        """Every format whose kernel routes from logits must be handed the encoding,
        or the simulation silently does nothing there."""
        bypassed_methods = set()
        for method, kwargs in self._trtllm_configs():
            with self.subTest(method=method, **kwargs):
                topk = self._topk(
                    MoeRunnerBackend.FLASHINFER_TRTLLM,
                    routing_method_type=method,
                    **kwargs,
                )
                if topk.topk_config.output_format != TopKOutputFormat.BYPASSED:
                    continue
                bypassed_methods.add(method)
                output, routed, encoded = self._forward(topk)
                self.assertIs(routed[0], encoded)
                self.assertTrue(output.topk_config.routing_simulation_encoded)
        # A method added here must also route the encoding in
        # test_encoded_logits_route_back_to_the_simulated_experts.
        self.assertEqual(
            bypassed_methods,
            {
                RoutingMethodType.Default,
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Llama4,
                RoutingMethodType.SigmoidRenorm,
                RoutingMethodType.Sigmoid,
            },
        )

        # GPT-OSS-style MXFP4 experts bypass at bind.
        topk = self._topk(MoeRunnerBackend.FLASHINFER_MXFP4, top_k=2)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.BYPASSED)
        _, routed, encoded = self._forward(topk)
        self.assertIs(routed[0], encoded)

    def test_bypass_chosen_after_bind_is_encoded(self):
        """MiMo binds FP4 experts as standard, then switches to bypassed once its
        MXFP4 runner accepts the config."""
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            Mxfp4FlashinferTrtllmMoEMethod,
        )

        topk = self._topk(
            MoeRunnerBackend.FLASHINFER_MXFP4,
            is_fp4_experts=True,
            top_k=2,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            correction_bias=torch.zeros(self.NUM_EXPERTS),
            scoring_func="sigmoid",
        )
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)
        mxfp4 = Mxfp4FlashinferTrtllmMoEMethod(fp8_method=None, prefix="")
        self.assertTrue(mxfp4.configure_bypassed_topk(topk.topk_config))
        topk.topk_config.output_format = TopKOutputFormat.BYPASSED
        output, routed, encoded = self._forward(topk)
        self.assertIs(routed[0], encoded)
        self.assertTrue(output.topk_config.routing_simulation_encoded)

    def test_encoded_logits_route_back_to_the_simulated_experts(self):
        """Each router's semantics must select exactly the encoded experts with the
        materialized simulation's 1/k weights, in the dtypes runners pass."""
        native = topk_module.fused_topk_torch_native
        routers = {
            RoutingMethodType.Default: lambda x, k: native(x, x, k, False),
            RoutingMethodType.Renormalize: lambda x, k: native(x, x, k, True),
            RoutingMethodType.RenormalizeNaive: lambda x, k: native(x, x, k, True),
            RoutingMethodType.SigmoidRenorm: lambda x, k: native(
                x, x, k, True, scoring_func="sigmoid"
            ),
            RoutingMethodType.Sigmoid: lambda x, k: native(
                x, x, k, False, scoring_func="sigmoid"
            ),
            RoutingMethodType.Llama4: lambda x, k: native(
                x, x, k, False, scoring_func="sigmoid"
            ),
            RoutingMethodType.DeepSeekV3: lambda x, k: _deepseek_v3_route(
                x, torch.zeros(x.shape[1]), top_k=k, num_groups=4, topk_group=2
            ),
            "DeepSeekV3 without renormalize": lambda x, k: _deepseek_v3_route(
                x,
                torch.zeros(x.shape[1]),
                top_k=k,
                num_groups=4,
                topk_group=2,
                renormalize=False,
            ),
        }
        # Distinct ids within two of four groups of two experts, as the grouped
        # generator emits; every other method can reach these sets too.
        sets = {
            1: torch.tensor([[0], [5], [7]]),
            2: torch.tensor([[0, 1], [2, 7], [5, 4]]),
            3: torch.tensor([[0, 1, 2], [3, 6, 7], [4, 5, 0]]),
        }
        for (method, route), k, dtype in itertools.product(
            routers.items(), (1, 2, 3), (torch.float32, torch.bfloat16)
        ):
            if method == RoutingMethodType.Llama4 and k != 1:
                continue
            with self.subTest(method=method, k=k, dtype=dtype):
                encoding = topk_module.simulated_logit_encoding(
                    topk_module.TopKConfig(top_k=k)
                )
                ids = sets[k]
                logits = torch.full((len(ids), self.NUM_EXPERTS), encoding.other_logit)
                logits.scatter_(1, ids, encoding.target_logit)
                weights, routed = route(logits.to(dtype), k)
                self.assertTrue(
                    torch.equal(routed.sort(-1).values, ids.sort(-1).values)
                )
                # bf16 rounds Sigmoid's -ln(k - 1) target and the routers' math.
                tolerance = {} if dtype == torch.float32 else dict(rtol=0, atol=4e-3)
                torch.testing.assert_close(
                    weights.float(), torch.full((len(ids), k), 1.0 / k), **tolerance
                )

    def test_flags_unset_leave_the_bypassed_output_untouched(self):
        """Serving must reuse the shared config and the caller's logits."""
        topk = self._topk(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            routing_method_type=RoutingMethodType.Renormalize,
            simulate=False,
            top_k=2,
        )
        self.assertFalse(topk._routing_simulated)
        self.assertIsNone(topk._simulated_zero_bias)
        logits = torch.zeros((2, self.NUM_EXPERTS))
        with patch.object(
            topk_module, "is_expert_remapping_enabled", return_value=False
        ):
            output = topk.forward_cuda(torch.zeros((2, 4)), logits)
        self.assertIs(output.raw_router_logits, logits)
        self.assertIs(output.topk_config, topk.topk_config)

    def test_simulated_bias_is_zero_and_the_shared_config_keeps_its_own(self):
        bias = torch.nn.Parameter(torch.arange(self.NUM_EXPERTS, dtype=torch.float32))
        topk = self._topk(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            top_k=2,
            use_grouped_topk=True,
            num_expert_group=4,
            topk_group=2,
            correction_bias=bias,
            scoring_func="sigmoid",
        )
        simulated = topk._simulated_topk_config()
        self.assertIs(topk.topk_config.correction_bias, bias)
        self.assertFalse(topk.topk_config.routing_simulation_encoded)
        self.assertTrue(torch.equal(simulated.correction_bias, torch.zeros(8)))
        self.assertEqual(simulated.correction_bias.dtype, bias.dtype)
        self.assertEqual(
            (simulated.num_expert_group, simulated.topk_group, simulated.layer_id),
            (4, 2, 3),
        )
        self.assertEqual(
            topk_module.simulated_logit_encoding(topk.topk_config).groups, (4, 2)
        )
        # Reused across forwards, also when a model reassigns the same bias each
        # forward (beam), but rebuilt when a field changes after bind.
        self.assertIs(topk._simulated_topk_config(), simulated)
        topk.topk_config.correction_bias = bias
        self.assertIs(topk._simulated_topk_config(), simulated)
        topk.topk_config.allow_routed_experts_capture = False
        rebuilt = topk._simulated_topk_config()
        self.assertFalse(rebuilt.allow_routed_experts_capture)
        self.assertIs(topk._simulated_topk_config(), rebuilt)
        # A rebuild keeps the one cast of the zero bias.
        cast = rebuilt.correction_bias_for_dtype(torch.bfloat16)
        topk.topk_config.renormalize = not topk.topk_config.renormalize
        self.assertIs(
            topk._simulated_topk_config().correction_bias_for_dtype(torch.bfloat16),
            cast,
        )

    def test_both_simulation_flags_raise_at_construction(self):
        """In-kernel layers must not quietly pick uniform when both flags are set."""
        from sglang.srt.environ import envs

        # Catch inside the overrides: EnvField.override does not restore on raise.
        with (
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.override(True),
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.override(True),
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                topk_module.TopK(layer_id=3, top_k=2)


if __name__ == "__main__":
    unittest.main()

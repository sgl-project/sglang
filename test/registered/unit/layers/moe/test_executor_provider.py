"""Expert selection preserves stock routing and the caller's constructor ABI."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import executor_provider as providers
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.executor_provider import (
    MoeExecutorSelection,
    register_moe_provider,
    resolve_moe_executor,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _TestPlatform(SRTPlatform):
    def __init__(self, key="test_experts"):
        self.device_name = key


class _CPUExperts(torch.nn.Module):
    """External executor boundary: return input scaled by supplied route weights."""

    runner_backend = None
    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.routing_method_type = kwargs.get("routing_method_type")
        self.register_parameter(
            "w13_weight",
            torch.nn.Parameter(
                torch.zeros(
                    kwargs["num_experts"],
                    2 * kwargs["intermediate_size"],
                    kwargs["hidden_size"],
                )
            ),
        )

    def forward(self, hidden_states, topk_output):
        return hidden_states * topk_output.topk_weights.sum(-1, keepdim=True)


class _OtherCPUExperts(_CPUExperts):
    pass


def _expert_args(**updates):
    args = dict(
        num_experts=3,
        hidden_size=4,
        intermediate_size=8,
        layer_id=0,
        top_k=1,
        quant_config=None,
        activation="gelu",
    )
    args.update(updates)
    return args


class TestExecutorProvider(CustomTestCase):
    def setUp(self):
        super().setUp()
        for scope in (
            patch.object(providers, "_registry", providers._MoeProviderRegistry()),
            patch("sglang.srt.plugins._plugins_loaded", True),
            patch("sglang.srt.platforms._current_platform", _TestPlatform()),
            get_context().override_server_args(device="cpu"),
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.NONE,
                runner_backend=MoeRunnerBackend.AUTO,
                in_speculative_scope=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ),
        ):
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)

    def register(self, selector, name="cpu-experts", platform="test_experts"):
        register_moe_provider(name, platform_key=platform, selector=selector)

    def test_stock_backend_selection_and_provider_decline(self):
        cases = [
            (MoeA2ABackend.NONE, False, FusedMoE),
            (MoeA2ABackend.DEEPEP, False, DeepEPMoE),
            (MoeA2ABackend.DEEPEP_V2, False, DeepEPMoE),
            (MoeA2ABackend.MORI, False, DeepEPMoE),
            (MoeA2ABackend.MORI, True, FusedMoE),
            (MoeA2ABackend.MOONCAKE, False, DeepEPMoE),
            (MoeA2ABackend.NIXL, False, DeepEPMoE),
            (MoeA2ABackend.PPLX, False, DeepEPMoE),
        ]
        for registered in (False, True):
            with patch.object(providers, "_registry", providers._MoeProviderRegistry()):
                if registered:
                    self.register(lambda request: None)
                for backend, mori_v2, expected in cases:
                    with (
                        self.subTest(
                            registered=registered, backend=backend, mori_v2=mori_v2
                        ),
                        get_flags().moe.override(a2a_backend=backend),
                        patch.dict(
                            os.environ, {"SGLANG_MORI_EP_V2": str(int(mori_v2))}
                        ),
                    ):
                        selection = resolve_moe_executor(_expert_args())
                        self.assertIs(selection.factory, expected)
                        self.assertIsNone(selection.required_topk_format)

    def test_inactive_platform_never_invokes_selector(self):
        from sglang.srt.platforms.cuda import CudaSRTPlatform
        from sglang.srt.platforms.rocm import RocmSRTPlatform

        def unselected_selector(request):
            raise AssertionError(
                "An inactive platform must not influence GPU selection"
            )

        self.register(unselected_selector, platform="other-accelerator")
        for platform in (CudaSRTPlatform(), RocmSRTPlatform()):
            with (
                self.subTest(platform=platform.device_name),
                patch("sglang.srt.platforms._current_platform", platform),
            ):
                selection = resolve_moe_executor(_expert_args())
                self.assertIs(selection.factory, FusedMoE)
                self.assertIsNone(selection.required_topk_format)

    def test_cuda_and_rocm_are_distinct_provider_scopes(self):
        from sglang.srt.platforms.cuda import CudaSRTPlatform
        from sglang.srt.platforms.rocm import RocmSRTPlatform

        self.register(
            lambda request: MoeExecutorSelection(_CPUExperts), platform="cuda"
        )
        self.register(
            lambda request: MoeExecutorSelection(_OtherCPUExperts), platform="rocm"
        )
        for platform, expected in (
            (CudaSRTPlatform(), _CPUExperts),
            (RocmSRTPlatform(), _OtherCPUExperts),
        ):
            with (
                self.subTest(platform=platform.device_name),
                patch("sglang.srt.platforms._current_platform", platform),
            ):
                selection = resolve_moe_executor(_expert_args())
                self.assertIs(selection.factory, expected)

    def test_generic_matching_and_per_construction_settings(self):
        seen = []

        def selector(request):
            seen.append(request)
            factory = (
                _CPUExperts
                if request.expert_args["activation"] == "gelu"
                else _OtherCPUExperts
            )
            return MoeExecutorSelection(factory, TopKOutputFormat.STANDARD)

        self.register(selector)
        args = _expert_args(params_dtype=torch.bfloat16)
        first = resolve_moe_executor(args, model_type="first")
        with get_flags().moe.override(in_speculative_scope=True):
            second = resolve_moe_executor(
                {**args, "activation": "situ", "hidden_size": 2}, model_type="draft"
            )
        self.assertIs(first.factory, _CPUExperts)
        self.assertIs(second.factory, _OtherCPUExperts)
        self.assertEqual(seen[0].expert_args, args)
        self.assertEqual(seen[1].expert_args["hidden_size"], 2)
        self.assertEqual([r.is_draft for r in seen], [False, True])
        self.assertEqual([r.model_type for r in seen], ["first", "draft"])
        self.assertIs(seen[0].params_dtype, torch.bfloat16)

    def test_selector_cannot_mutate_constructor_arguments(self):
        args = _expert_args()

        def selector(request):
            request.expert_args["activation"] = "situ"

        self.register(selector)
        with self.assertRaisesRegex(RuntimeError, "cpu-experts.*failed") as cm:
            resolve_moe_executor(args)
        self.assertIsInstance(cm.exception.__cause__, TypeError)
        self.assertEqual(args["activation"], "gelu")

    def test_duplicate_registration_remains_fatal_after_plugin_loader_catches_it(self):
        def plugin():
            self.register(lambda request: MoeExecutorSelection(_CPUExperts))
            self.register(lambda request: None)

        ep = SimpleNamespace(
            name="test-experts",
            value="test:register",
            dist=None,
            load=lambda: plugin,
        )
        with (
            patch.dict(os.environ, {"SGLANG_PLUGINS": "", "SGLANG_PLATFORM": ""}),
            patch("sglang.srt.plugins._plugins_loaded", False),
            patch("sglang.srt.plugins.entry_points", return_value=[ep]),
            patch("sglang.srt.plugins.HookRegistry.apply_hooks"),
            self.assertLogs("sglang.srt.plugins", level="ERROR"),
        ):
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                resolve_moe_executor(_expert_args())

    def test_plugin_registration_finishes_before_first_selection(self):
        def plugin():
            self.register(
                lambda request: MoeExecutorSelection(
                    _CPUExperts, TopKOutputFormat.STANDARD
                )
            )

        ep = SimpleNamespace(
            name="test-experts",
            value="test:register",
            dist=None,
            load=lambda: plugin,
        )
        with (
            patch.dict(os.environ, {"SGLANG_PLUGINS": "", "SGLANG_PLATFORM": ""}),
            patch("sglang.srt.plugins._plugins_loaded", False),
            patch("sglang.srt.plugins.entry_points", return_value=[ep]),
            patch("sglang.srt.plugins.HookRegistry.apply_hooks"),
        ):
            selection = resolve_moe_executor(_expert_args())
        self.assertIs(selection.factory, _CPUExperts)
        self.assertIs(selection.required_topk_format, TopKOutputFormat.STANDARD)

    def test_ambiguous_matching_and_late_registration(self):
        def selector(request):
            return MoeExecutorSelection(_CPUExperts)

        def later_selector(request):
            raise AssertionError(
                "Do not invoke later selectors after detecting ambiguity"
            )

        self.register(selector, name="second")
        self.register(selector, name="first")
        self.register(later_selector, name="third")
        with self.assertRaisesRegex(RuntimeError, "first, second"):
            resolve_moe_executor(_expert_args())
        with self.assertRaisesRegex(ValueError, "before the first"):
            self.register(selector, name="late")
        with self.assertRaisesRegex(RuntimeError, "before the first"):
            resolve_moe_executor(_expert_args())

    def test_bad_selector_result_is_not_a_decline(self):
        for result in (
            _CPUExperts,
            MoeExecutorSelection(None),
            MoeExecutorSelection(_CPUExperts, "standard"),
        ):
            with (
                self.subTest(result=result),
                patch.object(providers, "_registry", providers._MoeProviderRegistry()),
            ):
                self.register(lambda request: result)
                with self.assertRaisesRegex(RuntimeError, "cpu-experts.*failed") as cm:
                    resolve_moe_executor(_expert_args())
                self.assertIsInstance(cm.exception.__cause__, TypeError)

    def test_routing_requirement_precedes_backend_auto_binding(self):
        for required, expected in (
            (None, TopKOutputFormat.TRITON_KERNEL),
            (TopKOutputFormat.STANDARD, TopKOutputFormat.STANDARD),
        ):
            with self.subTest(required=required):
                selection = MoeExecutorSelection(_CPUExperts, required)
                topk = TopK(
                    top_k=1,
                    layer_id=0,
                    output_format=selection.topk_output_format(None),
                )
                topk.bind_runner_backend(MoeRunnerBackend.TRITON_KERNELS)
                self.assertEqual(topk.topk_config.output_format, expected)
        selection = MoeExecutorSelection(_CPUExperts, TopKOutputFormat.STANDARD)
        self.assertEqual(
            selection.topk_output_format(TopKOutputFormat.STANDARD),
            TopKOutputFormat.STANDARD,
        )
        with self.assertRaisesRegex(ValueError, "conflicts with the model"):
            selection.topk_output_format(TopKOutputFormat.BYPASSED)

    def test_gemma_preserves_scaled_routing_weights(self):
        from sglang.srt.models.gemma4_causal import Gemma4MoE

        seen = []

        def selector(request):
            seen.append(request)
            return MoeExecutorSelection(_CPUExperts, TopKOutputFormat.STANDARD)

        self.register(selector)
        config = SimpleNamespace(
            model_type="gemma4_text",
            num_experts=2,
            top_k_experts=1,
            hidden_size=4,
            moe_intermediate_size=8,
        )
        layer = Gemma4MoE(
            4, 3, config, prefix="model.layers.3.moe", reduce_results=False
        )
        with torch.no_grad():
            layer.per_expert_scale.copy_(torch.tensor([2.0, 3.0]))
        x = torch.ones(2, 4)
        result = layer(x, torch.tensor([[9.0, 0.0], [0.0, 9.0]]))
        self.assertIsInstance(layer.experts, _CPUExperts)
        self.assertEqual(result.tolist(), [[2.0] * 4, [3.0] * 4])
        self.assertEqual(seen[0].model_class, "Gemma4MoE")
        self.assertEqual(layer.experts.args["prefix"], "model.layers.3.moe.experts")
        self.assertFalse(layer.experts.args["reduce_results"])

    def test_kimi_passes_latent_width_and_converted_expert_quantization(self):
        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
        from sglang.srt.models.kimi_k3 import KimiK3MoE

        seen = []

        def selector(request):
            seen.append(request)
            return MoeExecutorSelection(_CPUExperts, TopKOutputFormat.STANDARD)

        self.register(selector)
        config = KimiLinearConfig(
            hidden_size=4,
            num_attention_heads=2,
            num_experts=3,
            num_experts_per_token=2,
            moe_intermediate_size=8,
            routed_expert_hidden_size=2,
            latent_moe_use_norm=True,
            hidden_act="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
        quant = SimpleNamespace(quant_format="mxfp4")
        group = SimpleNamespace(world_size=1, rank_in_group=0)
        with get_parallel().override(tp_group=group):
            layer = KimiK3MoE(config, quant_config=quant, layer_idx=4)
        args = layer.experts.args
        self.assertIsInstance(layer.experts, _CPUExperts)
        self.assertEqual(args["hidden_size"], 2)
        self.assertEqual(args["layer_id"], 4)
        self.assertEqual((args["gemm1_alpha"], args["gemm1_clamp_limit"]), (4.0, 25.0))
        self.assertFalse(args["gate_up_interleaved"])
        self.assertIsInstance(args["quant_config"], Mxfp4Config)
        self.assertIs(seen[0].expert_args["quant_config"], args["quant_config"])
        self.assertIs(layer.topk.topk_config.output_format, TopKOutputFormat.STANDARD)

    def test_factory_failure_is_not_replaced_with_stock_executor(self):
        from sglang.srt.models.gemma4_causal import Gemma4MoE

        def broken_factory(**kwargs):
            raise RuntimeError("executor construction failed")

        self.register(lambda request: MoeExecutorSelection(broken_factory))
        config = SimpleNamespace(
            model_type="gemma4_text",
            num_experts=2,
            top_k_experts=1,
            hidden_size=4,
            moe_intermediate_size=8,
        )
        with self.assertRaisesRegex(RuntimeError, "executor construction failed"):
            Gemma4MoE(4, 0, config)


if __name__ == "__main__":
    unittest.main()

import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.moe.topk as topk_module
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_model, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_dispatch_preserves_negative_routing_sentinel():
    dispatcher = object.__new__(StandardDispatcher)
    dispatcher.moe_ep_size = 2
    dispatcher.skip_local_expert_mapping = False
    dispatcher.local_expert_mapping = torch.tensor([-1, -1, 0, 1], dtype=torch.int32)
    dispatcher.use_aiter_moe_runner = False
    dispatcher.expert_mask_gpu = None

    topk_ids = torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32)
    topk_output = StandardTopKOutput(
        topk_weights=torch.ones_like(topk_ids, dtype=torch.float32),
        topk_ids=topk_ids,
        router_logits=None,
    )

    with patch(
        "sglang.srt.layers.moe.token_dispatcher.standard."
        "should_use_flashinfer_cutlass_moe_fp4_allgather",
        return_value=False,
    ):
        output = dispatcher.dispatch(torch.ones((2, 4)), topk_output)

    expected = torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32)
    assert torch.equal(output.topk_output.topk_ids, expected)


class TestLayerRunnerDispatch(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_device",
                return_value="cpu",
            )
        )
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.TRITON,
                a2a_backend=MoeA2ABackend.NONE,
            )
        )
        self.enterContext(
            get_parallel().override(
                tp_size=2,
                tp_rank=1,
                tp_group=None,
                attn_dp_size=1,
                attn_dp_rank=0,
                attn_cp_size=1,
                attn_cp_rank=0,
                attn_tp_size=2,
                attn_tp_rank=1,
                moe_ep_size=2,
                moe_ep_rank=1,
                moe_tp_size=1,
                moe_tp_rank=0,
            )
        )

    @staticmethod
    def _config():
        return MoeRunnerConfig(
            num_experts=4,
            num_local_experts=2,
            num_fused_shared_experts=0,
            hidden_size=4,
            intermediate_size_per_partition=96,
            top_k=2,
        )

    def _build_layer(self, method, *, scheme=None, routing_method_type=None):
        def create_weights(*, layer, **kwargs):
            if scheme is not None:
                layer.scheme = scheme

        with patch.object(method, "create_weights", side_effect=create_weights):
            return FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=4,
                intermediate_size=96,
                layer_id=0,
                quant_method=method,
                routing_method_type=routing_method_type,
            )

    def _assert_local_routes(self, config):
        dispatcher = StandardDispatcher(config)
        routes = StandardTopKOutput(
            topk_weights=torch.ones((2, 2)),
            topk_ids=torch.tensor([[-1, 3], [2, 0]], dtype=torch.int32),
            router_logits=None,
        )
        with patch(
            "sglang.srt.layers.moe.token_dispatcher.standard."
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ):
            output = dispatcher.dispatch(torch.ones((2, 4)), routes)
        self.assertTrue(
            torch.equal(
                output.topk_output.topk_ids,
                torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32),
            )
        )
        topk = TopK(top_k=2, layer_id=0, allow_routed_experts_capture=False)
        topk.bind_runner_backend(config.runner_backend)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)
        self.assertFalse(dispatcher.use_aiter_moe_runner)

    def test_compressed_routes_apply_scale_once_for_the_effective_runner(self):
        """Wrapped FP8/NVFP4 experts consume final materialized route weights."""
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )

        with (
            get_flags().moe.override(runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ),
            get_model().override(shared_experts=SharedExpertsConfig()),
            patch.object(
                topk_module,
                "grouped_topk_gpu",
                topk_module.grouped_topk_gpu._torchdynamo_orig_callable,
            ),
            patch(
                "sglang.srt.layers.quantization.compressed_tensors.schemes."
                "compressed_tensors_w4a4_nvfp4_moe.get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
        ):
            schemes = (
                CompressedTensorsW8A8Fp8MoE(
                    SimpleNamespace(
                        strategy="block", block_structure=[128, 128], dynamic=False
                    ),
                    SimpleNamespace(strategy="token", dynamic=True),
                ),
                CompressedTensorsW4A4Nvfp4MoE(),
            )
            hidden = torch.ones((1, 128))
            logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
            expert_outputs = torch.tensor(
                [4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
            )
            for scheme in schemes:
                for renormalize in (True, False):
                    for factor in (1.0, 2.5):
                        with self.subTest(
                            scheme=type(scheme).__name__,
                            renormalize=renormalize,
                            factor=factor,
                        ):
                            method = CompressedTensorsFusedMoEMethod(None)

                            def select_method(layer, prefix):
                                layer.scheme = scheme
                                return method

                            layer = FusedMoE(
                                num_experts=8,
                                top_k=2,
                                hidden_size=128,
                                intermediate_size=128,
                                layer_id=0,
                                quant_config=SimpleNamespace(
                                    get_quant_method=select_method,
                                    get_name=lambda: "compressed-tensors",
                                ),
                                routed_scaling_factor=factor,
                                routing_method_type=RoutingMethodType.DeepSeekV3,
                            )
                            topk = TopK(
                                top_k=2,
                                layer_id=0,
                                use_grouped_topk=True,
                                num_expert_group=2,
                                topk_group=1,
                                renormalize=renormalize,
                                routed_scaling_factor=factor,
                                apply_routed_scaling_factor_on_output=layer.should_fuse_routed_scaling_factor_in_topk,
                                allow_routed_experts_capture=False,
                            )
                            topk.bind_runner_backend(
                                layer.runner_backend,
                                routing_method_type=layer.routing_method_type,
                            )
                            routes = topk.forward_cuda(hidden, logits)
                            self.assertEqual(routes.format, TopKOutputFormat.STANDARD)
                            dispatched = layer.dispatcher.dispatch(hidden, routes)
                            actual = (
                                expert_outputs[dispatched.topk_output.topk_ids.long()]
                                * dispatched.topk_output.topk_weights
                            ).sum()
                            reference = logits.softmax(-1)[:, :2]
                            if renormalize:
                                reference = reference / reference.sum(-1, keepdim=True)
                            expected = factor * (reference * expert_outputs[:2]).sum()
                            torch.testing.assert_close(actual, expected)

    def test_materialized_router_scale_is_independent_of_renormalization(self):
        """Folding a routed factor does not require selected-weight normalization."""
        hidden = torch.ones((1, 4))
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        for name in (
            "grouped_topk_gpu",
            "kimi_k2_biased_topk_impl",
            "biased_topk_impl",
        ):
            router = getattr(topk_module, name)._torchdynamo_orig_callable
            for renormalize in (True, False):
                for fold in (True, False):
                    with self.subTest(router=name, renormalize=renormalize, fold=fold):
                        kwargs = dict(
                            hidden_states=hidden,
                            gating_output=logits,
                            topk=2,
                            renormalize=renormalize,
                            routed_scaling_factor=2.5,
                            apply_routed_scaling_factor_on_output=fold,
                        )
                        if name == "grouped_topk_gpu":
                            kwargs.update(num_expert_group=2, topk_group=1)
                            scores = logits.softmax(-1)
                        else:
                            kwargs["correction_bias"] = torch.zeros(8)
                            scores = logits.sigmoid()
                        weights, ids = router(**kwargs)
                        expected = scores.gather(1, ids.long())
                        if renormalize:
                            expected = expected / expected.sum(-1, keepdim=True)
                        if fold:
                            expected *= 2.5
                        torch.testing.assert_close(weights, expected)
                        if name != "kimi_k2_biased_topk_impl":
                            kwargs.update(topk=3, num_fused_shared_experts=1)
                            shared_weights, shared_ids = router(**kwargs)
                            torch.testing.assert_close(shared_weights[:, :2], expected)
                            self.assertEqual(int(shared_ids[0, -1]), logits.shape[-1])
                            torch.testing.assert_close(
                                shared_weights[:, -1],
                                torch.tensor([1.0 if fold else 1.0 / 2.5]),
                            )

    def test_fp8_fallback_uses_local_ids_and_materialized_routes(self):
        """FP8 experts under an FP4 runner request must not receive global IDs."""
        for requested in (
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with (
                self.subTest(requested=requested),
                get_flags().moe.override(runner_backend=requested),
            ):
                method = object.__new__(Fp8MoEMethod)
                method.is_fp4_expert = False
                method.use_mxfp8 = False
                layer = self._build_layer(method)
                self._assert_local_routes(layer.moe_runner_config)

    def test_unquantized_fallback_is_independent_of_requested_runner(self):
        """A draft BF16 layer may fall back while its target keeps global IDs."""
        for requested in (
            MoeRunnerBackend.FLASHINFER_MEGAMOE,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with (
                self.subTest(requested=requested),
                get_flags().moe.override(runner_backend=requested),
            ):
                layer = self._build_layer(UnquantizedFusedMoEMethod())
                self._assert_local_routes(layer.moe_runner_config)

    def test_unsupported_aiter_shape_keeps_triton_local_ids(self):
        """Enabling AITER must not suppress local remapping after a shape fallback."""
        with (
            get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO),
            patch("sglang.srt.layers.quantization.unquant._use_aiter", True),
        ):
            layer = self._build_layer(UnquantizedFusedMoEMethod())
            self._assert_local_routes(layer.moe_runner_config)

    def test_fp8_fallback_keeps_dp_allgather_in_bf16(self):
        """FP8 fallback experts still gather DP-local rows without FP4 quantization."""
        captured = {}

        class Group:
            world_size = 2

            def all_gatherv(self, payloads, sizes):
                captured["payloads"] = payloads
                captured["sizes"] = sizes
                weights, ids, hidden = payloads
                return [
                    torch.cat([weights, weights]),
                    torch.tensor([[0, 3], [2, 1]], dtype=ids.dtype),
                    torch.cat([hidden, hidden + 1]),
                ]

        config = self._config()
        config.runner_backend = MoeRunnerBackend.TRITON
        dispatcher = StandardDispatcher(config)
        hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
        topk = StandardTopKOutput(
            torch.ones((1, 2)), torch.tensor([[0, 3]], dtype=torch.int32), None
        )
        with (
            get_parallel().override(tp_group=Group()),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard."
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.standard.get_dp_global_num_tokens",
                return_value=[1, 1],
            ),
        ):
            output = dispatcher.dispatch(hidden, topk)
        self.assertIs(captured["payloads"][2], hidden)
        self.assertEqual(captured["sizes"], [1, 1])
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        self.assertIsNone(output.hidden_states_scale)
        self.assertTrue(
            torch.equal(
                output.topk_output.topk_ids,
                torch.tensor([[-1, 1], [0, -1]], dtype=torch.int32),
            )
        )

    def test_direct_kernels_declare_their_execution_contract(self):
        """A missing or dormant runner must not change a direct kernel's ID ABI."""
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe import (
            CompressedTensorsMxInt4MoE,
        )
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8MoEMethod
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            Mxfp4FlashinferTrtllmMoEMethod,
        )

        cases = [
            (
                CompressedTensorsFusedMoEMethod,
                CompressedTensorsMxInt4MoE,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                False,
            ),
            (
                Mxfp4FlashinferTrtllmMoEMethod,
                None,
                MoeRunnerBackend.FLASHINFER_MXFP4,
                True,
            ),
            (ModelOptFp8MoEMethod, None, MoeRunnerBackend.FLASHINFER_TRTLLM, False),
        ]
        for method_type, scheme_type, backend, fp4 in cases:
            with (
                self.subTest(method=method_type.__name__),
                get_flags().moe.override(runner_backend=backend),
            ):
                method = object.__new__(method_type)
                layer = self._build_layer(
                    method,
                    scheme=object.__new__(scheme_type) if scheme_type else None,
                    routing_method_type=RoutingMethodType.DeepSeekV3,
                )
                self.assertEqual(layer.runner_backend, backend)
                self.assertTrue(layer.dispatcher.skip_local_expert_mapping)
                topk = TopK(
                    top_k=2,
                    layer_id=0,
                    is_fp4_experts=fp4,
                    use_grouped_topk=True,
                    num_expert_group=1,
                    topk_group=1,
                    scoring_func="sigmoid",
                    correction_bias=torch.zeros(4),
                )
                topk.bind_runner_backend(
                    layer.runner_backend,
                    routing_method_type=layer.routing_method_type,
                )
                self.assertEqual(
                    topk.topk_config.output_format,
                    TopKOutputFormat.STANDARD if fp4 else TopKOutputFormat.BYPASSED,
                )
                if method_type is ModelOptFp8MoEMethod:
                    self.assertTrue(method.runner.runner_backend.is_triton())
                    self.assertTrue(layer.use_flashinfer_trtllm_moe)
                    self.assertFalse(layer.moe_runner_config.inplace)

    def test_compressed_wrapper_propagates_the_inner_direct_contract(self):
        """Wrapper methods expose their scheme's format and global-ID contract."""
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe import (
            CompressedTensorsMxInt4MoE,
        )

        layer = self._build_layer(
            object.__new__(CompressedTensorsFusedMoEMethod),
            scheme=object.__new__(CompressedTensorsMxInt4MoE),
        )
        self.assertEqual(layer.runner_backend, MoeRunnerBackend.FLASHINFER_TRTLLM)
        self.assertTrue(layer.use_flashinfer_trtllm_moe)
        self.assertTrue(layer.dispatcher.skip_local_expert_mapping)

    def test_gfx1250_internal_router_maps_global_routes_to_the_local_shard(self):
        """gfx1250 W4A8 routes from raw logits inside the kernel; with expert
        parallelism the kernel maps those global routes onto this rank's shard
        through its expert map, so the layer builds as a direct kernel."""
        from sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe import (
            build_ep_expert_map,
        )
        from sglang.srt.layers.quantization.quark.quark import QuarkFusedMoEMethod
        from sglang.srt.layers.quantization.quark.schemes.quark_w4a8_mxfp4_moe import (
            QuarkW4A8MXFp4MoE,
        )

        with patch(
            "sglang.srt.layers.quantization.quark.schemes.quark_w4a8_mxfp4_moe."
            "_use_gfx1250_w4a8",
            True,
        ):
            expert_parallel = self._build_layer(
                object.__new__(QuarkFusedMoEMethod),
                scheme=object.__new__(QuarkW4A8MXFp4MoE),
            )
            with get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ):
                single_rank = self._build_layer(
                    object.__new__(QuarkFusedMoEMethod),
                    scheme=object.__new__(QuarkW4A8MXFp4MoE),
                )
        for layer in (expert_parallel, single_rank):
            self.assertIsNone(layer.runner_backend)
            self.assertFalse(layer.dispatcher.skip_local_expert_mapping)
        # Rank 1 of 2 owns global experts 2 and 3; the extra trailing slot keeps
        # the -1 padding route at -1.
        self.assertEqual(
            build_ep_expert_map(
                num_experts=4, num_local_experts=2, ep_rank=1, device="cpu"
            ).tolist(),
            [-1, -1, 0, 1, -1],
        )

    def test_borrowed_inkling_dense_loader_keeps_canonical_weights(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import unquant
        from sglang.srt.models.inkling_common.dense_mlp import InklingBatchDenseMLP

        layer = InklingBatchDenseMLP(
            n_shared_experts=2,
            d_model=4,
            shared_d_mlp=8,
            layer_id=0,
            prefix="model.layers.0.mlp.shared_experts",
            linearized_bf16=True,
        )
        self.assertIsNone(layer.quant_method.get_moe_runner_backend(layer))
        for offset in (0, 100):
            w13 = torch.arange(128).reshape(2, 16, 4).float() + offset
            w2 = torch.arange(64).reshape(2, 4, 8).float() + offset
            layer.weight_loader_fused(
                layer.w13_weight, w13, "model.layers.0.mlp.experts.w13_weight", "w13"
            )
            layer.weight_loader_fused(
                layer.w2_weight, w2, "model.layers.0.mlp.experts.w2_weight", "w2"
            )
            with (
                envs.SGLANG_OPT_FUSE_SWIGLU_INTERLEAVED.override(True),
                patch.object(unquant, "_is_cuda", True),
                patch.object(unquant, "_is_cpu", False),
            ):
                layer.quant_method.process_weights_after_loading(layer)
            torch.testing.assert_close(layer.w13_weight, w13, rtol=0, atol=0)
            torch.testing.assert_close(layer.w2_weight, w2, rtol=0, atol=0)
            w13_linear, w2_linear = layer.get_bf16_linearized_weights()
            torch.testing.assert_close(w13_linear, w13.reshape(32, 4), rtol=0, atol=0)
            torch.testing.assert_close(
                w2_linear, w2.transpose(1, 2).reshape(16, 4), rtol=0, atol=0
            )

    def test_topk_does_not_re_read_the_target_backend(self):
        """A constructed router keeps its layer's format across draft scopes."""
        topk = TopK(top_k=2, layer_id=0, allow_routed_experts_capture=False)
        topk.bind_runner_backend(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            routing_method_type=RoutingMethodType.Renormalize,
        )
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
            output = topk.forward_cuda(torch.ones((1, 4)), torch.ones((1, 4)))
        self.assertEqual(output.format, TopKOutputFormat.BYPASSED)

    def test_auto_nvfp4_preserves_undeclared_routing_policy(self):
        """An AUTO-selected FP4 kernel must preserve renormalization and custom routes."""
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        hidden = torch.ones((1, 4))
        custom_ids = torch.tensor([[3, 1]], dtype=torch.int32)
        custom_weights = torch.tensor([[0.25, 0.75]])

        def custom_router(**kwargs):
            return custom_weights.clone(), custom_ids.clone()

        def cpu_router(hidden_states, gating_output, topk, renormalize, **kwargs):
            return topk_module.fused_topk_torch_native(
                hidden_states,
                gating_output,
                topk,
                renormalize,
                correction_bias=kwargs["correction_bias"],
                scoring_func=kwargs["scoring_func"],
            )

        for custom, routing_method_type in (
            (None, None),
            (custom_router, None),
            (custom_router, RoutingMethodType.Renormalize),
        ):
            with (
                self.subTest(custom=custom is not None, native=routing_method_type),
                get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO),
                patch(
                    "sglang.srt.layers.quantization.modelopt_quant.is_cuda",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.quantization.modelopt_quant.get_device_capability",
                    return_value=(10, 0),
                ),
                patch.object(topk_module, "fused_topk", side_effect=cpu_router),
            ):
                method = object.__new__(ModelOptNvFp4FusedMoEMethod)
                layer = self._build_layer(
                    method, routing_method_type=routing_method_type
                )
                self.assertEqual(
                    layer.runner_backend, MoeRunnerBackend.FLASHINFER_TRTLLM
                )
                self.assertTrue(layer.dispatcher.skip_local_expert_mapping)
                topk = TopK(
                    top_k=2,
                    layer_id=0,
                    renormalize=True,
                    custom_routing_function=custom,
                    allow_routed_experts_capture=False,
                )
                topk.bind_runner_backend(
                    layer.runner_backend,
                    routing_method_type=layer.routing_method_type,
                )
                output = topk.forward_cuda(hidden, logits)
                self.assertEqual(output.format, TopKOutputFormat.STANDARD)
                expected_ids = (
                    logits.topk(2, dim=-1).indices.to(torch.int32)
                    if custom is None
                    else custom_ids
                )
                expected_weights = (
                    torch.softmax(logits[:, :2], dim=-1)
                    if custom is None
                    else custom_weights
                )
                self.assertTrue(torch.equal(output.topk_ids, expected_ids))
                torch.testing.assert_close(output.topk_weights, expected_weights)

    def test_per_tensor_fp8_undeclared_router_selects_local_layout(self):
        """A router the per-tensor kernel cannot represent must not use its packed weights."""
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8MoEMethod

        with get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM
        ):
            for routing_method_type, expected in (
                (None, MoeRunnerBackend.TRITON),
                (RoutingMethodType.Unspecified, MoeRunnerBackend.TRITON),
                (RoutingMethodType.Llama4, MoeRunnerBackend.FLASHINFER_TRTLLM),
            ):
                with self.subTest(routing_method_type=routing_method_type):
                    layer = self._build_layer(
                        object.__new__(ModelOptFp8MoEMethod),
                        routing_method_type=routing_method_type,
                    )
                    self.assertEqual(layer.runner_backend, expected)
                    self.assertEqual(
                        layer.dispatcher.skip_local_expert_mapping,
                        expected == MoeRunnerBackend.FLASHINFER_TRTLLM,
                    )
                    self.assertEqual(
                        layer.use_flashinfer_trtllm_moe,
                        expected == MoeRunnerBackend.FLASHINFER_TRTLLM,
                    )
                    topk = TopK(
                        top_k=1,
                        layer_id=0,
                        renormalize=False,
                        custom_routing_function=lambda **kwargs: None,
                    )
                    topk.bind_runner_backend(
                        layer.runner_backend,
                        routing_method_type=layer.routing_method_type,
                    )
                    self.assertEqual(
                        topk.topk_config.output_format,
                        TopKOutputFormat.BYPASSED
                        if routing_method_type == RoutingMethodType.Llama4
                        else TopKOutputFormat.STANDARD,
                    )
                    if expected == MoeRunnerBackend.FLASHINFER_TRTLLM:
                        with self.assertRaisesRegex(
                            NotImplementedError, "materialized"
                        ):
                            layer.quant_method.apply(
                                layer,
                                StandardDispatchOutput(
                                    torch.ones((1, 4)),
                                    None,
                                    StandardTopKOutput(
                                        torch.ones((1, 2)),
                                        torch.tensor([[0, 1]], dtype=torch.int32),
                                        None,
                                    ),
                                ),
                            )

    def test_declared_softmax_kernel_cannot_replace_other_router_policies(self):
        """A native softmax enum must not override sigmoid, grouping, bias, or renormalization."""
        for kwargs in (
            dict(
                scoring_func="sigmoid",
                use_grouped_topk=True,
                num_expert_group=2,
                topk_group=1,
                correction_bias=torch.zeros(4),
            ),
            dict(scoring_func="sigmoid", correction_bias=torch.zeros(4)),
            dict(renormalize=False),
        ):
            with self.subTest(kwargs=kwargs):
                topk = TopK(top_k=2, layer_id=0, **kwargs)
                topk.bind_runner_backend(
                    MoeRunnerBackend.FLASHINFER_TRTLLM,
                    routing_method_type=RoutingMethodType.RenormalizeNaive,
                )
                self.assertEqual(
                    topk.topk_config.output_format, TopKOutputFormat.STANDARD
                )

    def test_materialized_softmax_and_custom_routes_fold_scale_once(self):
        """Materializers scale routed weights once while keeping shared weights unchanged."""
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        hidden = torch.ones((1, 4))

        def cpu_router(hidden_states, gating_output, topk, renormalize, **kwargs):
            return topk_module.fused_topk_torch_native(
                hidden_states, gating_output, topk, renormalize
            )

        def custom_router(**kwargs):
            ids = [3, 1] + ([4] if kwargs["topk"] == 3 else [])
            weights = [0.25, 0.75] + ([1.0] if kwargs["topk"] == 3 else [])
            return torch.tensor([weights]), torch.tensor([ids], dtype=torch.int32)

        for custom, shared in ((None, 0), (custom_router, 0), (custom_router, 1)):
            with (
                self.subTest(custom=custom is not None, shared=shared),
                get_model().override(
                    shared_experts=SharedExpertsConfig(num_fused_shared_experts=shared)
                ),
                patch.object(topk_module, "fused_topk", side_effect=cpu_router),
            ):
                topk = TopK(
                    top_k=2,
                    layer_id=0,
                    custom_routing_function=custom,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                    allow_routed_experts_capture=False,
                )
                topk.bind_runner_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)
                output = topk.forward_cuda(hidden, logits)
                expected = (
                    torch.softmax(logits[:, :2], dim=-1)
                    if custom is None
                    else torch.tensor([[0.25, 0.75]])
                ) * 2.5
                torch.testing.assert_close(output.topk_weights[:, :2], expected)
                if shared:
                    self.assertEqual(output.topk_weights[0, 2].item(), 1.0)

    def test_model_constructors_preserve_materialized_routed_scale(self):
        """Bailing and normal LLaDA2 routing must retain scale when AUTO selects an FP4 kernel."""
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.models import bailing_moe, bailing_moe_linear, llada2

        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

        def cpu_router(hidden_states, gating_output, topk, renormalize, **kwargs):
            return topk_module.fused_topk_torch_native(
                hidden_states, gating_output, topk, renormalize
            )

        cases = (
            (bailing_moe, bailing_moe.BailingMoESparseMoeBlock, None),
            (bailing_moe_linear, bailing_moe_linear.BailingMoE, 0),
            (llada2, llada2.LLaDA2MoeSparseMoeBlock, None),
        )
        for module, block_type, shared in cases:
            config = SimpleNamespace(
                num_experts_per_tok=2,
                norm_topk_prob=True,
                hidden_size=4,
                moe_intermediate_size=96,
                num_experts=4,
                num_shared_experts=shared,
                hidden_act="silu",
                routed_scaling_factor=2.5,
                score_function="softmax",
                router_dtype="fp32",
            )
            if module is bailing_moe:
                # Grouped softmax still requires materialized routes.
                config.n_group = 2
                config.topk_group = 1
            method = object.__new__(ModelOptNvFp4FusedMoEMethod)
            with (
                self.subTest(model=block_type.__name__),
                get_flags().moe.override(runner_backend=MoeRunnerBackend.AUTO),
                patch.object(method, "create_weights"),
                patch.object(
                    module,
                    "get_moe_impl_class",
                    return_value=partial(FusedMoE, quant_method=method),
                ),
                patch(
                    "sglang.srt.layers.quantization.modelopt_quant.is_cuda",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.quantization.modelopt_quant.get_device_capability",
                    return_value=(10, 0),
                ),
                patch.object(llada2, "_is_npu", False),
                patch.object(topk_module, "fused_topk", side_effect=cpu_router),
                patch.object(
                    topk_module,
                    "grouped_topk_gpu",
                    topk_module.grouped_topk_gpu._torchdynamo_orig_callable,
                ),
            ):
                block = block_type(config=config, layer_id=0)
                self.assertEqual(
                    block.experts.runner_backend, MoeRunnerBackend.FLASHINFER_TRTLLM
                )
                output = block.topk.forward_cuda(torch.ones((1, 4)), logits)
                self.assertEqual(output.format, TopKOutputFormat.STANDARD)
                torch.testing.assert_close(
                    output.topk_weights,
                    torch.softmax(logits[:, :2], dim=-1) * 2.5,
                )

    def test_router_rebinding_rejects_conflicting_contract_before_mutation(self):
        for first, second in (
            (MoeRunnerBackend.FLASHINFER_TRTLLM, MoeRunnerBackend.TRITON),
            (MoeRunnerBackend.FLASHINFER_MXFP4, None),
            (None, MoeRunnerBackend.FLASHINFER_MXFP4),
        ):
            with self.subTest(first=first, second=second):
                topk = TopK(
                    top_k=2,
                    layer_id=0,
                    fused_gate_packed_ids=True,
                    allow_routed_experts_capture=False,
                )
                topk.bind_runner_backend(first)
                original = vars(topk.topk_config).copy()
                with self.assertRaisesRegex(RuntimeError, "different MoE runner"):
                    topk.bind_runner_backend(second)
                self.assertEqual(vars(topk.topk_config), original)
                topk.bind_runner_backend(first)
                self.assertEqual(vars(topk.topk_config), original)

    def test_router_rebinding_rejects_a_different_native_policy(self):
        topk = TopK(top_k=2, layer_id=0)
        topk.bind_runner_backend(
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            routing_method_type=RoutingMethodType.Renormalize,
        )
        with self.assertRaisesRegex(RuntimeError, "different MoE runner"):
            topk.bind_runner_backend(
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                routing_method_type=RoutingMethodType.Default,
            )
        self.assertEqual(
            topk.topk_config.routing_method_type, RoutingMethodType.Renormalize
        )

    def test_implicit_router_without_a_layer_contract_fails_closed(self):
        """A forgotten model association must fail before invoking a wrong kernel."""
        topk = TopK(top_k=2, layer_id=0)
        with self.assertRaisesRegex(RuntimeError, "owning experts"):
            topk.forward_cuda(torch.ones((1, 4)), torch.ones((1, 4)))

    def test_explicit_standard_router_preserves_its_format(self):
        """Routers consumed by model-side operations must stay materialized."""
        topk = TopK(top_k=2, layer_id=0, output_format=TopKOutputFormat.STANDARD)
        topk.bind_runner_backend(MoeRunnerBackend.FLASHINFER_TRTLLM)
        self.assertEqual(topk.topk_config.output_format, TopKOutputFormat.STANDARD)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

"""CPU contracts for MiniMax M3 on the ordinary FlashInfer TRT-LLM MoE path.

The production kernels only run on Blackwell GPUs, so these tests stop at the
Python/FlashInfer boundary.  They pin the MiniMax2 routing scale and OA-SwiGLU
activation arguments while explicitly guarding the routed backend from taking
on the ordinary backend's new arguments.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib.util
import inspect
import sys
import types
import unittest
from contextlib import nullcontext
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.moe.flashinfer_trtllm_moe as flashinfer_wrappers
import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as flashinfer_runner
import sglang.srt.layers.quantization.fp8 as fp8_module
import sglang.srt.layers.quantization.fp8_utils as fp8_utils_module
import sglang.srt.layers.quantization.unquant as unquant_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopKConfig,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.utils import custom_op as custom_op_module
from sglang.test.test_utils import CustomTestCase


class _Fp8QuantizationType(IntEnum):
    DeepSeekFp8 = 1
    MxFp8 = 2


class _ActivationType(IntEnum):
    Gelu = 0
    Silu = 2
    Swiglu = 3
    Geglu = 4
    Relu2 = 6


def _fake_flashinfer_modules(**fused_moe_functions):
    """Build the small FlashInfer module surface exercised by these tests."""
    flashinfer = types.ModuleType("flashinfer")
    flashinfer.__path__ = []
    fused_moe = types.ModuleType("flashinfer.fused_moe")
    fused_moe.__path__ = []
    fused_moe.Fp8QuantizationType = _Fp8QuantizationType
    for name, function in fused_moe_functions.items():
        setattr(fused_moe, name, function)

    core = types.ModuleType("flashinfer.fused_moe.core")
    core.ActivationType = _ActivationType
    flashinfer.fused_moe = fused_moe
    fused_moe.core = core
    return {
        "flashinfer": flashinfer,
        "flashinfer.fused_moe": fused_moe,
        "flashinfer.fused_moe.core": core,
    }


def _load_undecorated_wrapper_module():
    """Load the wrapper source with custom-op registration replaced by identity."""

    def identity_register_custom_op(fn=None, **_kwargs):
        if fn is not None:
            return fn
        return lambda decorated: decorated

    source = Path(flashinfer_wrappers.__file__)
    spec = importlib.util.spec_from_file_location(
        "_test_flashinfer_trtllm_moe_raw", source
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with patch.object(
        custom_op_module, "register_custom_op", identity_register_custom_op
    ):
        spec.loader.exec_module(module)
    return module


class TestFlashInferTrtllmMiniMaxActivationConfig(CustomTestCase):
    def test_activation_tensors_are_per_expert_fp32(self):
        config = MoeRunnerConfig(gemm1_alpha=1.702, gemm1_clamp_limit=7.0)

        (
            alpha,
            beta,
            clamp,
        ) = flashinfer_runner.create_flashinfer_trtllm_gemm1_activation_tensors(
            config, local_num_experts=3, device=torch.device("cpu")
        )

        self.assertEqual(alpha.dtype, torch.float32)
        self.assertEqual(beta.dtype, torch.float32)
        self.assertEqual(clamp.dtype, torch.float32)
        self.assertEqual(alpha.shape, (3,))
        self.assertEqual(beta.shape, (3,))
        self.assertEqual(clamp.shape, (3,))
        torch.testing.assert_close(alpha, torch.full((3,), 1.702))
        torch.testing.assert_close(beta, torch.ones(3))
        torch.testing.assert_close(clamp, torch.full((3,), 7.0))

    def test_beta_exists_only_when_alpha_exists(self):
        config = MoeRunnerConfig(gemm1_alpha=None, gemm1_clamp_limit=7.0)

        (
            alpha,
            beta,
            clamp,
        ) = flashinfer_runner.create_flashinfer_trtllm_gemm1_activation_tensors(
            config, local_num_experts=2, device=torch.device("cpu")
        )

        self.assertIsNone(alpha)
        self.assertIsNone(beta)
        torch.testing.assert_close(clamp, torch.full((2,), 7.0))

    def test_minimax2_prefers_topk_routed_scaling_factor(self):
        config = MoeRunnerConfig(routed_scaling_factor=1.25)

        factor = flashinfer_runner.resolve_flashinfer_trtllm_routed_scaling_factor(
            RoutingMethodType.MiniMax2,
            topk_routed_scaling_factor=2.0,
            runner_config=config,
            default=1.0,
        )

        self.assertEqual(factor, 2.0)

    def test_other_routes_preserve_runner_scaling_behavior(self):
        config = MoeRunnerConfig(routed_scaling_factor=1.25)

        factor = flashinfer_runner.resolve_flashinfer_trtllm_routed_scaling_factor(
            RoutingMethodType.Renormalize,
            topk_routed_scaling_factor=2.0,
            runner_config=config,
            default=1.0,
        )

        self.assertEqual(factor, 1.25)

    def test_scaling_falls_back_to_the_callers_default(self):
        config = MoeRunnerConfig(routed_scaling_factor=None)

        factor = flashinfer_runner.resolve_flashinfer_trtllm_routed_scaling_factor(
            RoutingMethodType.MiniMax2,
            topk_routed_scaling_factor=None,
            runner_config=config,
            default=1.0,
        )

        self.assertEqual(factor, 1.0)


class TestFlashInferTrtllmMiniMaxWrapperContract(CustomTestCase):
    def test_only_ordinary_fp8_wrapper_adds_oa_swiglu_arguments(self):
        ordinary_parameters = inspect.signature(
            flashinfer_wrappers._fake_fp8_block_scale_moe_out
        ).parameters
        routed_parameters = inspect.signature(
            flashinfer_wrappers._fake_fp8_block_scale_routed_moe_out
        ).parameters

        activation_arguments = {
            "gemm1_alpha",
            "gemm1_beta",
            "gemm1_clamp_limit",
        }
        self.assertTrue(activation_arguments <= ordinary_parameters.keys())
        self.assertTrue(activation_arguments.isdisjoint(routed_parameters.keys()))

    def test_ordinary_fp8_wrapper_forwards_oa_swiglu_arguments(self):
        raw_wrappers = _load_undecorated_wrapper_module()
        kernel = MagicMock(return_value=None)
        fake_modules = _fake_flashinfer_modules(trtllm_fp8_block_scale_moe=kernel)
        hidden_states = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
        output = torch.empty((2, 4), dtype=torch.bfloat16)
        scale = torch.ones((2, 1), dtype=torch.uint8)
        weight = torch.empty((3, 8, 4), dtype=torch.float8_e4m3fn)
        alpha = torch.full((3,), 1.702, dtype=torch.float32)
        beta = torch.ones(3, dtype=torch.float32)
        clamp = torch.full((3,), 7.0, dtype=torch.float32)

        with patch.dict(sys.modules, fake_modules):
            result = raw_wrappers.trtllm_fp8_block_scale_moe_out_wrapper(
                routing_logits=torch.empty((2, 3)),
                routing_bias=None,
                hidden_states=hidden_states,
                hidden_states_scale=scale,
                gemm1_weights=weight,
                gemm1_weights_scale=scale,
                gemm2_weights=weight,
                gemm2_weights_scale=scale,
                output=output,
                num_experts=3,
                top_k=2,
                n_group=None,
                topk_group=None,
                intermediate_size=4,
                local_expert_offset=0,
                local_num_experts=3,
                routed_scaling_factor=2.0,
                gemm1_alpha=alpha,
                gemm1_beta=beta,
                gemm1_clamp_limit=clamp,
            )

        self.assertIsNone(result)
        kwargs = kernel.call_args.kwargs
        self.assertIs(kwargs["output"], output)
        self.assertIs(kwargs["gemm1_alpha"], alpha)
        self.assertIs(kwargs["gemm1_beta"], beta)
        self.assertIs(kwargs["gemm1_clamp_limit"], clamp)


class TestFlashInferTrtllmMiniMaxQuantMethodConstruction(CustomTestCase):
    NUM_EXPERTS = 3

    @staticmethod
    def _layer():
        return SimpleNamespace(
            num_experts=3,
            num_local_experts=3,
            moe_ep_rank=0,
            w13_weight=torch.empty((3, 16, 4)),
            w2_weight=torch.empty((3, 4, 8)),
            w13_weight_scale_inv=torch.empty((3, 1, 1)),
            w2_weight_scale_inv=torch.empty((3, 1, 1)),
        )

    @staticmethod
    def _runner_config():
        return MoeRunnerConfig(
            num_local_experts=3,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )

    @staticmethod
    def _new_fp8_method(*, use_mxfp8):
        method = object.__new__(fp8_module.Fp8MoEMethod)
        method.use_mxfp8 = use_mxfp8
        method.block_quant = True
        method.quant_config = SimpleNamespace(
            use_mxfp8=use_mxfp8,
            weight_block_size=[1, 32] if use_mxfp8 else [128, 128],
        )
        return method

    @staticmethod
    def _new_unquantized_method(*, use_flashinfer_trtllm_moe=True):
        method = object.__new__(unquant_module.UnquantizedFusedMoEMethod)
        method.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        method.use_flashinfer_cutlass = False
        method.use_deep_gemm = False
        method.use_triton_kernels = False
        return method

    def test_fp8_activation_tensors_require_exact_ordinary_mxfp8_backend(self):
        activation_tensors = tuple(
            torch.full((self.NUM_EXPERTS,), value, dtype=torch.float32)
            for value in (1.702, 1.0, 7.0)
        )

        cases = (
            (MoeRunnerBackend.FLASHINFER_TRTLLM, True, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM, False, False),
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True, False),
            (MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM, True, False),
        )
        for backend, use_mxfp8, should_create in cases:
            with self.subTest(backend=backend, use_mxfp8=use_mxfp8):
                method = self._new_fp8_method(use_mxfp8=use_mxfp8)
                helper = MagicMock(return_value=activation_tensors)
                runner_instance = SimpleNamespace(
                    runner_backend=backend,
                    run=MagicMock(return_value="runner-output"),
                )
                runner = MagicMock(return_value=runner_instance)
                with (
                    patch.object(
                        fp8_module, "get_moe_runner_backend", return_value=backend
                    ),
                    patch.object(fp8_module, "_is_hip", False),
                    patch.object(
                        fp8_module, "use_intel_amx_backend", return_value=False
                    ),
                    patch.object(
                        fp8_module, "use_intel_xpu_backend", return_value=False
                    ),
                    patch.object(
                        fp8_module,
                        "create_flashinfer_trtllm_gemm1_activation_tensors",
                        helper,
                    ),
                    patch.object(fp8_module, "MoeRunner", runner),
                    patch.object(
                        flashinfer_runner, "get_activation_type", return_value=3
                    ),
                ):
                    layer = self._layer()
                    method.create_moe_runner(layer, self._runner_config())
                    output = method.apply(
                        layer,
                        SimpleNamespace(hidden_states=torch.empty((2, 4))),
                    )

                self.assertEqual(output, "runner-output")
                quant_info = runner_instance.run.call_args.args[1]
                if should_create:
                    helper.assert_called_once()
                    self.assertIs(
                        method.flashinfer_trtllm_gemm1_alpha,
                        activation_tensors[0],
                    )
                    self.assertIs(
                        method.flashinfer_trtllm_gemm1_beta,
                        activation_tensors[1],
                    )
                    self.assertIs(
                        method.flashinfer_trtllm_gemm1_clamp_limit,
                        activation_tensors[2],
                    )
                    self.assertIs(quant_info.gemm1_alpha, activation_tensors[0])
                    self.assertIs(quant_info.gemm1_beta, activation_tensors[1])
                    self.assertIs(quant_info.gemm1_clamp_limit, activation_tensors[2])
                else:
                    helper.assert_not_called()
                    self.assertIsNone(method.flashinfer_trtllm_gemm1_alpha)
                    self.assertIsNone(method.flashinfer_trtllm_gemm1_beta)
                    self.assertIsNone(method.flashinfer_trtllm_gemm1_clamp_limit)
                    self.assertIsNone(quant_info.gemm1_alpha)
                    self.assertIsNone(quant_info.gemm1_beta)
                    self.assertIsNone(quant_info.gemm1_clamp_limit)
                runner.assert_called_once_with(backend, method.moe_runner_config)

    def test_unquantized_activation_tensors_and_quant_info_are_ordinary_only(self):
        activation_tensors = tuple(
            torch.full((self.NUM_EXPERTS,), value, dtype=torch.float32)
            for value in (1.702, 1.0, 7.0)
        )

        for backend, use_flashinfer, runner_backend, should_create in (
            (
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                True,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                True,
            ),
            (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                True,
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
                False,
            ),
            (
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                False,
                MoeRunnerBackend.TRITON,
                False,
            ),
        ):
            with self.subTest(backend=backend, use_flashinfer=use_flashinfer):
                method = self._new_unquantized_method(
                    use_flashinfer_trtllm_moe=use_flashinfer
                )
                helper = MagicMock(return_value=activation_tensors)
                runner_instance = SimpleNamespace(
                    runner_backend=runner_backend,
                    run=MagicMock(return_value="runner-output"),
                )
                runner_constructor = MagicMock(return_value=runner_instance)
                layer = self._layer()
                config = self._runner_config()

                with (
                    patch.object(
                        unquant_module, "get_moe_runner_backend", return_value=backend
                    ),
                    patch.object(unquant_module, "MoeRunner", runner_constructor),
                    patch.object(unquant_module, "_use_aiter", False),
                    patch.object(
                        flashinfer_runner,
                        "create_flashinfer_trtllm_gemm1_activation_tensors",
                        helper,
                    ),
                ):
                    method.create_moe_runner(layer, config)
                    output = method.forward_cuda(
                        layer,
                        SimpleNamespace(hidden_states=torch.empty((2, 4))),
                    )

                self.assertEqual(output, "runner-output")
                runner_constructor.assert_called_once_with(runner_backend, config)
                quant_info = runner_instance.run.call_args.args[1]
                if should_create:
                    helper.assert_called_once()
                    self.assertIs(quant_info.gemm1_alpha, activation_tensors[0])
                    self.assertIs(quant_info.gemm1_beta, activation_tensors[1])
                    self.assertIs(quant_info.gemm1_clamp_limit, activation_tensors[2])
                else:
                    helper.assert_not_called()
                    if use_flashinfer:
                        self.assertIsNone(quant_info.gemm1_alpha)
                        self.assertIsNone(quant_info.gemm1_beta)
                        self.assertIsNone(quant_info.gemm1_clamp_limit)
                    else:
                        self.assertNotIsInstance(
                            quant_info,
                            flashinfer_runner.FlashInferTrtllmBf16MoeQuantInfo,
                        )

    def test_minimax_moe_constructs_experts_with_minimax2_routing(self):
        from sglang.srt.models import minimax_m3 as minimax_module

        config = SimpleNamespace(
            n_shared_experts=None,
            num_local_experts=4,
            num_experts_per_tok=2,
            hidden_size=8,
            intermediate_size=16,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            scoring_func="sigmoid",
            routed_scaling_factor=2.0,
            use_routing_bias=True,
        )
        experts_factory = MagicMock(return_value=torch.nn.Identity())
        topk_factory = MagicMock(return_value=torch.nn.Identity())
        moe_config = SimpleNamespace(
            disable_shared_experts_fusion=True,
            ep_num_redundant_experts=0,
        )

        with (
            patch.object(
                minimax_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
            patch.object(
                minimax_module,
                "get_exec",
                return_value=SimpleNamespace(moe=moe_config),
            ),
            patch.object(
                minimax_module, "get_moe_impl_class", return_value=experts_factory
            ),
            patch.object(minimax_module, "TopK", topk_factory),
        ):
            moe = minimax_module.MiniMaxM3MoE(
                config=config,
                quant_config=None,
                prefix="model.layers.0.mlp",
                layer_id=0,
            )

        kwargs = experts_factory.call_args.kwargs
        self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.MiniMax2)
        self.assertEqual(kwargs["gemm1_alpha"], 1.702)
        self.assertEqual(kwargs["gemm1_clamp_limit"], 7.0)
        self.assertFalse(kwargs["gate_up_interleaved"])

        topk_kwargs = topk_factory.call_args.kwargs
        self.assertEqual(topk_kwargs["routed_scaling_factor"], 2.0)
        self.assertTrue(topk_kwargs["apply_routed_scaling_factor_on_output"])
        self.assertTrue(topk_kwargs["renormalize"])
        self.assertEqual(topk_kwargs["scoring_func"], "sigmoid")
        self.assertIs(topk_kwargs["correction_bias"], moe.e_score_correction_bias)


class TestFlashInferTrtllmMiniMaxRunnerContract(CustomTestCase):
    HIDDEN_SIZE = 4
    NUM_EXPERTS = 3
    TOP_K = 2

    def setUp(self):
        super().setUp()
        self.hidden_states = torch.randn(2, self.HIDDEN_SIZE, dtype=torch.bfloat16)
        self.router_logits = torch.randn(2, self.NUM_EXPERTS, dtype=torch.float32)
        self.routing_bias = torch.randn(self.NUM_EXPERTS, dtype=torch.float32)
        self.activation_tensors = tuple(
            torch.full((self.NUM_EXPERTS,), value, dtype=torch.float32)
            for value in (1.702, 1.0, 7.0)
        )

    def _bypassed_dispatch(self):
        topk_config = TopKConfig(
            top_k=self.TOP_K,
            renormalize=True,
            routed_scaling_factor=2.0,
            correction_bias=self.routing_bias,
        )
        topk_output = BypassedTopKOutput(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=topk_config,
        )
        return StandardDispatchOutput(self.hidden_states, None, topk_output)

    def _standard_dispatch(self):
        topk_output = StandardTopKOutput(
            topk_weights=torch.full((2, self.TOP_K), 0.5, dtype=torch.float32),
            topk_ids=torch.tensor([[0, 1], [1, 2]], dtype=torch.int32),
            router_logits=self.router_logits,
        )
        return StandardDispatchOutput(self.hidden_states, None, topk_output)

    def _runner_config(self, *, routed_scaling_factor=None):
        return MoeRunnerConfig(
            num_local_experts=self.NUM_EXPERTS,
            intermediate_size_per_partition=8,
            top_k=self.TOP_K,
            num_fused_shared_experts=0,
            routing_method_type=RoutingMethodType.MiniMax2,
            activation="silu",
            is_gated=True,
            routed_scaling_factor=routed_scaling_factor,
        )

    def _fp8_quant_info(self):
        alpha, beta, clamp = self.activation_tensors
        return flashinfer_runner.FlashInferTrtllmFp8MoeQuantInfo(
            w13_weight=torch.empty(self.NUM_EXPERTS, 16, self.HIDDEN_SIZE),
            w2_weight=torch.empty(self.NUM_EXPERTS, self.HIDDEN_SIZE, 8),
            global_num_experts=self.NUM_EXPERTS,
            local_expert_offset=0,
            local_num_experts=self.NUM_EXPERTS,
            intermediate_size=8,
            routing_method_type=int(RoutingMethodType.MiniMax2),
            block_quant=True,
            use_mxfp8=True,
            weight_block_k=32,
            w13_weight_scale_inv=torch.empty(self.NUM_EXPERTS, 1, 1),
            w2_weight_scale_inv=torch.empty(self.NUM_EXPERTS, 1, 1),
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=clamp,
        )

    def _bf16_quant_info(self):
        alpha, beta, clamp = self.activation_tensors
        return flashinfer_runner.FlashInferTrtllmBf16MoeQuantInfo(
            gemm1_weights=torch.empty(self.NUM_EXPERTS, 16, self.HIDDEN_SIZE),
            gemm2_weights=torch.empty(self.NUM_EXPERTS, self.HIDDEN_SIZE, 8),
            global_num_experts=self.NUM_EXPERTS,
            local_expert_offset=0,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=clamp,
        )

    def _common_runner_patches(self):
        return (
            patch.object(
                flashinfer_runner,
                "use_symmetric_memory",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch.object(flashinfer_runner, "get_tp_group", return_value=None),
            patch.object(
                flashinfer_runner, "is_allocation_symmetric", return_value=False
            ),
        )

    def test_ordinary_mxfp8_passes_minimax_activation_and_topk_scale(self):
        kernel = MagicMock(return_value=None)
        fake_modules = _fake_flashinfer_modules()
        quantize = MagicMock(
            return_value=(
                self.hidden_states.clone(),
                torch.ones((self.hidden_states.shape[0], 1), dtype=torch.uint8),
            )
        )
        context_patch, group_patch, allocation_patch = self._common_runner_patches()

        with (
            patch.dict(sys.modules, fake_modules),
            context_patch,
            group_patch,
            allocation_patch,
            patch.object(fp8_utils_module, "flashinfer_mxfp8_quantize", quantize),
            patch.object(
                flashinfer_runner,
                "trtllm_fp8_block_scale_moe_out_wrapper",
                kernel,
            ),
        ):
            flashinfer_runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                self._bypassed_dispatch(),
                self._fp8_quant_info(),
                self._runner_config(),
            )

        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["routed_scaling_factor"], 2.0)
        self.assertIs(kwargs["routing_bias"], self.routing_bias)
        self.assertIs(kwargs["gemm1_alpha"], self.activation_tensors[0])
        self.assertIs(kwargs["gemm1_beta"], self.activation_tensors[1])
        self.assertIs(kwargs["gemm1_clamp_limit"], self.activation_tensors[2])

    def test_routed_mxfp8_call_remains_unchanged(self):
        kernel = MagicMock(return_value=None)
        fake_modules = _fake_flashinfer_modules()
        quantize = MagicMock(
            return_value=(
                self.hidden_states.clone(),
                torch.ones((self.hidden_states.shape[0], 1), dtype=torch.uint8),
            )
        )
        context_patch, group_patch, allocation_patch = self._common_runner_patches()

        with (
            patch.dict(sys.modules, fake_modules),
            context_patch,
            group_patch,
            allocation_patch,
            patch.object(fp8_utils_module, "flashinfer_mxfp8_quantize", quantize),
            patch.object(
                flashinfer_runner.PackTopkIds,
                "execute",
                return_value=torch.zeros((2, self.TOP_K), dtype=torch.int32),
            ),
            patch.object(
                flashinfer_runner,
                "trtllm_fp8_block_scale_routed_moe_out_wrapper",
                kernel,
            ),
        ):
            flashinfer_runner.fused_experts_none_to_flashinfer_trtllm_fp8(
                self._standard_dispatch(),
                self._fp8_quant_info(),
                self._runner_config(routed_scaling_factor=1.25),
                use_routed_topk=True,
            )

        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["routed_scaling_factor"], 1.25)
        self.assertNotIn("gemm1_alpha", kwargs)
        self.assertNotIn("gemm1_beta", kwargs)
        self.assertNotIn("gemm1_clamp_limit", kwargs)

    def test_ordinary_bf16_passes_minimax_activation_and_topk_scale(self):
        kernel = MagicMock(return_value=self.hidden_states.clone())
        fake_modules = _fake_flashinfer_modules(trtllm_bf16_moe=kernel)
        context_patch, group_patch, allocation_patch = self._common_runner_patches()

        with (
            patch.dict(sys.modules, fake_modules),
            context_patch,
            group_patch,
            allocation_patch,
        ):
            flashinfer_runner.fused_experts_none_to_flashinfer_trtllm_bf16(
                self._bypassed_dispatch(),
                self._bf16_quant_info(),
                self._runner_config(),
            )

        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["routed_scaling_factor"], 2.0)
        self.assertIs(kwargs["routing_bias"], self.routing_bias)
        self.assertIs(kwargs["gemm1_alpha"], self.activation_tensors[0])
        self.assertIs(kwargs["gemm1_beta"], self.activation_tensors[1])
        self.assertIs(kwargs["gemm1_clamp_limit"], self.activation_tensors[2])

    def test_routed_bf16_call_remains_unchanged(self):
        kernel = MagicMock(return_value=self.hidden_states.clone())
        fake_modules = _fake_flashinfer_modules(trtllm_bf16_routed_moe=kernel)
        context_patch, group_patch, allocation_patch = self._common_runner_patches()

        with (
            patch.dict(sys.modules, fake_modules),
            context_patch,
            group_patch,
            allocation_patch,
            patch.object(
                flashinfer_runner.PackTopkIds,
                "execute",
                return_value=torch.zeros((2, self.TOP_K), dtype=torch.int32),
            ),
        ):
            flashinfer_runner.fused_experts_none_to_flashinfer_trtllm_bf16(
                self._standard_dispatch(),
                self._bf16_quant_info(),
                self._runner_config(routed_scaling_factor=1.25),
                use_routed_topk=True,
            )

        kwargs = kernel.call_args.kwargs
        self.assertEqual(kwargs["routed_scaling_factor"], 1.25)
        self.assertNotIn("gemm1_alpha", kwargs)
        self.assertNotIn("gemm1_beta", kwargs)
        self.assertNotIn("gemm1_clamp_limit", kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=3)

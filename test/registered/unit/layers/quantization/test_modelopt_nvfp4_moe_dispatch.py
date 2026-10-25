"""CPU unit tests for NVFP4 fused-MoE backend dispatch in ModelOptNvFp4FusedMoEMethod.apply.

apply() picks the kernel path from the backend the method cached in
create_moe_runner, not from the process-wide MoE runner backend, which
speculative decoding changes after the weights were prepared. These tests pin
that for the FlashInfer TRT-LLM path, which serves the regular and the routed
TRT-LLM backend from one weight prep.

The platform check and the runner are stubbed and the layer is a bag of small
tensors, so the tests stay on CPU; the kernels are covered on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import sys
import unittest
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

# Import modelopt_quant before flashinfer_trtllm (see
# test_modelopt_nvfp4_moe_scales.py for the circular-import reason).
# isort: off
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp4MoeQuantInfo,
)

# isort: on
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 32


class _Runner:
    """Records the quant_info apply() hands to the runner."""

    def __init__(self):
        self.calls = []

    def run(self, dispatch_output, quant_info):
        self.calls.append((dispatch_output, quant_info))
        return "combine-input"


def _trtllm_prepared_layer() -> SimpleNamespace:
    """A layer as align_fp4_moe_weights_for_flashinfer_trtllm leaves it: packed FP4
    weights, FP8 block scales and the TRT-LLM output scalars, incl. g1_scale_c."""

    def p(t: torch.Tensor) -> torch.nn.Parameter:
        return torch.nn.Parameter(t, requires_grad=False)

    return SimpleNamespace(
        w13_weight=p(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8)
        ),
        w2_weight=p(
            torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8)
        ),
        w13_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16, dtype=torch.float8_e4m3fn
            )
        ),
        w2_weight_scale=p(
            torch.zeros(
                NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16, dtype=torch.float8_e4m3fn
            )
        ),
        g1_scale_c=p(torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32)),
        g1_alphas=p(torch.full((NUM_EXPERTS,), 0.5, dtype=torch.float32)),
        g2_alphas=p(torch.full((NUM_EXPERTS,), 0.75, dtype=torch.float32)),
        w13_input_scale_quant=torch.tensor(2.0, dtype=torch.float32),
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        moe_ep_rank=0,
        intermediate_size_per_partition=INTERMEDIATE,
        # The weight alignment sets this; HIDDEN needs no padding.
        trtllm_padded_hidden_size=HIDDEN,
        # FusedMoE.__init__ sets this; apply() reads it to reject the fused
        # fallback for MegaMoE experts.
        _mega_moe_nvfp4=False,
    )


@contextmanager
def _live_backend(backend: MoeRunnerBackend):
    """The process-wide MoE runner backend, as the method sees it."""
    with mock.patch.object(
        modelopt_quant, "get_moe_runner_backend", return_value=backend
    ):
        yield


def _method_set_up_for(backend: MoeRunnerBackend) -> ModelOptNvFp4FusedMoEMethod:
    """A method constructed and given its runner while ``backend`` was the live one."""
    with (
        _live_backend(backend),
        mock.patch.object(
            modelopt_quant,
            "get_platform",
            return_value=SimpleNamespace(is_blackwell=True),
        ),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            SimpleNamespace(use_per_token_activation=False)
        )
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=NUM_EXPERTS,
                num_local_experts=NUM_EXPERTS,
                hidden_size=HIDDEN,
                intermediate_size_per_partition=INTERMEDIATE,
                activation="silu",
                is_gated=True,
            ),
        )
    method.runner = _Runner()
    return method


class TestNvFp4MoeDispatch(CustomTestCase):
    def test_online_activation_scale_fallback_is_initialized_for_each_setting(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization.nvfp4_online import NvFp4OnlineConfig

        for setting in (False, True, "prefill", "decode"):
            with (
                self.subTest(setting=setting),
                envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.override(setting),
                _live_backend(MoeRunnerBackend.FLASHINFER_TRTLLM),
            ):
                method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM)
                method.quant_config = NvFp4OnlineConfig()
                layer = torch.nn.Module()
                layer.num_local_experts = layer.num_experts = NUM_EXPERTS
                layer.moe_runner_config = method.moe_runner_config
                method.create_weights(
                    layer,
                    NUM_EXPERTS,
                    HIDDEN,
                    INTERMEDIATE,
                    torch.bfloat16,
                    weight_loader=lambda *args, **kwargs: None,
                )
                torch.testing.assert_close(
                    layer.w13_input_scale,
                    torch.ones(NUM_EXPERTS, 2),
                )
                torch.testing.assert_close(
                    layer.w2_input_scale,
                    torch.ones(NUM_EXPERTS),
                )

    def test_megamoe_w4a16_uses_weight_scales_without_activation_scales(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import flashinfer_megamoe

        # Exercise apply's scale selection without constructing a distributed
        # MegaMoE runner; this test records the kernel arguments on CPU.
        method = ModelOptNvFp4FusedMoEMethod.__new__(ModelOptNvFp4FusedMoEMethod)
        method._moe_runner_backend = MoeRunnerBackend.FLASHINFER_MEGAMOE
        method.moe_runner_config = MoeRunnerConfig(activation="silu")
        method.runner = _Runner()
        layer = SimpleNamespace(
            _mega_moe_nvfp4=False,
            _flashinfer_megamoe_forward=object(),
            _w4a16_g1_alphas=torch.tensor([2.0, 4.0]),
            w2_weight_scale_2=torch.tensor([3.0, 6.0]),
            g1_alphas=torch.tensor([1.0, 2.0]),
            g2_alphas=torch.tensor([0.75, 1.5]),
            w2_input_scale_quant=torch.tensor([4.0, 4.0]),
            should_fuse_routed_scaling_factor_in_topk=False,
        )
        identities = {name: id(value) for name, value in vars(layer).items()}
        for setting in (False, True):
            with (
                self.subTest(setting=setting),
                envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(setting),
                mock.patch.object(
                    flashinfer_megamoe,
                    "ensure_nvfp4_moe_layer_for_flashinfer_megamoe",
                    return_value=object(),
                ),
            ):
                method.apply(layer, object())
                quant_info = method.runner.calls[-1][1]
                self.assertIs(
                    quant_info.fc1_alpha,
                    layer._w4a16_g1_alphas if setting else layer.g1_alphas,
                )
                self.assertIs(
                    quant_info.fc2_alpha,
                    layer.w2_weight_scale_2 if setting else layer.g2_alphas,
                )
                self.assertIs(
                    quant_info.fc1_norm_const,
                    None if setting else layer.w2_input_scale_quant,
                )
                self.assertEqual(
                    {name: id(value) for name, value in vars(layer).items()}, identities
                )

    def test_routed_trtllm_takes_the_trtllm_path(self):
        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            MoeRunnerBackend.FLASHINFER_TRTLLM,
        ):
            with self.subTest(backend=backend.value):
                method = _method_set_up_for(backend)
                layer = _trtllm_prepared_layer()
                dispatch_output = object()

                with _live_backend(backend):
                    out = method.apply(layer, dispatch_output)

                self.assertEqual(out, "combine-input")
                ((seen_dispatch, quant_info),) = method.runner.calls
                self.assertIs(seen_dispatch, dispatch_output)
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)
                self.assertEqual(
                    quant_info.g1_scale_c.data_ptr(), layer.g1_scale_c.data_ptr()
                )
                self.assertEqual(
                    quant_info.g1_alphas.data_ptr(), layer.g1_alphas.data_ptr()
                )
                self.assertEqual(quant_info.local_num_experts, NUM_EXPERTS)
                self.assertEqual(
                    quant_info.intermediate_size_per_partition, INTERMEDIATE
                )

    def test_dispatch_follows_the_backend_the_runner_was_created_for(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        for live in (
            MoeRunnerBackend.AUTO,
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            with self.subTest(live=live.value):
                method.runner = _Runner()

                with _live_backend(live):
                    method.apply(_trtllm_prepared_layer(), object())

                ((_, quant_info),) = method.runner.calls
                self.assertIsInstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo)

    def test_materialized_router_weights_reach_fp8_and_bf16_routed_kernels(self):
        """Materialized custom weights must not be sent to a logits router or normalized again."""
        from sglang.srt.layers.moe import topk as topk_module
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as kernels
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
        from sglang.srt.runtime_context import get_parallel

        logits = torch.tensor([[0.0, 1.0, -2.0, 2.0]])
        ids = torch.tensor([[3, 1]], dtype=torch.int32)
        routed_scale = 2.5
        weights = torch.sigmoid(logits).gather(1, ids.long())
        weights = weights / weights.sum(dim=-1, keepdim=True) * routed_scale
        hidden = torch.zeros((1, 4), dtype=torch.bfloat16)
        config = MoeRunnerConfig(
            top_k=2,
            activation="silu",
            num_local_experts=4,
            num_fused_shared_experts=0,
            intermediate_size_per_partition=128,
            routed_scaling_factor=routed_scale,
        )
        expected = (ids.float() * weights).sum(dim=-1, keepdim=True).expand_as(hidden)
        fake_flashinfer = ModuleType("flashinfer.fused_moe")
        fake_flashinfer.Fp8QuantizationType = SimpleNamespace(MxFp8=1, DeepSeekFp8=0)
        fake_core = ModuleType("flashinfer.fused_moe.core")
        fake_core.ActivationType = SimpleNamespace(
            Swiglu=SimpleNamespace(value=0),
            Geglu=SimpleNamespace(value=1),
            Situ=SimpleNamespace(value=2),
        )

        def routed_result(kwargs):
            self.assertEqual(kwargs["routing_method_type"], RoutingMethodType.TopK)
            routing = kwargs["topk_ids"]
            if isinstance(routing, tuple):
                seen_ids, seen_weights = routing
            else:
                seen_ids, seen_weights = routing, kwargs["topk_weights"]
            torch.testing.assert_close(seen_ids, ids)
            torch.testing.assert_close(seen_weights, weights)
            return (
                (seen_ids.float() * seen_weights)
                .sum(dim=-1, keepdim=True)
                .expand_as(hidden)
                .to(hidden.dtype)
            )

        def fp8_kernel(**kwargs):
            kwargs["output"].copy_(routed_result(kwargs))

        def bf16_kernel(**kwargs):
            return routed_result(kwargs)

        fake_flashinfer.trtllm_bf16_routed_moe = bf16_kernel
        common = dict(
            w13_weight=torch.zeros((4, 256, 4)),
            w2_weight=torch.zeros((4, 4, 128)),
            global_num_experts=4,
            local_num_experts=4,
            local_expert_offset=0,
            routing_method_type=None,
            intermediate_size=128,
        )
        quant_info = SimpleNamespace(
            **common,
            block_quant=True,
            use_mxfp8=False,
            weight_block_k=128,
            w13_weight_scale_inv=torch.ones((4, 2, 1)),
            w2_weight_scale_inv=torch.ones((4, 1, 1)),
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            activation_type=None,
        )
        bf16_info = SimpleNamespace(
            gemm1_weights=common["w13_weight"],
            gemm2_weights=common["w2_weight"],
            global_num_experts=4,
            local_expert_offset=0,
        )
        with (
            get_parallel().override(tp_group=None),
            mock.patch.dict(
                sys.modules,
                {
                    "flashinfer.fused_moe": fake_flashinfer,
                    "flashinfer.fused_moe.core": fake_core,
                },
            ),
            mock.patch.object(kernels, "is_allocation_symmetric", return_value=False),
            mock.patch.object(kernels, "trtllm_moe_enable_pdl", return_value=False),
            mock.patch.object(
                kernels,
                "per_token_group_quant_fp8",
                return_value=(hidden, torch.ones((1, 1))),
            ),
            mock.patch.object(
                kernels, "trtllm_fp8_block_scale_routed_moe_out_wrapper", fp8_kernel
            ),
        ):
            for method_type, kernel, info in (
                (
                    Fp8MoEMethod,
                    kernels.fused_experts_none_to_flashinfer_trtllm_fp8,
                    quant_info,
                ),
                (
                    UnquantizedFusedMoEMethod,
                    kernels.fused_experts_none_to_flashinfer_trtllm_bf16,
                    bf16_info,
                ),
            ):
                with self.subTest(method=method_type.__name__):
                    fold_scale = _fuses_routed_scaling_factor_in_topk(
                        object.__new__(method_type), MoeRunnerBackend.FLASHINFER_TRTLLM
                    )
                    produced_weights, produced_ids = (
                        topk_module.biased_grouped_topk_impl.__wrapped__(
                            hidden,
                            logits,
                            torch.zeros(4),
                            topk=2,
                            renormalize=True,
                            num_expert_group=1,
                            topk_group=1,
                            routed_scaling_factor=routed_scale,
                            apply_routed_scaling_factor_on_output=fold_scale,
                        )
                    )
                    dispatch = StandardDispatchOutput(
                        hidden,
                        None,
                        StandardTopKOutput(
                            produced_weights, produced_ids.to(torch.int32), None
                        ),
                    )
                    output = kernel(dispatch, info, config)
                    torch.testing.assert_close(
                        output.hidden_states, expected.to(hidden.dtype)
                    )

    def test_fp8_default_routing_policy_keeps_zero_enum(self):
        """The native softmax selector's zero value must not select grouped sigmoid routing."""
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as kernels
        from sglang.srt.layers.quantization import fp8

        method = object.__new__(fp8.Fp8MoEMethod)
        method.is_fp4_expert = False
        method.block_quant = True
        method.quant_config = SimpleNamespace(
            use_mxfp8=False, weight_block_size=[128, 128]
        )
        method.moe_runner_config = MoeRunnerConfig(activation="silu")
        method.runner = SimpleNamespace(
            runner_backend=MoeRunnerBackend.FLASHINFER_TRTLLM,
            run=lambda dispatch, quant_info: quant_info.routing_method_type,
        )
        layer = SimpleNamespace(
            num_experts=4,
            num_local_experts=4,
            moe_ep_rank=0,
            w13_weight=torch.zeros((4, 256, 4)),
            w2_weight=torch.zeros((4, 4, 128)),
            w13_weight_scale_inv=torch.ones((4, 2, 1)),
            w2_weight_scale_inv=torch.ones((4, 1, 1)),
            _flashinfer_trtllm_gemm1_alpha=None,
            _flashinfer_trtllm_gemm1_beta=None,
            _flashinfer_trtllm_gemm1_clamp_limit=None,
        )
        with (
            mock.patch.object(fp8, "use_intel_amx_backend", return_value=False),
            mock.patch.object(fp8, "is_xpu", return_value=False),
            mock.patch.object(
                fp8,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
            ),
            mock.patch.object(kernels, "get_activation_type", return_value=0),
        ):
            for policy, expected in (
                (RoutingMethodType.Default, 0),
                (RoutingMethodType.Renormalize, 1),
                (None, 2),
            ):
                with self.subTest(policy=policy):
                    layer.routing_method_type = method.get_moe_routing_method_type(
                        layer, policy
                    )
                    self.assertEqual(
                        method.apply(
                            layer, SimpleNamespace(hidden_states=torch.zeros((1, 4)))
                        ),
                        expected,
                    )

    def test_missing_trtllm_prep_names_the_missing_field(self):
        method = _method_set_up_for(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED)
        layer = _trtllm_prepared_layer()
        del layer.g1_scale_c

        with _live_backend(MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED):
            with self.assertRaises(AttributeError) as ctx:
                method.apply(layer, object())

        self.assertIn("g1_scale_c", str(ctx.exception))
        self.assertEqual(method.runner.calls, [])


class TestNvFp4CuteDslPhaseState(CustomTestCase):
    def test_per_token_fc2_normalizer_preserves_default_and_uses_4over6_recipe(self):
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl as cutedsl

        quantization = ModuleType("flashinfer.quantization.nvfp4_quantization_utils")
        unit = torch.ones(1)
        online = torch.tensor([1.0 / (6.0 * 448.0)])
        for recipe in (None, object()):
            with self.subTest(use_4over6=recipe is not None):
                quantization.current_nvfp4_4over6_config = lambda: recipe
                make_scale = quantization.make_nvfp4_global_scale = mock.Mock(
                    return_value=online
                )
                with mock.patch.dict(
                    sys.modules,
                    {"flashinfer.quantization.nvfp4_quantization_utils": quantization},
                ):
                    fc2 = cutedsl._make_per_token_global_scale(
                        unit, keep_unit_without_4over6=True
                    )
                if recipe is None:
                    self.assertIs(fc2, unit)
                    make_scale.assert_not_called()
                else:
                    self.assertIs(fc2, online)
                    make_scale.assert_called_once_with(
                        unit,
                        per_token_activation=True,
                        nvfp4_4over6_config=recipe,
                    )

    def test_phase_switch_and_reload_keep_distinct_scales_and_workspaces(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutedsl as cutedsl
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        wrappers = []

        class Wrapper:
            def __init__(self, **kwargs):
                self.quant_mode = kwargs["quant_mode"]
                wrappers.append(self)

        flashinfer = ModuleType("flashinfer")
        flashinfer.CuteDslMoEWrapper = Wrapper
        flashinfer.ActivationType = SimpleNamespace(Swiglu="swiglu", Relu2="relu2")
        layer = SimpleNamespace(
            num_experts=2,
            num_local_experts=2,
            moe_ep_rank=0,
            top_k=1,
            hidden_size=128,
            intermediate_size_per_partition=128,
            moe_runner_config=SimpleNamespace(
                activation="silu", params_dtype=torch.bfloat16
            ),
            quant_config=SimpleNamespace(use_per_token_activation=False),
            w13_weight=torch.zeros(2, 256, 64, dtype=torch.uint8),
            g1_alphas=torch.tensor([1.0, 2.0]),
            g2_alphas=torch.tensor([3.0, 4.0]),
            w13_weight_scale_2=torch.tensor([2.0, 4.0]),
            w2_weight_scale_2=torch.tensor([12.0, 16.0]),
            w13_input_scale_quant=torch.tensor([2.0, 2.0]),
            w2_input_scale_quant=torch.tensor([4.0, 4.0]),
        )
        with (
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override("decode"),
            mock.patch.dict(sys.modules, {"flashinfer": flashinfer}),
            mock.patch.object(cutedsl, "cuda_graph_fully_disabled", return_value=True),
            mock.patch.object(
                cutedsl, "get_parallel", return_value=SimpleNamespace(num_dp_ranks=1)
            ),
            mock.patch.object(cutedsl, "cutedsl_moe_max_num_tokens", return_value=16),
        ):
            identities = None
            for phase in ("prefill", "decode", "prefill"):
                with forward_context(
                    ForwardContext(attn_backend=None, moe_forward_phase=phase)
                ):
                    state = cutedsl.ensure_cutedsl_wrapper(layer)
                    self.assertEqual(
                        state.wrapper.quant_mode,
                        "w4a16" if phase == "decode" else "w4a4",
                    )
                    current = {name: id(value) for name, value in vars(layer).items()}
                    if identities is None:
                        identities = current
                    self.assertEqual(current, identities)
            self.assertEqual(len(wrappers), 2)
            self.assertIsNot(wrappers[0], wrappers[1])
            states = layer._cutedsl_states
            torch.testing.assert_close(
                states["w4a4", False].scales[0], torch.tensor([1.0, 2.0])
            )
            torch.testing.assert_close(
                states["w4a16", False].scales[0], torch.tensor([2.0, 4.0])
            )
            torch.testing.assert_close(
                states["w4a4", False].scales[2], torch.tensor([3.0, 4.0])
            )
            torch.testing.assert_close(
                states["w4a16", False].scales[2], torch.tensor([12.0, 16.0])
            )
            pointers = {
                mode: [
                    tensor.data_ptr() for tensor in (*state.scales, state.input_scale)
                ]
                for mode, state in states.items()
            }
            layer.w13_weight_scale_2.mul_(2)
            layer.w2_weight_scale_2.mul_(2)
            cutedsl.refresh_cutedsl_standard_scales_for_weight_update(layer)
            self.assertEqual(
                {name: id(value) for name, value in vars(layer).items()}, identities
            )
            for mode, state in states.items():
                self.assertEqual(
                    [
                        tensor.data_ptr()
                        for tensor in (*state.scales, state.input_scale)
                    ],
                    pointers[mode],
                )
            torch.testing.assert_close(
                states["w4a4", False].scales[0], torch.tensor([2.0, 4.0])
            )
            torch.testing.assert_close(
                states["w4a16", False].scales[0], torch.tensor([4.0, 8.0])
            )
            torch.testing.assert_close(
                states["w4a4", False].scales[2], torch.tensor([6.0, 8.0])
            )
            torch.testing.assert_close(
                states["w4a16", False].scales[2], torch.tensor([24.0, 32.0])
            )


if __name__ == "__main__":
    unittest.main()

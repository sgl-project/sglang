"""Unit tests for srt/layers/quantization/fp8.py MoE runner ownership.

`--moe-runner-backend flashinfer_trtllm[_routed]` is a global setting, so
`Fp8MoEMethod.process_weights_after_loading` must also require that this
instance owns a MoeRunner before materializing the TRT-LLM SwiGLU params:
the MxFP4 wrapper methods borrow an `Fp8MoEMethod` for weight loading only
and never give it a `moe_runner_config` (issue #36264).
"""

import math
import sys
import types
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import (
    TRITON_PADDING_SIZE,
    MoeA2ABackend,
    MoeRunnerBackend,
    RoutingMethodType,
    get_moe_padding_size,
)
from sglang.srt.layers.quantization import fp8 as fp8_module
from sglang.srt.layers.quantization import mxfp4_flashinfer_cutlass_moe as cutlass_mxfp4
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    HybridFp8NvFp4Config,
    ModelOptFp4Config,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    Mxfp4FlashinferTrtllmMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4_humming_moe import Mxfp4HummingMoEMethod
from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_ACTIVATION_PARAMS = ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")


class TestFp8MoEMarlinOwnership(CustomTestCase):
    @staticmethod
    def _methods():
        from compressed_tensors.quantization import QuantizationArgs

        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod

        for family, config in (
            ("fp8", dict(weight_block_size=[128, 128])),
            ("mxfp8", dict(use_mxfp8=True)),
        ):
            yield (
                family,
                Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True, **config)),
            )
        yield "fp8_channel", W8A8FP8MoEMethod(None)
        for family, weights, inputs in (
            (
                "fp8",
                dict(strategy="block", block_structure=[128, 128]),
                dict(strategy="group", group_size=128),
            ),
            ("fp8_channel", dict(strategy="channel"), dict(strategy="token")),
            ("fp8_tensor", dict(strategy="tensor"), dict(strategy="tensor")),
        ):
            yield (
                family,
                CompressedTensorsW8A8Fp8MoE(
                    QuantizationArgs(num_bits=8, type="float", **weights),
                    QuantizationArgs(num_bits=8, type="float", dynamic=True, **inputs),
                ),
            )

    def test_only_lora_marlin_owns_and_repacks_fp8_experts(self):
        from sglang.srt.layers.quantization import marlin_utils_fp8
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        from sglang.srt.lora.moe.families import family_of_layer
        from sglang.srt.lora.moe.quant_info import MoeLoraFp8MarlinQuantInfo

        with get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN):
            for _, method in self._methods():
                self.assertFalse(method.use_marlin, type(method).__name__)

        with (
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_MARLIN),
            patch.object(
                marlin_utils_fp8,
                "gptq_marlin_repack",
                lambda b_q_weight, perm, size_k, size_n, num_bits: b_q_weight.reshape(
                    size_k // 16, size_n * 4
                ),
                create=True,
            ),
        ):
            for family, method in self._methods():
                with self.subTest(family=family, method=type(method).__name__):
                    layer = torch.nn.Module()
                    layer.moe_tp_size = 1
                    layer.use_presharded_weights = False
                    layer.hidden_size = 256
                    layer.intermediate_size_per_partition = 128
                    layer.num_local_experts = 2
                    layer.params_dtype = torch.bfloat16
                    layer.moe_runner_config = MoeRunnerConfig()
                    layer.quant_method = method
                    if isinstance(method, CompressedTensorsW8A8Fp8MoE):
                        layer.quant_method = CompressedTensorsFusedMoEMethod(None)
                        layer.scheme = method
                    loader = layer.quant_method
                    loader.create_weights(layer, 2, 256, 128, torch.bfloat16)
                    loader.create_moe_runner(layer, layer.moe_runner_config)
                    self.assertIsNone(method.runner)
                    if isinstance(method, CompressedTensorsW8A8Fp8MoE):
                        self.assertEqual(
                            layer.w2_weight_scale.shape,
                            {
                                "fp8": (2, 2, 1),
                                "fp8_channel": (2, 256, 1),
                                "fp8_tensor": (2,),
                            }[family],
                        )
                    for param in layer.parameters():
                        param.data.fill_(127 if param.dtype == torch.uint8 else 1)
                    loader.process_weights_after_loading(layer)
                    self.assertEqual(layer.w13_weight.shape, (2, 16, 1024))
                    self.assertEqual(layer.w2_weight.shape, (2, 8, 1024))
                    self.assertEqual(layer.w13_weight.dtype, torch.int32)
                    self.assertEqual(layer.w2_weight_scale.dtype, torch.bfloat16)
                    self.assertEqual(family_of_layer(layer), family)
                    if family == "mxfp8":
                        info = MoeLoraFp8MarlinQuantInfo.from_layer(layer)
                        self.assertEqual(info.group_size, 32)
                        self.assertTrue((info.w2_scales == 2.0**120).all())


class TestFp8MoERunnerOwnership(CustomTestCase):
    def setUp(self):
        moe = get_flags().moe
        self._saved_runner_backend = moe.runner_backend
        moe.runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM
        # _use_hip_int4 would divert this to the ROCm int4 branch, past the guard.
        hip_int4 = patch("sglang.srt.layers.quantization.fp8._use_hip_int4", False)
        hip_int4.start()
        self.addCleanup(hip_int4.stop)

    def tearDown(self):
        get_flags().moe.runner_backend = self._saved_runner_backend

    @staticmethod
    def _make_block_fp8_method() -> Fp8MoEMethod:
        # The real constructor's _owns_moe_runner default is what a delegate relies on.
        return Fp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        )

    @staticmethod
    def _make_layer(num_local_experts: int = 2) -> SimpleNamespace:
        return SimpleNamespace(
            num_local_experts=num_local_experts,
            w13_weight=torch.empty(num_local_experts, 4),
        )

    def _run_post_load(self, method: Fp8MoEMethod, layer: SimpleNamespace) -> None:
        with patch.object(method, "process_weights_after_loading_block_quant") as work:
            method.process_weights_after_loading(layer)
        work.assert_called_once_with(layer)

    def _assert_activation_params_absent(self, layer: SimpleNamespace) -> None:
        for name in _ACTIVATION_PARAMS:
            self.assertFalse(hasattr(layer, f"_flashinfer_trtllm_{name}"))

    def test_fp4_weight_creation_preserves_e8m0_scales(self):
        """Both FP4 loaders create E8M0 scales without duplicate dtype arguments."""
        from sglang.srt.lora.moe.families import family_of_layer

        with (
            get_context().override_server_args(model_path="dummy", device="cpu"),
            patch.object(cutlass_mxfp4, "is_flashinfer_available", return_value=True),
        ):
            for backend, use_mxfp8, dtype in (
                (MoeRunnerBackend.FLASHINFER_MXFP4, False, torch.float8_e8m0fnu),
                (MoeRunnerBackend.LORA_CUTEDSL, False, torch.float8_e8m0fnu),
                (MoeRunnerBackend.LORA_CUTEDSL, True, torch.float8_e8m0fnu),
                (MoeRunnerBackend.AUTO, False, torch.float32),
            ):
                with (
                    self.subTest(backend=backend, use_mxfp8=use_mxfp8),
                    get_flags().moe.override(runner_backend=backend),
                ):
                    method = Fp8MoEMethod(
                        Fp8Config(
                            is_checkpoint_fp8_serialized=True,
                            weight_block_size=None if use_mxfp8 else [128, 128],
                            use_mxfp8=use_mxfp8,
                            is_fp4_experts=True,
                        )
                    )
                    if backend.is_flashinfer_mxfp4():
                        method = cutlass_mxfp4.Mxfp4FlashinferCutlassMoEMethod(
                            method, prefix=""
                        )
                    layer = torch.nn.Module()
                    layer.moe_tp_size = 1
                    layer.moe_runner_config = MoeRunnerConfig()
                    method.create_weights(layer, 2, 128, 128, torch.bfloat16)
                    self.assertEqual(layer.w13_weight_scale_inv.dtype, dtype)
                    self.assertEqual(layer.w2_weight_scale_inv.dtype, dtype)
                    if use_mxfp8:
                        layer.quant_method = method
                        layer.params_dtype = torch.bfloat16
                        self.assertEqual(family_of_layer(layer), "mxfp4")

    def test_borrowed_delegate_skips_trtllm_activation_params(self):
        """A method with no MoeRunner must not read moe_runner_config; doing so
        aborts weight loading whenever a TRT-LLM runner backend is selected."""
        method = self._make_block_fp8_method()
        layer = self._make_layer()

        self._run_post_load(method=method, layer=layer)

        self._assert_activation_params_absent(layer)

    def test_owning_method_prepares_trtllm_activation_params(self):
        """The owning method must still materialize the params it consumes;
        apply() dereferences layer._flashinfer_trtllm_* on the TRT-LLM branch."""
        method = self._make_block_fp8_method()
        layer = self._make_layer()
        method.create_moe_runner(
            layer=layer,
            moe_runner_config=MoeRunnerConfig(
                gemm1_alpha=1.5, gemm1_beta=0.25, gemm1_clamp_limit=None
            ),
        )

        self._run_post_load(method=method, layer=layer)

        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_alpha,
                torch.full((2,), 1.5, dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                layer._flashinfer_trtllm_gemm1_beta,
                torch.full((2,), 0.25, dtype=torch.float32),
            )
        )
        # None stays None: a zero-filled tensor would not mean "no clamp".
        self.assertIsNone(layer._flashinfer_trtllm_gemm1_clamp_limit)

    def test_owning_method_skips_params_on_non_trtllm_backend(self):
        """Ownership alone must not materialize params no kernel consumes."""
        get_flags().moe.runner_backend = MoeRunnerBackend.TRITON
        method = self._make_block_fp8_method()
        layer = self._make_layer()
        method.create_moe_runner(
            layer=layer, moe_runner_config=MoeRunnerConfig(gemm1_alpha=1.5)
        )

        self._run_post_load(method=method, layer=layer)

        self._assert_activation_params_absent(layer)


class TestMxfp4MoEMethodContract(CustomTestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(
            get_context().override_server_args(model_path="dummy", device="cpu")
        )
        self.stack.enter_context(
            get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE)
        )
        self.stack.enter_context(
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
            )
        )
        # Keep real method selection, constructors and layer policy binding;
        # replace expert storage and native runner construction with CPU fixtures.
        adapters = (
            cutlass_mxfp4.Mxfp4FlashinferCutlassMoEMethod,
            Mxfp4FlashinferTrtllmMoEMethod,
            Mxfp4HummingMoEMethod,
            Mxfp4MarlinMoEMethod,
        )
        for adapter in adapters:
            self.stack.enter_context(
                patch.object(
                    adapter, "create_weights", staticmethod(self._create_cpu_weights)
                )
            )
        for module in (
            "sglang.srt.layers.moe.moe_runner",
            "sglang.srt.layers.moe.moe_runner.runner",
        ):
            self.stack.enter_context(
                patch(f"{module}.MoeRunner", side_effect=self._make_runner)
            )
        self.stack.enter_context(
            patch.object(cutlass_mxfp4, "is_flashinfer_available", return_value=True)
        )
        self.stack.enter_context(
            patch.object(
                Fp8MoEMethod,
                "get_moe_routing_method_type",
                side_effect=AssertionError("The borrowed FP8 loader owns no runner"),
            )
        )

    @staticmethod
    def _create_cpu_weights(layer, num_experts, **kwargs):
        layer.w13_weight = torch.empty((num_experts, 1), device="cpu")

    @staticmethod
    def _make_runner(runner_backend, config):
        return SimpleNamespace(runner_backend=runner_backend, config=config)

    @staticmethod
    def _fp4_expert_config():
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            is_fp4_experts=True,
        )

    def _check_construction(self, config, backend, platform, expected_type, prefix):
        with (
            get_flags().moe.override(runner_backend=backend),
            patch.object(fp8_module, "get_platform", return_value=platform),
            patch.object(cutlass_mxfp4, "get_platform", return_value=platform),
        ):
            for requested in (
                None,
                RoutingMethodType.Default,
                RoutingMethodType.DeepSeekV3,
            ):
                with self.subTest(requested=requested):
                    layer = FusedMoE(
                        num_experts=2,
                        hidden_size=128,
                        intermediate_size=128,
                        layer_id=0,
                        top_k=1,
                        quant_config=config,
                        prefix=prefix,
                        routing_method_type=requested,
                    )
                    self.assertIs(type(layer.quant_method), expected_type)
                    self.assertIs(layer.routing_method_type, requested)
                    self.assertIs(
                        layer.moe_runner_config.routing_method_type, requested
                    )
                    expected_backend = (
                        MoeRunnerBackend.MARLIN
                        if backend is MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN
                        else backend
                    )
                    self.assertEqual(layer.runner_backend, expected_backend)
                    self.assertIs(layer.quant_method._fp8.quant_config, config)
                    self.assertFalse(layer.quant_method._fp8._owns_moe_runner)

    def test_mxfp4_merge_applies_routed_scale_once(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            _fuses_routed_scaling_factor_in_topk,
        )
        from sglang.srt.layers.quantization.expert_pack import ExpertPackMoEMethod
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            maybe_fuse_routed_scale_and_shared_add,
        )

        marlin = Mxfp4MarlinMoEMethod(None, "")
        for method, backend, folded_in_topk, routed in (
            (marlin, MoeRunnerBackend.LORA_MARLIN, False, 5.0),
            (marlin, MoeRunnerBackend.LORA_CUTEDSL, False, 5.0),
            (marlin, MoeRunnerBackend.LORA_TRITON, False, 5.0),
            (marlin, MoeRunnerBackend.MARLIN, True, 5.0),
            (ExpertPackMoEMethod(None, ""), None, False, 2.0),
        ):
            self.assertEqual(
                _fuses_routed_scaling_factor_in_topk(method, backend), folded_in_topk
            )
            experts = SimpleNamespace(
                quant_method=method,
                moe_runner_config=MoeRunnerConfig(runner_backend=backend),
                should_fuse_routed_scaling_factor_in_topk=folded_in_topk,
            )
            for has_shared in (False, True):
                with self.subTest(backend=backend, has_shared=has_shared):
                    shared = torch.tensor([1.0]) if has_shared else None
                    actual = maybe_fuse_routed_scale_and_shared_add(
                        experts, torch.tensor([routed]), shared, 2.5
                    )
                    torch.testing.assert_close(
                        actual, torch.tensor([6.0 if has_shared else 5.0])
                    )

    def test_fp4_expert_adapters_preserve_routing_policy(self):
        """Each selected MXFP4 adapter must construct without borrowing its FP8
        loader's routing policy, including an explicit zero-valued policy."""
        cases = (
            (MoeRunnerBackend.MARLIN, False, False, Mxfp4MarlinMoEMethod),
            (
                MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN,
                True,
                False,
                Mxfp4MarlinMoEMethod,
            ),
            (MoeRunnerBackend.HUMMING, False, False, Mxfp4HummingMoEMethod),
            (
                MoeRunnerBackend.FLASHINFER_MXFP4,
                True,
                False,
                cutlass_mxfp4.Mxfp4FlashinferCutlassMoEMethod,
            ),
            (
                MoeRunnerBackend.FLASHINFER_MXFP4,
                False,
                True,
                cutlass_mxfp4.Mxfp4FlashinferCutlassMoEMethod,
            ),
            (
                MoeRunnerBackend.FLASHINFER_MXFP4,
                False,
                False,
                Mxfp4FlashinferTrtllmMoEMethod,
            ),
        )
        for backend, sm90, sm120, expected_type in cases:
            with self.subTest(backend=backend, sm90=sm90, sm120=sm120):
                self._check_construction(
                    config=self._fp4_expert_config(),
                    backend=backend,
                    platform=SimpleNamespace(is_sm90=sm90, is_sm120=sm120),
                    expected_type=expected_type,
                    prefix="model.layers.0.mlp.experts",
                )

    def test_hybrid_mtp_adapter_preserves_routing_policy(self):
        """Excluded MXFP4 MTP experts retain the same construction contract as
        adapters selected directly by Fp8Config."""
        prefix = "model.mtp.experts"
        config = HybridFp8NvFp4Config(
            self._fp4_expert_config(),
            ModelOptFp4Config(exclude_modules=[prefix]),
        )
        self._check_construction(
            config=config,
            backend=MoeRunnerBackend.FLASHINFER_MXFP4,
            platform=SimpleNamespace(is_sm90=False, is_sm120=False),
            expected_type=Mxfp4FlashinferTrtllmMoEMethod,
            prefix=prefix,
        )


class TestMxfp4MarlinPlatformGate(CustomTestCase):
    def test_sm100_is_admitted_under_lora_marlin_only(self):
        from sglang.srt.layers.quantization import mxfp4_marlin_moe

        method = Mxfp4MarlinMoEMethod(
            SimpleNamespace(process_weights_after_loading=lambda layer: None),
            prefix="model.layers.0.mlp.experts",
        )
        with (
            patch.object(
                mxfp4_marlin_moe,
                "get_platform",
                return_value=SimpleNamespace(
                    is_sm90=False, is_sm100=True, is_sm120=False
                ),
            ),
            # Stop after the platform gate, before CUDA repacking.
            patch(
                "sglang.srt.layers.quantization.marlin_utils."
                "check_moe_marlin_supports_layer",
                return_value=False,
            ),
        ):
            for backend, error in (
                (MoeRunnerBackend.MARLIN, "requires SM90 or SM120"),
                (MoeRunnerBackend.LORA_MARLIN, "does not satisfy Marlin"),
            ):
                with (
                    get_flags().moe.override(runner_backend=backend),
                    self.assertRaisesRegex(RuntimeError, error),
                ):
                    method.process_weights_after_loading(SimpleNamespace())


class TestFp8MoEAiterQuantInfo(CustomTestCase):
    """maybe_get_hip_aiter_quant_info assembles what the AITER runner consumes.

    The gfx950 e2e builds AiterMoeQuantInfo by hand, so dropping the gate/up
    layout or the clamp here would leave it passing while served experts read
    the gate and up halves swapped.
    """

    def test_block_fp8_forwards_separated_layout_and_clamp(self):
        method = Fp8MoEMethod(
            Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        )
        # create_moe_runner is not called: it resolves a global backend and
        # builds a MoeRunner, none of which this assembly reads.
        method.moe_runner_config = MoeRunnerConfig(swiglu_limit=10.0)
        layer = SimpleNamespace(
            w13_weight=torch.zeros((1, 4, 4), dtype=torch.float8_e4m3fn),
            w2_weight=torch.zeros((1, 4, 2), dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            w2_weight_scale_inv=torch.ones((1, 4, 1), dtype=torch.float32),
            hidden_pad=0,
            intermediate_pad=0,
            _aiter_gate_up_interleaved=False,
            dispatcher=SimpleNamespace(expert_mask_gpu=torch.tensor([True, False])),
        )
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )

        with (
            patch.dict(sys.modules, {"aiter.ops.flydsl.moe_common": fake_moe_common}),
            patch.object(fp8_module, "_use_aiter", True),
        ):
            quant_info = method.maybe_get_hip_aiter_quant_info(layer)

        self.assertIsNotNone(quant_info)
        self.assertEqual(quant_info.quant_type, AiterQuantType.PER_128X128)
        self.assertEqual(quant_info.swiglu_limit, 10.0)
        self.assertEqual(quant_info.fused_moe_kwargs, {"gate_mode": "separated"})
        self.assertIs(quant_info.expert_mask, layer.dispatcher.expert_mask_gpu)


class TestFp8MoEAiterWeightLayout(CustomTestCase):
    @staticmethod
    def _make_layer():
        layer = torch.nn.Module()
        for name, shape in (("w13_weight", (2, 32, 16)), ("w2_weight", (2, 16, 16))):
            weight = (torch.arange(math.prod(shape)).reshape(shape) % 13).to(
                torch.float8_e4m3fn
            )
            weight.view(torch.int8).reshape(-1)[0] = -128
            layer.register_parameter(
                name, torch.nn.Parameter(weight, requires_grad=False)
            )
            layer.register_parameter(
                f"{name}_scale_inv",
                torch.nn.Parameter(torch.ones((2, 1, 1)), requires_grad=False),
            )
        return layer

    @staticmethod
    def _shuffle(weight, block):
        return weight.view(torch.uint8).flip(-1).contiguous().view(weight.dtype)

    def _check_layout(self, fnuz):
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
            for backend in (MoeRunnerBackend.TRITON, None, MoeRunnerBackend.AITER):
                with self.subTest(backend=backend, fnuz=fnuz):
                    method = Fp8MoEMethod(
                        Fp8Config(
                            is_checkpoint_fp8_serialized=True,
                            weight_block_size=[128, 128],
                        )
                    )
                    if backend is not None:
                        method.runner = SimpleNamespace(runner_backend=backend)
                    layer = self._make_layer()
                    expected = {}
                    for name in ("w13_weight", "w2_weight"):
                        weight = getattr(layer, name).detach().clone()
                        if fnuz:
                            weight, _, _ = fp8_module.normalize_e4m3fn_to_e4m3fnuz(
                                weight, getattr(layer, f"{name}_scale_inv"), None
                            )
                        if backend == MoeRunnerBackend.AITER:
                            weight = self._shuffle(weight, (16, 16))
                        expected[name] = weight
                    with (
                        patch.object(fp8_module, "_is_fp8_fnuz", fnuz),
                        patch.object(fp8_module, "_use_aiter", True),
                        patch.object(
                            fp8_module, "shuffle_weight", self._shuffle, create=True
                        ),
                    ):
                        method.process_weights_after_loading_block_quant(layer)
                    for name, weight in expected.items():
                        actual = getattr(layer, name)
                        self.assertEqual(actual.dtype, weight.dtype)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.uint8), weight.view(torch.uint8)
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                getattr(layer, f"{name}_scale_inv"),
                                torch.full((2, 1, 1), 2.0 if fnuz else 1.0),
                            )
                        )

    def test_fnuz_shuffles_only_for_aiter_runner(self):
        self._check_layout(fnuz=True)

    def test_non_fnuz_shuffles_only_for_aiter_runner(self):
        self._check_layout(fnuz=False)


class TestMoePaddingEnv(CustomTestCase):
    def test_moe_padding_accepts_bool_spellings(self):
        """`SGLANG_MOE_PADDING=true` crashed the import of the Triton MoE modules."""
        for value, expected in (
            ("true", TRITON_PADDING_SIZE),
            ("1", TRITON_PADDING_SIZE),
            ("false", 0),
            ("0", 0),
        ):
            with self.subTest(value=value), envs.SGLANG_MOE_PADDING.override(value):
                self.assertEqual(get_moe_padding_size(False), expected)


if __name__ == "__main__":
    unittest.main()

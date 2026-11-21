from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMxfp4PreparedQuantInfo(CustomTestCase):
    def test_loader_padded_hidden_width_is_rejected(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp4MarlinQuantInfo

        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=torch.zeros((2, 16, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 512), dtype=torch.int32),
            w13_weight_scale=torch.zeros((2, 8, 256), dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            w2_weight_scale=torch.zeros((2, 4, 256), dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=200,
        )
        with self.assertRaisesRegex(NotImplementedError, "hidden_size % 256"):
            MoeLoraMxfp4MarlinQuantInfo.from_layer(layer)


class TestMxfp8Admission(CustomTestCase):
    def test_unaligned_hidden_size_is_refused(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraMxfp8QuantInfo

        layer = SimpleNamespace(
            params_dtype=torch.bfloat16,
            w13_weight=torch.zeros((2, 128, 192), dtype=torch.float8_e4m3fn),
            w13_weight_scale_inv=torch.zeros((2, 128, 6), dtype=torch.uint8),
            w2_weight=torch.zeros((2, 192, 64), dtype=torch.float8_e4m3fn),
            w2_weight_scale_inv=torch.zeros((2, 192, 2), dtype=torch.uint8),
            num_local_experts=2,
            intermediate_size_per_partition=64,
            hidden_size=192,
        )
        with self.assertRaisesRegex(NotImplementedError, "hidden_size % 128"):
            MoeLoraMxfp8QuantInfo.from_layer(layer)


class TestProviderGeometry(CustomTestCase):
    def test_triton_fp8_k_tiles_stay_within_scale_groups(self):
        from sglang.srt.lora.moe.base_gemm_provider.triton import (
            TritonBf16PairProvider,
        )

        # The FP8 provider shares the method; the stand-in below carries its
        # block shape and dtype tag.
        config_for = TritonBf16PairProvider._config_for
        module = types.ModuleType(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
        )
        # The FP8 GEMM loads one scale per K block: a 256-wide block over
        # [128, 128] scale groups would apply the first group's scale to its
        # second half, and upstream never clamps override or table configs.
        original = {"BLOCK_SIZE_K": 256, "USE_TMA": True}
        upstream = (dict(original), dict(original))
        module.try_get_optimal_moe_config = mock.Mock(
            return_value=(upstream[0], (upstream[1], None))
        )
        provider = SimpleNamespace(
            quant_info=SimpleNamespace(
                w13_weight=SimpleNamespace(shape=(2, 256, 128)),
                w2_weight=SimpleNamespace(shape=(2, 128, 128)),
            ),
            _config_dtype_tag="fp8_w8a8",
            _config_block_shape=[128, 128],
            _config_per_channel_quant=False,
            _configs={},
        )
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            resolved = config_for(provider, 32, 2)
            self.assertIs(config_for(provider, 32, 2), resolved)
        for actual, cached in zip(resolved, upstream):
            self.assertEqual(actual, {"BLOCK_SIZE_K": 128})
            self.assertEqual(cached, original)
            self.assertIsNot(actual, cached)
        module.try_get_optimal_moe_config.assert_called_once()


class TestDerivedBaseWeights(CustomTestCase):
    def test_providers_expose_whether_base_weights_are_derived(self):
        from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls

        cases = (
            ("nvfp4", "cutedsl", True),
            ("fp8", "cutedsl", False),
            ("bf16", "triton", False),
        )
        for family, vendor, derived in cases:
            cls = select_provider_cls("expert_major", family, vendor)
            self.assertEqual(
                cls.derived_base_weights is not None, derived, cls.__name__
            )


class TestNvFp4LayerScales(CustomTestCase):
    @staticmethod
    def _layer(*, per_token):
        def scales(n):
            return torch.ones((2, n, 8)).to(torch.float8_e4m3fn)

        return SimpleNamespace(
            params_dtype=torch.bfloat16,
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=128,
            moe_runner_config=SimpleNamespace(is_gated=True),
            w13_weight=torch.zeros((2, 256, 64), dtype=torch.uint8),
            w2_weight=torch.zeros((2, 128, 64), dtype=torch.uint8),
            w13_weight_scale=scales(256),
            w2_weight_scale=scales(128),
            w13_weight_scale_2=torch.tensor([[0.5, 1.0], [2.0, 4.0]]),
            w2_weight_scale_2=torch.tensor([1.0, 2.0]),
            w13_input_scale=torch.tensor([0.25, 0.5]),
            w2_input_scale=torch.tensor([0.25, 0.125]),
            quant_method=SimpleNamespace(
                quant_config=SimpleNamespace(use_per_token_activation=per_token)
            ),
        )

    def test_only_static_alpha_includes_the_activation_scale(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4QuantInfo

        for per_token in (False, True):
            with self.subTest(per_token=per_token):
                layer = self._layer(per_token=per_token)
                info = MoeLoraNvFp4QuantInfo.from_layer(layer)
                self.assertEqual(
                    info.act_scale_mode, "per_token" if per_token else "static"
                )
                static_scale = (None, None) if per_token else (0.5, 0.25)
                self.assertEqual(
                    (info.w13_input_scale, info.w2_input_scale), static_scale
                )
                expected_gate = (
                    [[0.5, 1.0], [2.0, 4.0]] if per_token else [[0.25, 0.5], [1.0, 2.0]]
                )
                expected_down = [[1.0], [2.0]] if per_token else [[0.25], [0.5]]
                torch.testing.assert_close(info.alpha[0], torch.tensor(expected_gate))
                torch.testing.assert_close(info.alpha[1], torch.tensor(expected_down))

    def test_4over6_recipe_is_per_token(self):
        from sglang.srt.environ import envs
        from sglang.srt.lora.moe.quant_info import (
            MoeLoraNvFp4QuantInfo,
            NvFp4ActivationRecipe,
        )

        with (
            envs.FLASHINFER_NVFP4_4OVER6.override(True),
            envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.override(True),
            envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override("mse"),
            envs.FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH.override(True),
        ):
            token = MoeLoraNvFp4QuantInfo.from_layer(self._layer(per_token=True))
            static = MoeLoraNvFp4QuantInfo.from_layer(self._layer(per_token=False))
            self.assertEqual(
                token.recipe,
                NvFp4ActivationRecipe(
                    e4m3_max=256, err_mode="MSE", err_use_fast_math=True
                ),
            )
            self.assertEqual(token.activation_denominator, 1536.0)
            self.assertIsNone(static.recipe)
            self.assertEqual(static.activation_denominator, 2688.0)
            with (
                envs.FLASHINFER_NVFP4_4OVER6_ERR_MODE.override("rms"),
                self.assertRaisesRegex(ValueError, "MAE or MSE"),
            ):
                MoeLoraNvFp4QuantInfo.from_layer(self._layer(per_token=True))


class TestBlockScaledQuantArgs(CustomTestCase):
    def test_quantized_launch_arguments_match_kernel_signatures(self):
        from sglang.kernels.ops.lora.moe.quant_blockscaled import (
            BlockScaledRows,
            act_quant_args,
        )

        codes = torch.empty((128, 64), dtype=torch.uint8)
        storage = dict(
            data=codes,
            scales=torch.empty(1024, dtype=torch.uint8),
            sf_vec_size=16,
            rows_per_slab=128,
        )
        rows = BlockScaledRows(
            **storage,
            row_scales=torch.ones(128),
            row_denominator=1536.0,
            use_4over6=True,
            err_mode="MSE",
        )
        static_rows = BlockScaledRows(**storage, global_scale=0.25)
        quant_args = rows.kernel_args()
        consumers = (
            ("quant_blockscaled", "_quant_rows_kernel", (quant_args,)),
            (
                "dispatch_contiguous",
                "_fill_rows_contiguous_blockscaled_kernel",
                (quant_args,),
            ),
            (
                "dispatch_masked",
                "_dispatch_fill_masked_blockscaled_kernel",
                (quant_args,),
            ),
            (
                "activation_delta",
                "_act_delta_kernel",
                tuple(
                    act_quant_args(target, fallback=codes) for target in (rows, None)
                ),
            ),
            (
                "fused_act",
                "_b_act_kernel",
                tuple(act_quant_args(target) for target in (static_rows, None)),
            ),
        )
        sources = (
            Path(__file__).resolve().parents[4] / "python/sglang/kernels/ops/lora/moe"
        )
        launch_options = {"num_warps", "num_stages", "num_ctas", "maxnreg"}
        for module, kernel, variants in consumers:
            tree = ast.parse((sources / f"{module}.py").read_text())
            signature = next(
                node.args
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == kernel
            )
            positional = [arg.arg for arg in (*signature.posonlyargs, *signature.args)]
            allowed = set(positional) | {arg.arg for arg in signature.kwonlyargs}
            required = set(positional[: len(positional) - len(signature.defaults)])
            required.update(
                arg.arg
                for arg, default in zip(signature.kwonlyargs, signature.kw_defaults)
                if default is None
            )
            (launch,) = (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and getattr(node.func.value, "id", None) == kernel
            )
            self.assertEqual(sum(arg.arg is None for arg in launch.keywords), 1, kernel)
            explicit = positional[: len(launch.args)] + [
                arg.arg
                for arg in launch.keywords
                if arg.arg and arg.arg not in launch_options
            ]
            for quant in variants:
                with self.subTest(kernel=kernel, keys=set(quant)):
                    supplied = explicit + list(quant)
                    # Optional parameters may be absent; each supplied key occurs once.
                    self.assertCountEqual(
                        supplied, required | (allowed & set(supplied))
                    )


class TestActQuantReadiness(CustomTestCase):
    @staticmethod
    def _provider(cls, **attributes):
        provider = object.__new__(cls)
        provider.quant_info = SimpleNamespace(
            num_local_experts=2, intermediate_size=128, hidden_size=128
        )
        provider._launch = mock.Mock()
        for name, value in attributes.items():
            setattr(provider, name, value)
        return provider

    @staticmethod
    def _row_state(cls, **fields):
        import msgspec

        names, defaults = cls.__struct_fields__, cls.__struct_defaults__
        # The defaults tuple omits required fields before the first default.
        defaults = (msgspec.NODEFAULT,) * (len(names) - len(defaults)) + defaults
        required = dict.fromkeys(
            name
            for name, default in zip(names, defaults)
            if default is msgspec.NODEFAULT
        )
        return cls(**(required | fields))

    @staticmethod
    def _activate(provider, row_state, *, fused, tile_width=64):
        from sglang.kernels.ops.lora.moe import activation_delta, fused_act

        with (
            mock.patch.object(activation_delta, "act_delta") as act_delta,
            mock.patch.object(fused_act, "fused_b_act") as fused_b_act,
        ):
            if not fused:
                provider.act_with_delta(row_state, None, None, None, None, None)
                return act_delta.call_args.kwargs["act_quant"]
            provider.fused_act(
                row_state,
                activation="silu",
                base_gateup=None,
                act_rows=None,
                act_pairs=None,
                routing=None,
                config={"BLOCK_SIZE_W": tile_width},
            )
            return fused_b_act.call_args.kwargs["act_quant"]

    def test_block_scaled_down_uses_only_rows_written_by_the_last_activation(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_blockscaled as bs

        for per_token in (False, True):
            with self.subTest(per_token=per_token):
                provider = self._provider(
                    bs.CuteDslNvFp4ContiguousProvider,
                    _per_row=per_token,
                    _rows_spec=mock.Mock(return_value=object()),
                    act_out_shape=lambda state: (8, 128),
                    _quantize_rows=mock.Mock(),
                )
                state = self._row_state(
                    bs.CuteDslBlockScaledContiguousRowState,
                    act_q="act_q",
                    act_sf="act_sf",
                    act_rs="act_rs" if per_token else None,
                    pair_to_row="pair_to_row",
                )
                act_out = torch.zeros((8, 128), dtype=torch.bfloat16)
                for fused, width, ready in (
                    (False, 32, True),
                    (True, 24, False),
                    (True, 32, not per_token),
                ):
                    target = self._activate(
                        provider, state, fused=fused, tile_width=width
                    )
                    self.assertEqual(target is not None, ready)
                    self.assertEqual(state.act_quant_ready, ready)
                    provider._quantize_rows.reset_mock()
                    provider.down(state, act_out, "out")
                    self.assertEqual(provider._quantize_rows.called, not ready)

    def test_fp8_down_quantizes_after_fused_activation(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_fp8

        for cls, group in (
            (cutedsl_fp8.CuteDslFp8ContiguousProvider, 128),
            (cutedsl_fp8.CuteDslFp8ChannelContiguousProvider, 256),
        ):
            with (
                self.subTest(provider=cls.__name__),
                mock.patch.object(
                    cutedsl_fp8, "quantize_pair_rows_fp8", return_value=("q", "scales")
                ) as quantize,
            ):
                provider = self._provider(cls)
                provider.quant_info.intermediate_size = 256
                state = self._row_state(
                    provider._row_state_cls,
                    act_fp8="act_fp8",
                    act_scale="act_scale",
                    pair_to_row="pair_to_row",
                )
                self.assertEqual(
                    self._activate(provider, state, fused=False),
                    ("act_fp8", "act_scale", group),
                )
                provider.down(state, "act_out", "out")
                quantize.assert_not_called()
                self.assertIsNone(self._activate(provider, state, fused=True))
                provider.down(state, "act_out", "out")
                quantize.assert_called_once_with(
                    "act_out",
                    q_out="act_fp8",
                    scale_out="act_scale",
                    pair_to_row="pair_to_row",
                    group=group,
                )

    def test_triton_fp8_down_takes_rows_only_from_the_activation_that_just_ran(self):
        from sglang.srt.lora.moe.base_gemm_provider import triton

        provider = self._provider(
            triton.TritonFp8PairProvider,
            act_out_shape=lambda row_state: (4, 128),
            _invoke=mock.Mock(),
        )
        provider.quant_info.w2_weight = provider.quant_info.w2_scale = None
        state = self._row_state(
            triton.TritonRowState,
            pair_to_row=torch.zeros(4, dtype=torch.int32),
            workspace=None,
            down_config={},
        )

        def down_input():
            provider._invoke.reset_mock()
            provider.down(state, "act_out", "out")
            call = provider._invoke.call_args
            return call.args[0], call.kwargs["a_scale"]

        self.assertIsNotNone(self._activate(provider, state, fused=False))
        rows, scales = state.act_quant
        self.assertTrue(state.act_quant_ready)
        self.assertEqual([id(x) for x in down_input()], [id(rows), id(scales)])
        # Fused B/activation leaves stale FP8 buffers, so down quantizes BF16 again.
        self.assertIsNone(self._activate(provider, state, fused=True))
        self.assertIs(state.act_quant[0], rows)
        self.assertFalse(state.act_quant_ready)
        self.assertEqual(down_input(), ("act_out", None))


class TestCompileCache(CustomTestCase):
    def test_stage_constants_separate_compiled_code(self):
        """Different gate/up and down epilogue constants must compile separately."""
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_common

        class Provider(cutedsl_common.CuteDslTileMixin):
            _ROW_MODE = "masked"
            _DTYPE_TAG = "nvfp4"

            def __init__(self):
                self.quant_info = SimpleNamespace(num_local_experts=2)
                self._compiled = {64: {}}
                self._as_dynamic_cute_tensor = lambda tensor, leading_dim: tensor
                self.compiles = 0

            def _stage_weight(self, stage):
                return torch.zeros((2, 256, 128), dtype=torch.uint8)

            def _stage_constexprs(self, stage):
                return (16, True, 128 if stage == "gemm1" else 1 << 30)

            def _prepare_dummy(self, stage, config):
                self.compiles += 1
                return SimpleNamespace(launch=lambda: None, compiled_fn=object())

        with mock.patch.dict(cutedsl_common._COMPILE_CACHE, clear=True):
            provider = Provider()
            provider._compile_stage(64, "gemm1", config="tile")
            provider._compile_stage(64, "gemm2", config="tile")
            self.assertEqual(provider.compiles, 2)
            calls = provider._compiled[64]
            self.assertIsNot(calls["gemm1"].compiled_fn, calls["gemm2"].compiled_fn)
            # The same geometry and constants reuse the compiled code.
            provider._compile_stage(64, "gemm1", config="tile")
            self.assertEqual(provider.compiles, 2)


class TestBlockScaledProvider(CustomTestCase):
    def test_masked_buffers_keep_the_expert_dimension_for_one_expert(self):
        from sglang.srt.lora.moe.base_gemm_provider import cutedsl_blockscaled as bs

        quant_buffers = bs._CuteDslBlockScaledMixin._quant_buffers
        provider = SimpleNamespace(
            _pack=2,
            _per_row=False,
            _w13=torch.zeros(1, dtype=torch.uint8),
            SF_VEC_SIZE=32,
            _scale_atom_bytes=lambda rows, k, vec, slabs=1: slabs * rows * k // vec,
        )
        # The masked kernels address [experts, rows, K] slabs for one expert too.
        data, scales, row_scales = quant_buffers(
            provider, None, "masked:rows", (1, 64), 256, "cpu"
        )
        self.assertIsNone(row_scales)  # static mode: no stored row scales
        self.assertEqual(tuple(data.shape), (1, 64, 128))
        self.assertEqual(scales.numel(), 64 * 256 // 32)


class TestMaterializedFinalize(CustomTestCase):
    """Execute the host dispatch with real CPU buffers and mocked GPU launches."""

    def test_materialized_finalize_preserves_strided_weights(self):
        from sglang.srt.lora.moe.base_gemm_provider.base import MoeBaseProvider

        finalize = types.ModuleType("sglang.kernels.ops.lora.moe.finalize")
        finalize.invoke_small_finalize = mock.Mock()
        reorder = types.ModuleType("sglang.kernels.ops.moe.ep_moe_kernels")
        reorder.post_reorder_deepgemm = mock.Mock()
        # Production hands over strided top-k weight views; post_reorder_deepgemm
        # addresses them contiguously, so even a small batch must not take the
        # gather path and must receive a contiguous copy.
        tokens = 64
        weights = torch.arange(tokens * 6).view(tokens, 6)[:, ::2]
        ids = torch.zeros((tokens, 3), dtype=torch.int64)
        output = torch.empty((tokens, 5))
        delta = torch.empty((tokens, 3, 5))
        with mock.patch.dict(
            sys.modules, {finalize.__name__: finalize, reorder.__name__: reorder}
        ):
            MoeBaseProvider().finalize(
                SimpleNamespace(pair_to_row=torch.arange(tokens * 3)),
                torch.empty((tokens * 3, 5)),
                ids,
                weights,
                2.5,
                output,
                lora_delta=delta,
            )
        finalize.invoke_small_finalize.assert_not_called()
        call = reorder.post_reorder_deepgemm.call_args
        actual = call.args[4]
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, weights)
        self.assertIs(call.kwargs["lora_delta"], delta)


class TestRunnerAdmission(CustomTestCase):
    """Layers the runner would otherwise serve with silently wrong numerics,
    and the plan check binding must surface. The device capability is the
    only GPU boundary, pinned to SM90."""

    def setUp(self):
        patcher = mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _base_layer(**overrides):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        quant_method = object.__new__(UnquantizedFusedMoEMethod)
        dispatcher = object.__new__(StandardDispatcher)
        dispatcher.skip_local_expert_mapping = overrides.get("skip_local", False)
        return SimpleNamespace(
            quant_method=quant_method,
            dispatcher=dispatcher,
            w13_weight=torch.zeros(2, 8, 4, dtype=torch.bfloat16),
            w2_weight=torch.zeros(2, 4, 4, dtype=torch.bfloat16),
            num_local_experts=2,
            should_fuse_routed_scaling_factor_in_topk=overrides.get(
                "fused_scaling", False
            ),
            with_bias=overrides.get("with_bias", False),
            moe_runner_config=SimpleNamespace(
                activation="silu",
                is_gated=True,
                gemm1_alpha=None,
                gemm1_clamp_limit=None,
                swiglu_limit=None,
                apply_router_weight_on_input=overrides.get(
                    "apply_router_weight_on_input", False
                ),
                no_combine=overrides.get("no_combine", False),
                num_fused_shared_experts=overrides.get("num_fused_shared_experts", 0),
                top_k=2,
                routed_scaling_factor=1.0,
            ),
        )

    def test_rejects_layers_it_would_serve_wrongly(self):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        cases = {
            # Global expert IDs would index the wrong rank's experts.
            "global_expert_ids": (dict(skip_local=True), "EP-local"),
            # Scaling folded into top-k would be applied twice.
            "prefolded_routed_scaling": (dict(fused_scaling=True), "routed scaling"),
            # Providers add no bias term; a biased layer would lose it.
            "expert_biases": (dict(with_bias=True), "expert biases"),
            # The runner weights routes in finalize, never on the input rows.
            "router_weight_on_input": (
                dict(apply_router_weight_on_input=True),
                "owned by finalize",
            ),
            # The runner's finalize always combines the top-k rows.
            "no_combine": (dict(no_combine=True), "owned by finalize"),
            "fused_shared_experts": (
                dict(num_fused_shared_experts=1),
                "fused shared experts",
            ),
        }
        for name, (overrides, message) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                MoeLoraRunner._admit(self._base_layer(**overrides))

    def test_binding_surfaces_plan_validation(self):
        """A provider exposing one gate/up slice cannot serve a gated layer."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.moe import runner as runner_module
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.runtime_context import get_context, get_flags

        class QuantInfo:
            @staticmethod
            def from_layer(base_layer):
                return None

        class Provider:
            contract = SimpleNamespace(quant_info_cls=QuantInfo)
            hidden_size = 2048
            intermediate_size = 768
            num_local_experts = 256
            gate_up_slices = 1

            def __init__(self, quant_info):
                self.quant_info = quant_info

        layer = SimpleNamespace(
            w2_weight=SimpleNamespace(
                device=torch.device("cuda"), shape=(256, 2048, 768)
            ),
            num_local_experts=256,
            hidden_size=2048,
            moe_runner_config=SimpleNamespace(
                activation="silu", top_k=8, routed_scaling_factor=1.0, is_gated=True
            ),
        )
        # from_layer reads the published MoE runner backend for its vendor.
        with (
            mock.patch.object(MoeLoraRunner, "_admit", return_value="bf16"),
            mock.patch.object(
                runner_module, "select_provider_cls", return_value=Provider
            ),
            get_context().override_server_args(),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
        ):
            with self.assertRaisesRegex(ValueError, "gate/up slices"):
                MoeLoraRunner.from_layer(
                    layer, workspace=object(), is_shared_outer=False, physical_rank=64
                )
            Provider.gate_up_slices = 2
            runner = MoeLoraRunner.from_layer(
                layer, workspace=object(), is_shared_outer=False, physical_rank=64
            )
        self.assertIs(runner.providers["expert_major"], runner.providers["route_major"])


class TestLayerRunnerBackendResolution(CustomTestCase):
    def test_v2_binds_the_selected_moe_runner(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.lora.layers import FusedMoEWithLoRA
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        base_layer = SimpleNamespace(
            quant_method=mock.MagicMock(),
            moe_runner_config=MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                layer_id=0,
                top_k=2,
                num_fused_shared_experts=0,
                params_dtype=torch.bfloat16,
                activation="silu",
                inplace=True,
            ),
            dispatcher=None,
            num_local_experts=8,
            should_fuse_routed_scaling_factor_in_topk=False,
            moe_tp_size=1,
            moe_tp_rank=0,
            intermediate_size_per_partition=32,
            runner=SimpleNamespace(runner_backend=MoeRunnerBackend.DEEP_GEMM),
            run_moe_core=mock.Mock(),
        )
        backend = SimpleNamespace(name="triton_v2", lora_workspace=object())
        original_run_moe_core = base_layer.run_moe_core
        with (
            mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.LORA_CUTEDSL,
            ),
            mock.patch.object(MoeLoraRunner, "from_layer") as bind,
        ):
            layer = FusedMoEWithLoRA(
                base_layer,
                backend,
                experts_shared_outer_loras=True,
                max_lora_rank=16,
            )
        self.assertIs(layer._lora_runner_backend, MoeRunnerBackend.LORA_CUTEDSL)
        bind.assert_called_once_with(
            base_layer,
            workspace=backend.lora_workspace,
            is_shared_outer=True,
            physical_rank=16,
        )
        self.assertIs(layer._moe_lora_runner, bind.return_value)
        self.assertIs(layer._base_run_moe_core, original_run_moe_core)
        self.assertEqual(base_layer.run_moe_core, layer._run_moe_core_with_lora)


if __name__ == "__main__":
    unittest.main()

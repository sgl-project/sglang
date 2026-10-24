"""Real provider methods test admission, geometry, and finalization with GPU boundaries mocked."""

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


ROOT = Path(__file__).resolve().parents[4]
PROVIDER = ROOT / "python/sglang/srt/lora/moe/base_gemm_provider"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


class TestNvFp4PreparedQuantInfo(CustomTestCase):
    def test_non_bf16_activations_fail_without_mutation(self):
        from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo

        # FP16 is the silent case: Marlin accepts half activations while the
        # runner allocates BF16 stage buffers; FP32 fails loudly in the JIT.
        layer = SimpleNamespace(
            params_dtype=torch.float16,
            w13_weight=torch.zeros((2, 4, 512), dtype=torch.int32),
            w2_weight=torch.zeros((2, 8, 128), dtype=torch.int32),
            w13_weight_scale=torch.zeros((2, 4, 256), dtype=torch.float8_e4m3fn),
            w2_weight_scale=torch.zeros((2, 8, 64), dtype=torch.float8_e4m3fn),
            w13_weight_scale_2=torch.ones(2, dtype=torch.float16),
            w2_weight_scale_2=torch.zeros(2, dtype=torch.float16),
            moe_runner_config=SimpleNamespace(is_gated=True),
            num_local_experts=2,
            intermediate_size_per_partition=128,
            hidden_size=64,
        )
        original = vars(layer).copy()
        with self.assertRaisesRegex(NotImplementedError, "BF16 activations"):
            MoeLoraNvFp4MarlinQuantInfo.from_layer(layer)
        for name, value in original.items():
            self.assertIs(getattr(layer, name), value)


class TestProviderGeometry(CustomTestCase):
    def test_triton_fp8_k_tiles_stay_within_scale_groups(self):
        source = _function((PROVIDER / "triton_bf16.py").read_text(), "_config_for")
        namespace = {}
        exec("from __future__ import annotations\n" + source, namespace)
        config_for = namespace["_config_for"]
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
        from sglang.srt.lora.moe.runner import MoeLoraRunner
        from sglang.srt.runtime_context import get_context, get_flags

        provider = SimpleNamespace(
            hidden_size=2048,
            intermediate_size=768,
            num_local_experts=256,
            gate_up_slices=1,
            contract=SimpleNamespace(lora_delta_dtype=torch.bfloat16),
        )
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
            mock.patch.object(MoeLoraRunner, "_build_provider", return_value=provider),
            get_context().override_server_args(),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.LORA_CUTEDSL),
            self.assertRaisesRegex(ValueError, "gate/up slices"),
        ):
            MoeLoraRunner.from_layer(
                layer, workspace=object(), is_shared_outer=False, physical_rank=64
            )


class TestLayerRunnerBackendResolution(CustomTestCase):
    """The selected lora runner backend binds the new MoE runner and rebinds
    run_moe_core."""

    def test_v2_accepts_each_new_moe_runner(self):
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

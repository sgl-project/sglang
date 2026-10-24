from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestEngineAdmission(CustomTestCase):
    """Layers the runner would otherwise serve with silently wrong numerics."""

    def _base_layer(self, **overrides):
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

    def _admit(self, **overrides):
        from sglang.srt.lora.moe.runner import MoeLoraRunner

        MoeLoraRunner._admit(self._base_layer(**overrides))

    def test_rejects_global_expert_id_dispatch(self):
        with self.assertRaisesRegex(NotImplementedError, "EP-local"):
            self._admit(skip_local=True)

    def test_rejects_prefolded_routed_scaling(self):
        """Scaling folded into top-k would be applied twice."""
        with self.assertRaisesRegex(NotImplementedError, "routed scaling"):
            self._admit(fused_scaling=True)

    def test_rejects_expert_biases(self):
        """Providers add no bias term; a biased layer would lose it."""
        with self.assertRaisesRegex(NotImplementedError, "expert biases"):
            self._admit(with_bias=True)

    def test_rejects_router_weight_applied_on_input(self):
        """The runner weights routes in finalize, never on the input rows."""
        with self.assertRaisesRegex(NotImplementedError, "owned by finalize"):
            self._admit(apply_router_weight_on_input=True)

    def test_rejects_no_combine(self):
        """The runner's finalize always combines the top-k rows."""
        with self.assertRaisesRegex(NotImplementedError, "owned by finalize"):
            self._admit(no_combine=True)

    def test_rejects_fused_shared_experts(self):
        with self.assertRaisesRegex(NotImplementedError, "fused shared experts"):
            self._admit(num_fused_shared_experts=1)


class TestLayerRunnerBackendResolution(CustomTestCase):
    """The selected lora runner backend binds the new MoE runner and rebinds
    run_moe_core."""

    def test_v2_accepts_each_new_moe_runner(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
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

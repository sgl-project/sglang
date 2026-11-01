"""LoRA MoE must not overwrite inputs shared experts read on another stream.

In-place reduction races with that read during decode graph replay. FusedMoE passes
``inplace=True`` explicitly, so changing the config default does not fix it:
the real LoRA wrapper must disable it on the shared runner config object.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def _make_base_layer() -> types.SimpleNamespace:
    """The attribute surface FusedMoEWithLoRA.__init__ reads off its base FusedMoE."""
    config = MoeRunnerConfig(
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
    )
    # Let AUTO use Triton; quant-info getters remain mocked.
    quant_method = mock.MagicMock()
    quant_method.runner = None
    return types.SimpleNamespace(
        quant_method=quant_method,
        moe_runner_config=config,
        dispatcher=None,
        num_local_experts=8,
        should_fuse_routed_scaling_factor_in_topk=False,
        moe_tp_size=1,
        moe_tp_rank=0,
        intermediate_size_per_partition=32,
        runner=None,
    )


class FusedMoEWithLoRAInplaceTest(unittest.TestCase):
    def test_constructor_forces_inplace_off(self):
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        base_layer = _make_base_layer()
        lora_backend = types.SimpleNamespace(name="triton")
        self.assertTrue(base_layer.moe_runner_config.inplace)
        layer = FusedMoEWithLoRA(base_layer, lora_backend)
        self.assertIs(layer.moe_runner_config, base_layer.moe_runner_config)
        self.assertFalse(layer.moe_runner_config.inplace)


if __name__ == "__main__":
    unittest.main()

"""Regression test: LoRA MoE layers must never run their runner in-place.

The LoRA MoE runner cores write their final reduction into ``output =
hidden_states if inplace else torch.empty_like(hidden_states)``. Dual-stream
MoE forwards (e.g. ``DeepseekV2MoE.forward_normal_dual_stream``) read
``hidden_states`` for the shared experts on the alt stream with no ordering
edge against that write — with ``inplace=True`` this is a write-after-read
race inside the decode cuda graph that intermittently corrupts long LoRA
generations (fixed by forcing ``inplace=False`` at ``FusedMoEWithLoRA``
construction).

These tests execute the real ``FusedMoEWithLoRA.__init__`` against a config
that was explicitly constructed with ``inplace=True`` — exactly how
``FusedMoE.__init__`` builds it from its own signature default — and assert
the constructor forces it off. Asserting on the *shared config object* (not a
copy) also guards the historical footgun where flipping the
``MoeRunnerConfig`` dataclass default silently changed nothing because
``FusedMoE.__init__`` passes ``inplace`` explicitly.

Usage:
    python -m pytest test/registered/unit/lora/test_lora_moe_inplace_unit.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")

import types
import unittest
import unittest.mock as mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig


def _make_base_layer(quant_method=None) -> types.SimpleNamespace:
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
        # FusedMoE.__init__ passes its own `inplace: bool = True` signature
        # default into the config explicitly; mirror that here.
        inplace=True,
    )
    if quant_method is None:
        # `.runner = None` -> AUTO backend falls back to triton; quant-info
        # getters auto-mock.
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
    )


class FusedMoEWithLoRAInplaceTest(unittest.TestCase):
    def _construct(self, quant_method=None):
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        base_layer = _make_base_layer(quant_method)
        lora_backend = types.SimpleNamespace()
        self.assertTrue(base_layer.moe_runner_config.inplace)
        layer = FusedMoEWithLoRA(base_layer, lora_backend)
        return layer, base_layer

    def test_constructor_forces_inplace_off(self):
        layer, base_layer = self._construct()
        # The wrapper and the base layer share one config object; both views
        # (and therefore the MoeRunner built from it) must see inplace=False.
        self.assertIs(layer.moe_runner_config, base_layer.moe_runner_config)
        self.assertFalse(layer.moe_runner_config.inplace)

    def test_marlin_lora_runner_core_sees_non_inplace_config(self):
        # The marlin LoRA runner core is the code that aliases its output onto
        # hidden_states when inplace=True; make sure the core the layer builds
        # is wired to the (now non-inplace) shared config.
        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsFusedMoEMethod,
        )

        with mock.patch(
            "sglang.srt.layers.moe.utils.get_moe_runner_backend",
            return_value=MoeRunnerBackend.MARLIN,
        ):
            layer, base_layer = self._construct(
                quant_method=mock.MagicMock(spec=CompressedTensorsFusedMoEMethod)
            )
        core = getattr(layer._lora_runner, "runner_core", None)
        self.assertIsNotNone(core)
        core_config = getattr(core, "config", None) or getattr(
            core, "runner_config", None
        )
        self.assertIsNotNone(core_config)
        self.assertIs(core_config, base_layer.moe_runner_config)
        self.assertFalse(core_config.inplace)


if __name__ == "__main__":
    unittest.main()

"""w13 buffers must be sized by the layer's gating, not assumed to be gate+up fused.

A non-gated MoE (e.g. NemotronH: relu2, checkpoint carries up_proj/down_proj and
no gate_proj) fuses a single projection into w13. Sizing w13 as 2*intermediate
leaves the upper half as uninitialised ``torch.empty`` that no weight loader ever
writes, which silently corrupts quantized MoE weights. The weight scales must
follow the same shard count, or the two describe different tensors.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 640
BLOCK_N = 128
BLOCK_K = 128


class _RecordingLayer:
    """Collects the parameters ``create_fp8_moe_weight_`` registers."""

    def __init__(self, is_gated: bool):
        self.moe_runner_config = MoeRunnerConfig(is_gated=is_gated)
        self.params = {}

    def register_parameter(self, name, param):
        self.params[name] = param


def _create_weights(is_gated: bool, block_quant: bool):
    from sglang.srt.layers.quantization import fp8 as fp8_quant

    layer = _RecordingLayer(is_gated)
    quant_config = MagicMock(
        weight_block_size=[BLOCK_N, BLOCK_K],
        activation_scheme="dynamic",
        is_checkpoint_fp8_serialized=False,
    )

    with patch.object(fp8_quant, "get_parallel") as parallel:
        parallel.return_value.tp_size = 1
        fp8_quant.Fp8MoEMethod.create_fp8_moe_weight_(
            layer=layer,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            block_quant=block_quant,
            quant_config=quant_config,
            use_mxfp8=False,
            is_checkpoint_fp8_serialized=False,
            is_fp4_expert=False,
            params_dtype=torch.bfloat16,
        )
    return layer.params


class TestFp8MoEWeightGating(CustomTestCase):
    def test_gated_fuses_gate_and_up(self):
        params = _create_weights(is_gated=True, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], 2 * INTERMEDIATE)
        self.assertEqual(
            params["w13_weight_scale_inv"].shape[1], 2 * (INTERMEDIATE // BLOCK_N)
        )

    def test_non_gated_w13_holds_up_only(self):
        # Regression: w13 was always sized 2*intermediate, so the upper half
        # stayed uninitialised for NemotronH.
        params = _create_weights(is_gated=False, block_quant=True)
        self.assertEqual(params["w13_weight"].shape[1], INTERMEDIATE)

    def test_non_gated_block_scale_matches_weight(self):
        params = _create_weights(is_gated=False, block_quant=True)
        weight_rows = params["w13_weight"].shape[1]
        scale_rows = params["w13_weight_scale_inv"].shape[1]
        self.assertEqual(scale_rows * BLOCK_N, weight_rows)

    def test_non_gated_per_tensor_scale_is_single(self):
        # One shard means one scale per expert; nothing to fuse afterwards.
        params = _create_weights(is_gated=False, block_quant=False)
        self.assertEqual(params["w13_weight_scale"].shape, (NUM_EXPERTS, 1))


if __name__ == "__main__":
    unittest.main()

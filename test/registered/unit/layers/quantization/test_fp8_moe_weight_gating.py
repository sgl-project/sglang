"""w13 must be sized by the layer's gating, not assumed to be gate+up fused.

A non-gated MoE (e.g. NemotronH: relu2, checkpoint carries up_proj/down_proj and
no gate_proj) has a single projection fused into w13. Sizing it as 2*intermediate
leaves the upper half as uninitialised ``torch.empty`` that no weight loader ever
writes, which silently corrupts quantized MoE weights.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.test.test_utils import CustomTestCase

INTERMEDIATE = 640


class TestGetMoeWeightSizes(CustomTestCase):
    """The sizing primitive itself."""

    def test_gated_fuses_gate_and_up(self):
        w13_up_dim, w2_down_dim, _ = get_moe_weight_sizes(
            INTERMEDIATE, is_concat=True, is_packed=False, is_aiter_moe=False
        )
        self.assertEqual(w13_up_dim, 2 * INTERMEDIATE)
        self.assertEqual(w2_down_dim, INTERMEDIATE)

    def test_non_gated_holds_up_only(self):
        w13_up_dim, w2_down_dim, _ = get_moe_weight_sizes(
            INTERMEDIATE, is_concat=False, is_packed=False, is_aiter_moe=False
        )
        self.assertEqual(w13_up_dim, INTERMEDIATE)
        self.assertEqual(w2_down_dim, INTERMEDIATE)


class TestFp8MoEWeightGating(CustomTestCase):
    """Fp8MoEMethod must forward the layer's gating into the sizing call."""

    def _is_concat_used_for(self, is_gated: bool) -> bool:
        from sglang.srt.layers.quantization import fp8 as fp8_quant

        layer = MagicMock()
        layer.moe_runner_config.is_gated = is_gated

        # Stop right after the sizing decision; we only assert on its argument.
        with patch.object(
            fp8_quant, "get_moe_weight_sizes", return_value=(0, 0, False)
        ) as sizes, patch.object(fp8_quant, "get_parallel") as parallel:
            parallel.return_value.tp_size = 1
            with self.assertRaises(Exception):
                fp8_quant.Fp8MoEMethod.create_fp8_moe_weight_(
                    layer=layer,
                    num_experts=8,
                    hidden_size=128,
                    intermediate_size_per_partition=INTERMEDIATE,
                    block_quant=True,
                    quant_config=MagicMock(weight_block_size=[1, 32]),
                    use_mxfp8=True,
                    is_checkpoint_fp8_serialized=False,
                    is_fp4_expert=False,
                    params_dtype=None,
                )
        self.assertTrue(sizes.called, "get_moe_weight_sizes was never reached")
        return sizes.call_args.kwargs["is_concat"]

    def test_gated_layer_requests_concat(self):
        self.assertTrue(self._is_concat_used_for(is_gated=True))

    def test_non_gated_layer_does_not_request_concat(self):
        # Regression: this was hardcoded True, over-allocating w13 for NemotronH.
        self.assertFalse(self._is_concat_used_for(is_gated=False))


if __name__ == "__main__":
    unittest.main()

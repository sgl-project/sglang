import unittest
from unittest.mock import patch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDenseMlpScatterModeUnderPrefillCP(CustomTestCase):
    """A TP-sharded dense MLP under prefill CP must gather tokens across CP ranks
    before its all-reduce, or CP pairs sum partial outputs of different tokens
    (issue #38019: Qwen3-32B emitted garbage that never reached EOS)."""

    def test_dense_mlp_gathers_across_cp(self):
        with (
            patch.object(comm, "_generic_prefill_cp_shards_tokens", return_value=True),
            patch.object(comm, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(comm, "is_mla_prefill_cp_enabled", return_value=False),
            patch.object(comm, "enable_moe_dense_fully_dp", return_value=False),
        ):
            modes = LayerScatterModes.init_new(
                layer_id=1,
                num_layers=4,
                is_layer_sparse=False,
                is_previous_layer_sparse=False,
                is_next_layer_sparse=False,
            )
        self.assertEqual(modes.mlp_mode, ScatterMode.MOE_FULL)
        self.assertEqual(modes.layer_output_mode, ScatterMode.TP_ATTN_FULL)


if __name__ == "__main__":
    unittest.main()

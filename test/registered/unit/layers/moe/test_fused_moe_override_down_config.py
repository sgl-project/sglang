"""A fused-MoE config override must be able to carry a separate down-projection config.

The Triton MoE tuner times every candidate up config under `override_config`, while
the server pairs that up config with the `_down` config file. When the override
dropped the down config, the tuner timed a pairing that serving never runs.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    override_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (  # noqa: E402
    try_get_optimal_moe_config,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

UP_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 4,
}
DOWN_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 16,
    "num_warps": 4,
    "num_stages": 3,
    "USE_TMA": True,
}


def _resolve_configs():
    return try_get_optimal_moe_config(
        (257, 512, 6144),
        (257, 6144, 256),
        9,
        "fp8_w8a8",
        64,
        block_shape=[128, 128],
        return_down_config=True,
    )


class TestOverrideDownConfig(CustomTestCase):
    def test_override_pairs_up_config_with_given_down_config(self):
        with override_config(UP_CONFIG, down_config=DOWN_CONFIG):
            up, (down, _) = _resolve_configs()
        self.assertEqual(up, UP_CONFIG)
        # Both GEMMs share one moe_align_block_size sort, so the up config's
        # BLOCK_SIZE_M wins; the caller's dict is left as given.
        self.assertEqual(down, {**DOWN_CONFIG, "BLOCK_SIZE_M": 16})
        self.assertEqual(DOWN_CONFIG["BLOCK_SIZE_M"], 64)

    def test_override_without_down_config_reuses_up_config(self):
        with override_config(UP_CONFIG):
            _, (down, _) = _resolve_configs()
        self.assertIsNone(down)


if __name__ == "__main__":
    unittest.main()

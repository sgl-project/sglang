"""Regression tests for issue #28019.

The default fused-MoE Triton configs are tuned for >=160KB-smem datacenter
GPUs; SM120/SM121 (RTX 5090 / RTX PRO 6000 / GB10) expose only ~99KB
(101376 B) per block, so the fp8_w8a8 prefill default (128x256x128, 4
stages -> 147456 B) aborted the first forward pass with
``triton OutOfResources: shared memory, Required: 147456, Hardware limit:
101376``. ``clamp_config_to_shared_mem`` degrades such configs until they
fit. Pure config logic; no GPU kernels invoked (the limit is mocked).
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    _estimate_fused_moe_smem_bytes,
    clamp_config_to_shared_mem,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

SM12X_LIMIT = 101376  # bytes/block on SM120 and SM121

# The exact config from the #28019 crash (fp8_w8a8, block_shape=None, M > E).
FP8_PREFILL_DEFAULT = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 8,
    "num_stages": 4,
}

_LIMIT_PATH = (
    "sglang.srt.layers.moe.moe_runner.triton_utils."
    "fused_moe_triton_config._get_max_shared_mem_bytes"
)


class TestFusedMoeSmemClamp(CustomTestCase):
    def test_estimate_matches_triton_reported_requirement(self):
        # triton reported exactly 147456 bytes for this config in #28019.
        self.assertEqual(
            _estimate_fused_moe_smem_bytes(FP8_PREFILL_DEFAULT, "fp8_w8a8"), 147456
        )

    def test_crash_config_clamped_under_sm12x_limit(self):
        with patch(_LIMIT_PATH, return_value=SM12X_LIMIT):
            clamped = clamp_config_to_shared_mem(FP8_PREFILL_DEFAULT, "fp8_w8a8")
        self.assertLessEqual(
            _estimate_fused_moe_smem_bytes(clamped, "fp8_w8a8"), SM12X_LIMIT
        )
        # num_stages is degraded first (cheapest perf-wise): 4 stages over
        # the limit -> 3 stages = 2 * 384 * 128 = 98304 <= 101376 fits.
        self.assertEqual(clamped["num_stages"], 3)
        self.assertEqual(clamped["BLOCK_SIZE_N"], 256)

    def test_fitting_config_untouched(self):
        small = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        with patch(_LIMIT_PATH, return_value=SM12X_LIMIT):
            self.assertIs(clamp_config_to_shared_mem(small, None), small)

    def test_unknown_limit_is_a_noop(self):
        with patch(_LIMIT_PATH, return_value=None):
            self.assertIs(
                clamp_config_to_shared_mem(FP8_PREFILL_DEFAULT, "fp8_w8a8"),
                FP8_PREFILL_DEFAULT,
            )

    def test_block_shape_only_degrades_stages(self):
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 6,
        }
        with patch(_LIMIT_PATH, return_value=49152):
            clamped = clamp_config_to_shared_mem(config, "fp8_w8a8", [128, 128])
        # Quant-block geometry must be preserved.
        self.assertEqual(clamped["BLOCK_SIZE_N"], 128)
        self.assertEqual(clamped["BLOCK_SIZE_K"], 128)
        self.assertEqual(clamped["num_stages"], 3)

    def test_tiny_limit_degrades_blocks_too(self):
        with patch(_LIMIT_PATH, return_value=24576):
            clamped = clamp_config_to_shared_mem(FP8_PREFILL_DEFAULT, "fp8_w8a8")
        self.assertLessEqual(_estimate_fused_moe_smem_bytes(clamped, "fp8_w8a8"), 24576)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# gfx950 LDS per CU.
GFX950_LDS_BYTES = 160 * 1024


class TestMatmulPersistentConfigs(CustomTestCase):
    def test_gfx950_bf16_preset(self):
        """The gfx950 bf16 preset is a measured choice; a change must be deliberate
        and still fit gfx950's LDS."""
        with patch.object(batch_invariant_ops, "_is_gfx95", True):
            cfg = batch_invariant_ops._matmul_persistent_configs()[torch.bfloat16]
        self.assertEqual(
            cfg,
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
                "matrix_instr_nonkdim": 16,
            },
        )
        tile_bytes = (
            cfg["BLOCK_SIZE_M"] * cfg["BLOCK_SIZE_K"]
            + cfg["BLOCK_SIZE_K"] * cfg["BLOCK_SIZE_N"]
        ) * torch.bfloat16.itemsize
        self.assertLessEqual(cfg["num_stages"] * tile_bytes, GFX950_LDS_BYTES)

    def test_other_gpus_keep_the_default(self):
        """The preset needs more LDS than gfx942 has, so it must stay gfx950-only."""
        with patch.object(batch_invariant_ops, "_is_gfx95", False):
            configs = batch_invariant_ops._matmul_persistent_configs()
        self.assertEqual(
            configs[torch.bfloat16],
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_stages": 3,
                "num_warps": 8,
            },
        )


if __name__ == "__main__":
    unittest.main()

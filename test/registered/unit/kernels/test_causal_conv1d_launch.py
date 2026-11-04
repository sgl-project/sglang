"""CPU tests for the causal_conv1d forward launch configuration."""

import importlib
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.ops.mamba import causal_conv1d_triton as conv
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class TestCausalConv1dLaunchConfig(CustomTestCase):
    def tearDown(self):
        importlib.reload(conv)

    def test_unset_keeps_the_default_launch(self):
        field = envs.SGLANG_CAUSAL_CONV1D_FWD_CONFIG
        for value in ("", "0"):
            with self.subTest(value=value), field.override(value):
                importlib.reload(conv)
                self.assertEqual(conv._fwd_launch_config(None, 4), conv._STOCK_FWD_LAUNCH)
        self.assertEqual(
            conv._STOCK_FWD_LAUNCH,
            {"BLOCK_M": conv.CAUSAL_CONV1D_FWD_BLOCK_M, "BLOCK_N": 256, "num_stages": 2},
        )

    def test_config_applies_only_to_the_rectangular_launch(self):
        with envs.SGLANG_CAUSAL_CONV1D_FWD_CONFIG.override("16,256,1,3"):
            importlib.reload(conv)
        tile = {"BLOCK_M": 16, "BLOCK_N": 256, "num_warps": 1, "num_stages": 3}
        for width in (2, 3, 4):
            self.assertEqual(conv._fwd_launch_config(None, width), tile)
        # A block table counts chunks of CAUSAL_CONV1D_FWD_BLOCK_M tokens, and the
        # tap chain covers widths 2-4 only: both keep the default tile.
        block_table = torch.zeros(4, 2, dtype=torch.int32)
        self.assertEqual(conv._fwd_launch_config(block_table, 4), conv._STOCK_FWD_LAUNCH)
        self.assertEqual(conv._fwd_launch_config(None, 5), conv._STOCK_FWD_LAUNCH)

    def test_out_of_range_tiles_fail_at_import(self):
        # BLOCK_M below 8 would let a chunk's prior-token reads cross into the
        # previous sequence; BLOCK_N must cover at least one element per thread.
        for value in ("4,256,2,2", "8,32,2,2", "8,384,2,2", "8,256,3,2", "8,256,2", "a,b,c,d"):
            with self.subTest(value=value), envs.SGLANG_CAUSAL_CONV1D_FWD_CONFIG.override(value):
                with self.assertRaises(ValueError):
                    importlib.reload(conv)


if __name__ == "__main__":
    unittest.main()

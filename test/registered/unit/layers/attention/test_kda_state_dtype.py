import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
    kimi_linear_state_dtype,
    mamba2_state_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestKDAStateDtype(unittest.TestCase):
    def test_state_precision_and_override(self):
        for name, dtype in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
            with self.subTest(dtype=name), patch.dict(os.environ, clear=False):
                os.environ["SGLANG_MAMBA_SSM_DTYPE"] = name
                os.environ.pop("SGLANG_MAMBA_CONV_DTYPE", None)
                expected = Mamba2StateDType(conv=dtype, temporal=dtype)
                self.assertEqual(kimi_linear_state_dtype(), expected)
                params = KimiLinearCacheParams(shape=None, layers=[])
                self.assertEqual(params.dtype, expected)
                # Other Mamba model families keep their convolution default.
                self.assertEqual(mamba2_state_dtype().conv, torch.bfloat16)
                os.environ["SGLANG_MAMBA_CONV_DTYPE"] = "bfloat16"
                self.assertEqual(
                    kimi_linear_state_dtype(),
                    Mamba2StateDType(conv=torch.bfloat16, temporal=dtype),
                )

    def test_replay_budget_follows_ssm_dtype(self):
        shape = KimiLinearStateShape.create(tp_world_size=1, num_heads=2, head_dim=128)
        for ssm in (torch.float32, torch.bfloat16):
            budgets = []
            for conv in (torch.float32, torch.bfloat16):
                params = KimiLinearCacheParams(
                    shape=shape,
                    layers=[0],
                    dtype=Mamba2StateDType(conv=conv, temporal=ssm),
                )
                budgets.append(params.replayssm_ring_bytes_per_req(16))
            expected = 4 * 2 * 16 * 128 * ssm.itemsize + (2 * 16 * 128 + 2 * 16) * 4
            self.assertEqual(budgets, [expected, expected])

    def test_explicit_dtype_is_preserved(self):
        dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
        params = KimiLinearCacheParams(shape=None, layers=[], dtype=dtype)
        self.assertEqual(params.dtype, dtype)


if __name__ == "__main__":
    unittest.main()

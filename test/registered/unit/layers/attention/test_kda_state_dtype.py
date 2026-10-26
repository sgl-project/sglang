import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
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

    def test_explicit_dtype_is_preserved(self):
        dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
        params = KimiLinearCacheParams(shape=None, layers=[], dtype=dtype)
        self.assertEqual(params.dtype, dtype)


if __name__ == "__main__":
    unittest.main()

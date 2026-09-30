"""Kimi-K3: Cake KDA prefill export vs the Triton prefill, served at TP8.

K3's KDA layers use the bounded gate (``gate_lower_bound=-5``), so both the
BF16 and the TF32 Cake exports must serve every prefill and reproduce the
Triton reference. One B300 node (8 GPUs) holds the 1.5 TB checkpoint.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kda_cake_prefill_parity_kit import (
    BF16_ARM,
    TF32_ARM,
    KDACakePrefillParityMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=7200, stage="nightly", runner_config="8-gpu-b300")

KIMI_K3_MODEL = os.environ.get("SGLANG_TEST_KIMI_K3_MODEL", "moonshotai/Kimi-K3")


class TestKimiK3CakePrefillParity(KDACakePrefillParityMixin, CustomTestCase):
    model = KIMI_K3_MODEL
    tp_size = 8
    arms = (BF16_ARM, TF32_ARM)
    base_args = KDACakePrefillParityMixin.base_args + (
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    )


if __name__ == "__main__":
    unittest.main()

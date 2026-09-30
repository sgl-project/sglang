"""Kimi-K3: Cake KDA prefill and decode vs the Triton kernels, served at TP8.

K3's KDA layers use the bounded gate (``gate_lower_bound=-5``), so both the
BF16 and the TF32 Cake prefill exports must serve every prefill and reproduce
the Triton reference; the Cake decode is checked behind the Triton prefill
and in the production configuration (Cake prefill + Cake decode). One B300
node (8 GPUs) holds the 1.5 TB checkpoint; two 4-GPU GB300 nodes work through
the multi-node knobs documented in the kit.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kda_cake_parity_kit import (
    BF16_ARM,
    BF16_DECODE_ARM,
    DECODE_ARM,
    TF32_ARM,
    KDACakeParityMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=7200, stage="nightly", runner_config="8-gpu-b300")

KIMI_K3_MODEL = os.environ.get("SGLANG_TEST_KIMI_K3_MODEL", "moonshotai/Kimi-K3")


class TestKimiK3CakeKDAParity(KDACakeParityMixin, CustomTestCase):
    model = KIMI_K3_MODEL
    tp_size = 8
    arms = (BF16_ARM, TF32_ARM, DECODE_ARM, BF16_DECODE_ARM)
    base_args = KDACakeParityMixin.base_args + (
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    )


if __name__ == "__main__":
    unittest.main()

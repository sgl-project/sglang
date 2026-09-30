"""Kimi-Linear-48B-A3B: Cake KDA prefill and decode vs the Triton kernels, served.

TP2 on Blackwell (the Cake export ships sm_100a/sm_103a modules). The BF16
prefill arm must reproduce the Triton reference; the TF32 prefill arm is
expected to fall back to Triton because Kimi-Linear uses the unbounded
softplus gate, which the TF32 export does not serve (see
``--kda-cake-prefill-precision``). The Cake decode is checked behind the
Triton prefill and in the production configuration (Cake prefill + Cake
decode).
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kda_cake_parity_kit import (
    BF16_ARM,
    BF16_DECODE_ARM,
    DECODE_ARM,
    TF32_ARM,
    CakeArm,
    KDACakeParityMixin,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=3600, stage="nightly", runner_config="4-gpu-b200")

KIMI_LINEAR_MODEL = os.environ.get(
    "SGLANG_TEST_KIMI_LINEAR_MODEL", "moonshotai/Kimi-Linear-48B-A3B-Instruct"
)


class TestKimiLinearCakeKDAParity(KDACakeParityMixin, CustomTestCase):
    model = KIMI_LINEAR_MODEL
    tp_size = 2
    arms = (
        BF16_ARM,
        CakeArm(
            name=TF32_ARM.name,
            extra_args=TF32_ARM.extra_args,
            env=dict(TF32_ARM.env),
            expect_cake_fallback=True,
        ),
        DECODE_ARM,
        BF16_DECODE_ARM,
    )


if __name__ == "__main__":
    unittest.main()

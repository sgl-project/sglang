"""page_size > 1 variants at topk=1 (flashinfer).

EAGLE3 page64 (spec v2) + EAGLE/Llama-2 page4 (spec v1). topk>1 page variants
live in test_spec_eagle_topk.py. Runs on the cheap (5090) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=385, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Page64(Eagle3Base, SpecAccuracyKit, SpecLogprobKit, SpecFeatureKit):
    """EAGLE3 spec v2, page_size=64 (flashinfer): + logprob losslessness."""

    page_size = 64
    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Page4Topk1(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Llama-2 topk=1 + page_size=4."""

    spec_topk = 1
    spec_tokens = 6
    page_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()

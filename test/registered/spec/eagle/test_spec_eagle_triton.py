"""triton attention backend (EAGLE3 spec v2 + EAGLE/Llama-2 spec v1).

triton runs everywhere, so this stays on the cheap (5090) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=455, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Triton(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE3 spec v2 on triton (kits listed in bases)."""

    attention_backend = "triton"
    max_running_requests = 64
    cuda_graph_max_bs = 64
    gsm8k_num_examples = 1000
    gsm8k_check_accept_len = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Triton(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """EAGLE/Llama-2 topk=8 on triton (spec v1)."""

    attention_backend = "triton"
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()

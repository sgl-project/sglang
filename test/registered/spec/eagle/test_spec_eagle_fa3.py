"""fa3 attention backend -- Hopper-only (FlashAttention-3 is sm_90).

fa3 is the real H200 default for MHA spec at topk=1, so this also covers the
"what an H200 user actually runs" path. Requires the large (Hopper) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecPerfKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=302, stage="base-b", runner_config="1-gpu-large")


class TestEagle3Fa3(Eagle3Base, SpecCorrectnessKit, SpecAccuracyKit, SpecLogprobKit):
    """EAGLE3 spec v2 topk=1 on fa3 (the H200 default backend)."""

    attention_backend = "fa3"
    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Fa3Page256(
    EagleLlama2Base,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecPerfKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=5 tree on fa3 + page_size=256 (spec v1)."""

    spec_topk = 5
    spec_steps = 8
    attention_backend = "fa3"
    page_size = 256
    chunked_prefill_size = 4096  # must be divisible by page_size (256)
    cuda_graph_max_bs = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()

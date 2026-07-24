"""topk > 1 tree drafting at page_size > 1 (EAGLE3 topk8 + EAGLE/Llama-2 topk8).

page64 stays on spec v2 (overlap), page4 runs on spec v1 (no overlap). flashinfer is
pinned because this runs on the cheap (5090) runner, where fa3 (Hopper-only) isn't
available -- functional sanity only, no perf/stress. (page>1 topk>1 on fa3 is covered
on the Hopper runner in test_spec_eagle_fa3.py.)
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=720, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Page64Topk8(Eagle3Base, SpecAccuracyKit, SpecFeatureKit):
    """EAGLE3 topk=8 tree + page_size=64 (spec v2)."""

    page_size = 64
    spec_topk = 8
    spec_tokens = 32
    disable_overlap = False
    cuda_graph_max_bs_decode = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Page4Topk8(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Llama-2 topk>1 tree + page_size=4 (spec v1)."""

    page_size = 4
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()

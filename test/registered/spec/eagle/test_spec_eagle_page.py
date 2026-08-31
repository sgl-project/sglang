"""EAGLE3 chain drafting (topk=1) at page_size > 1, flashinfer.

topk=1 takes its own fast path in the draft worker, so this cell is not
covered by the tree variants in test_spec_eagle_topk_page.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=212, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Page64(
    Eagle3Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecFeatureKit,
):
    """Overlap scheduler, page_size=64: + logprob losslessness."""

    page_size = 64
    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()

"""EAGLE3 spec-decoding core: overlap x no-overlap matrix at the standard
config (topk=1, page_size=1); only ``disable_overlap`` differs. Both run the
same EAGLEWorkerV2 -- the scheduler just drives it synchronously when overlap
is off.
flashinfer is pinned (the 5090 default) so a default-selection change can't
silently alter what this exercises.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=403, stage="base-b", runner_config="1-gpu-small")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(Eagle3Base):
    env_overrides = (
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
        (envs.SGLANG_ENABLE_GRAPH_POOL_BORROW, 1),
    )


class TestEagle3Overlap(_Core, *_KITS):
    """Overlap scheduler on."""

    disable_overlap = False


class TestEagle3NoOverlap(_Core, *_KITS):
    """Overlap scheduler off (synchronous)."""

    disable_overlap = True


if __name__ == "__main__":
    unittest.main()

"""EAGLE3 spec-decoding core: overlap (spec v2) x no-overlap (spec v1) matrix.

Both cases run the full sanity kit set on the same standard config (EAGLE3,
topk=1, page_size=1, flashinfer); the only difference is ``disable_overlap``.
flashinfer is pinned (the 5090 default) so a future default change can't silently
alter what this exercises. Other backends / page sizes / topk>1 / heavy checks
live in the sibling test_spec_eagle_*.py files.
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

register_cuda_ci(est_time=480, stage="base-b", runner_config="1-gpu-small")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(Eagle3Base):
    # Busy-time pool accounting check (topk=1 only).
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3Overlap(_Core, *_KITS):
    """Spec v2 (overlap scheduler on)."""

    disable_overlap = False


class TestEagle3NoOverlap(_Core, *_KITS):
    """Spec v1 (overlap scheduler off)."""

    disable_overlap = True


if __name__ == "__main__":
    unittest.main()

"""EAGLE3 spec-decoding core test.

Validates Overlap (Spec v2) and No-overlap (Spec v1) independently against
the same correctness baseline: output, logprobs, penalties, and stop behavior.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(Eagle3Base):
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)
    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128


class TestEagle3Overlap(_Core, *_KITS):
    """Testcase: EAGLE3 spec-decoding core test.
    Validates correctness with overlap scheduling enabled (Spec v2).
    Covers output, logprobs, penalties, and stop behavior.

    [Test Category] Functionality
    [Test Target] EAGLE3 spec-decoding with overlap scheduling (Spec v2)
    """

    disable_overlap = False


class TestEagle3NoOverlap(_Core, *_KITS):
    """Testcase: EAGLE3 spec-decoding core test.
    Validates correctness with overlap scheduling disabled (Spec v1).
    Covers output, logprobs, penalties, and stop behavior.

    [Test Category] Functionality
    [Test Target] EAGLE3 spec-decoding with overlap disabled (Spec v1)
    """

    disable_overlap = True


if __name__ == "__main__":
    unittest.main()

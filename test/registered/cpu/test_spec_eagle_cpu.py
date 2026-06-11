"""EAGLE spec-decoding core on CPU: the standard config (topk=1, page_size=1)
plus a topk=4 tree-drafting suite, both on the synchronous (non-overlap) path.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import EagleLlama2Base

register_cpu_ci(est_time=900, suite="base-b-test-cpu")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(EagleLlama2Base):
    """EAGLE (Llama-2) preset on CPU.
    """

    attention_backend = "intel_amx"
    disable_overlap = True
    mem_fraction_static = 0.3
    gsm8k_num_examples = 64
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2NoOverlap(_Core, *_KITS):
    """Spec v1 (overlap scheduler off) -- the only mode reachable on CPU."""

    # Standard chain config (topk=1, page_size=1), same shape as the CUDA core.
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    # EAGLE/Llama-2 topk=1 accepts modestly; tune against CI if needed.
    acc_length_thres = 1.6
    batch_accept_len_thres = 1.3
    gsm8k_accept_len_thres = 1.3


class TestEagleLlama2Topk4(
    _Core,
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE/Llama-2 topk=4 tree coverage (kits listed in bases)."""

    spec_steps = 3
    spec_topk = 4
    spec_tokens = 8
    acc_length_thres = 2.4
    batch_accept_len_thres = 1.6
    gsm8k_accept_len_thres = 2.0


if __name__ == "__main__":
    unittest.main()

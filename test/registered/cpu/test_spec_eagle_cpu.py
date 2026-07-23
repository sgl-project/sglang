"""EAGLE spec-decoding core on CPU: the standard config (topk=1, page_size=1)
on the synchronous (non-overlap) path. topk > 1 tree drafting is covered in
test_spec_eagle_topk_cpu.py (split to stay under the per-file CI timeout).
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

# Measured 780s all-green on a 40-core GNR socket (1 launch + 18 methods).
register_cpu_ci(est_time=800, suite="base-b-test-cpu")

_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class _Core(EagleLlama2Base):
    """EAGLE (Llama-2) preset on CPU."""

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

    @unittest.skip(
        "constrained decoding on CPU needs a vocab-mask CPU branch in the "
        "xgrammar backend (upstream gap, not spec-specific); the other grammar "
        "backends lack the rollback spec verification requires"
    )
    def test_constrained_decoding(self):
        pass


if __name__ == "__main__":
    unittest.main()

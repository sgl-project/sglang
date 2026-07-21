"""EAGLE topk > 1 tree drafting on CPU (Llama-2 topk=4, synchronous path).

Split from test_spec_eagle_cpu.py, mirroring the CUDA test_spec_eagle.py /
test_spec_eagle_topk.py layout, so each file stays under the per-file CI
timeout.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import EagleLlama2Base

# Measured 830s all-green on a 40-core GNR socket (1 launch + 14 methods).
register_cpu_ci(
    est_time=850,
    suite="base-b-test-cpu",
    disabled="EagleLlama2Base needs gated meta-llama/Llama-2-7b-chat-hf",
)


class _Core(EagleLlama2Base):
    """EAGLE (Llama-2) preset on CPU."""

    attention_backend = "intel_amx"
    disable_overlap = True
    mem_fraction_static = 0.3
    gsm8k_num_examples = 64
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


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

    @unittest.skip(
        "constrained decoding on CPU needs a vocab-mask CPU branch in the "
        "xgrammar backend (upstream gap, not spec-specific); the other grammar "
        "backends lack the rollback spec verification requires"
    )
    def test_constrained_decoding(self):
        pass


if __name__ == "__main__":
    unittest.main()

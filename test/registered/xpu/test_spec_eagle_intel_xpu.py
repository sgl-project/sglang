"""intel_xpu attention backend (EAGLE3 topk=1 chain + EAGLE/Llama-2 spec)."""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecHiddenStatesKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_xpu_ci(est_time=1800, suite="nightly-xpu-1-gpu", nightly=True)


class TestEagle3IntelXPU(
    Eagle3Base,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """EAGLE3 spec v2 on the intel_xpu attention backend (kits listed in bases)."""

    attention_backend = "intel_xpu"
    max_running_requests = 24
    gsm8k_num_examples = 300
    gsm8k_check_accept_len = True
    mem_fraction_static = 0.95
    extra_args = ("--max-total-tokens", "16384", "--disable-decode-cuda-graph")
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2IntelXPU(
    EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit, SpecHiddenStatesKit
):
    """EAGLE/Llama-2 on intel_xpu using the supported topk > 1 paged config."""

    attention_backend = "intel_xpu"
    page_size = 64
    gsm8k_check_accept_len = True
    gsm8k_num_examples = 300
    enable_return_hidden_states = True
    mem_fraction_static = 0.95
    max_running_requests = 6
    chunked_prefill_size = 512
    extra_args = ("--max-total-tokens", "16384", "--disable-decode-cuda-graph")


if __name__ == "__main__":
    unittest.main()

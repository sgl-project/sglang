"""intel_xpu attention backend (EAGLE3 topk=1 chain, EAGLE3 topk=2 tree on
paged KV with XPU graphs, EAGLE/Llama-2 spec)."""

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

register_xpu_ci(est_time=2400, suite="nightly-kernel-main-test-1-gpu-xpu", nightly=True)


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


class TestEagle3TreePagedIntelXPU(
    Eagle3Base, SpecAccuracyKit, SpecLogprobKit, SpecFeatureKit
):
    """EAGLE3 topk=2 tree at page_size 64 with XPU decode graphs: the draft-decode
    and target-verify cascades read branch KV by slot id from paged KV, captured."""

    attention_backend = "intel_xpu"
    page_size = 64
    spec_steps = 3
    spec_topk = 2
    spec_tokens = 6
    max_running_requests = 8
    gsm8k_num_examples = 300
    gsm8k_check_accept_len = True
    # Measured 1.85 on B60 (5-shot completion prompts; the preset's 1.3 is topk=1).
    gsm8k_accept_len_thres = 1.6
    # Validated with graphs at 0.8; the other classes' 0.95 is untested here.
    mem_fraction_static = 0.8
    extra_args = ("--max-total-tokens", "16384", "--cuda-graph-backend-decode", "full")
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2IntelXPU(
    EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit, SpecHiddenStatesKit
):
    """EAGLE/Llama-2 on intel_xpu using the supported topk = 1 paged config."""

    attention_backend = "intel_xpu"
    spec_topk = 1
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

"""EAGLE3 spec-v2 config variations not covered by the main matrix.

The EAGLE3 overlap/no-overlap matrix (test_eagle_infer_a.py) runs the flashinfer
backend at page_size=1. This file covers the triton backend and page_size>1,
which are sensitive to backend/paging, with the logprob-match + matched-stop kits.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import SpecEagleServerBase

register_cuda_ci(est_time=369, stage="base-b", runner_config="1-gpu-small")


class _Eagle3Triton(SpecEagleServerBase):
    """EAGLE3 spec v2 on the triton backend."""

    spec_algo = "EAGLE3"
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    attention_backend = "triton"
    page_size = 1
    max_running_requests = 64
    cuda_graph_max_bs = 64
    chunked_prefill_size = 1024
    mem_fraction_static = 0.75
    disable_overlap = False  # spec v2
    gsm8k_num_examples = 1000
    gsm8k_score_thres = 0.7
    gsm8k_check_accept_len = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagle3Triton(
    _Eagle3Triton,
    MatchedStopMixin,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
):
    """gsm8k + spec-v2 logprob match + matched-stop + radix on triton."""


class TestEagle3Page64(TestEagle3Triton):
    page_size = 64


if __name__ == "__main__":
    unittest.main()

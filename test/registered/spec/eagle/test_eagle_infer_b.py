"""EAGLE (Llama-2) feature coverage + config-variation smokes.

The EAGLE3 overlap/no-overlap matrix lives in test_eagle_infer_a.py. This file
keeps the second model (EAGLE on Llama-2) and the config variations that need a
specific server setup (page size, attention backend, retract, timeouts,
token-map). All run spec v1 (EAGLE topk>1 routes to v1).
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.abort_timeout_kit import (
    AbortAllMixin,
    RunningTimeoutTwoWaveMixin,
    WaitingTimeoutMixin,
)
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import SpecEagleServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
)

register_cuda_ci(est_time=847, stage="base-b", runner_config="1-gpu-large")


class _EagleLlama2(SpecEagleServerBase):
    """EAGLE on Llama-2 (spec v1). gsm8k accuracy is low for this model."""

    model = DEFAULT_TARGET_MODEL_EAGLE
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE
    spec_algo = "EAGLE"
    spec_steps = 5
    spec_topk = 8
    spec_tokens = 64
    attention_backend = "flashinfer"
    chunked_prefill_size = 128
    max_running_requests = 8
    mem_fraction_static = 0.7
    gsm8k_score_thres = 0.20
    # EAGLE topk>1 already routes to v1; force it explicitly to preserve intent.
    env_overrides = ((envs.SGLANG_ENABLE_SPEC_V2, False),)


class TestEagleLlama2Suite(
    _EagleLlama2, SpecAccuracyKit, SpecLogprobKit, SpecPenaltyKit, SpecFeatureKit
):
    """Full feature coverage on EAGLE/Llama-2: gsm8k, logprobs, penalty, radix,
    abort, constrained decoding."""


class TestEagleLlama2Fa3Page256(_EagleLlama2, SpecAccuracyKit):
    spec_topk = 5
    spec_steps = 8
    attention_backend = "fa3"
    page_size = 256
    cuda_graph_max_bs = 5


class TestEagleLlama2PageTopk1(_EagleLlama2, SpecAccuracyKit):
    # topk=1 + page>1; busy-time pool accounting check (topk=1 only).
    spec_topk = 1
    spec_tokens = 6
    page_size = 4
    env_overrides = (
        (envs.SGLANG_ENABLE_SPEC_V2, False),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama2PageTopk(_EagleLlama2, SpecAccuracyKit):
    # topk>1 + page>1 (v1 tree path).
    page_size = 4


class TestEagleLlama2Triton(_EagleLlama2, SpecAccuracyKit, SpecFeatureKit):
    attention_backend = "triton"


class TestEagleLlama2Retract(_EagleLlama2, SpecAccuracyKit, SpecFeatureKit):
    max_running_requests = 64
    extra_args = ("--max-total-tokens", 4500)  # small KV to trigger retract
    env_overrides = (
        (envs.SGLANG_ENABLE_SPEC_V2, False),
        (envs.SGLANG_TEST_RETRACT, True),
    )


class TestEagleLlama2TokenMap(_EagleLlama2, SpecAccuracyKit):
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model = "lmsys/sglang-EAGLE-LLaMA3-Instruct-8B"
    spec_topk = 4
    spec_tokens = 8
    cuda_graph_max_bs = 5
    extra_args = (
        "--speculative-token-map",
        "thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt",
    )


class TestEagleLlama2AbortAll(_EagleLlama2, AbortAllMixin):
    abort_all_max_new_tokens = 4000


class TestEagleLlama2WaitingTimeout(_EagleLlama2, WaitingTimeoutMixin):
    max_running_requests = 1
    env_overrides = (
        (envs.SGLANG_ENABLE_SPEC_V2, False),
        (envs.SGLANG_REQ_WAITING_TIMEOUT, 0.001),
    )


class TestEagleLlama2RunningTimeout(_EagleLlama2, RunningTimeoutTwoWaveMixin):
    # Regression: https://github.com/sgl-project/sglang/pull/18760
    max_running_requests = 16
    env_overrides = (
        (envs.SGLANG_ENABLE_SPEC_V2, False),
        (envs.SGLANG_REQ_RUNNING_TIMEOUT, 3),
    )


if __name__ == "__main__":
    unittest.main()

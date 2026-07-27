"""Perf + stress: throughput, retract-under-pressure, abort storms, timeouts.

These need memory headroom / measure load behavior, so they run on the large
(Hopper) runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ascend.test_ascend_utils import (
    EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH,
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.abort_timeout_kit import (
    AbortAllMixin,
    RunningTimeoutTwoWaveMixin,
    WaitingTimeoutMixin,
)
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestEagleLlama3Retract(Eagle3Base, SpecAccuracyKit, SpecFeatureKit):
    """Testcase: EAGLE3 retract under tight KV cache budget.
    Validates memory safety when requests are retracted due to limited KV space.
    Ensures no memory leaks or corruption under high memory pressure.

    [Test Category] Stability
    [Test Target] EAGLE3 KV cache retract with strict memory checking
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 64
    extra_args = ("--max-total-tokens", 4500)  # small KV to trigger retract
    env_overrides = (
        (envs.SGLANG_TEST_RETRACT, True),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama3AbortAll(Eagle3Base, AbortAllMixin):
    """Testcase: EAGLE3 abort-all storm under heavy speculation load.
    Stresses request cancellation paths while speculative decoding is active.
    Verifies clean shutdown and resource cleanup during abort storms.

    [Test Category] Stability
    [Test Target] EAGLE3 abort-all handling with strict memory checking
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    abort_all_max_new_tokens = 4000
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama3WaitingTimeout(Eagle3Base, WaitingTimeoutMixin):
    """Testcase: EAGLE3 waiting queue timeout with minimal concurrency.
    Validates request timeout behavior while waiting in the scheduling queue.
    Ensures timed-out requests are safely discarded without affecting others.

    [Test Category] Robustness
    [Test Target] EAGLE3 waiting-timeout handling under single-request load
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 1
    env_overrides = (
        (envs.SGLANG_REQ_WAITING_TIMEOUT, 0.001),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama3RunningTimeout(Eagle3Base, RunningTimeoutTwoWaveMixin):
    """Testcase: EAGLE3 running-timeout regression (two-wave pattern).
    Regression test for https://github.com/sgl-project/sglang/pull/18760.
    Validates correct timeout handling for long-running speculative requests
    under concurrent load with strict memory checks enabled.

    [Test Category] Regression
    [Test Target] EAGLE3 running-timeout handling (multi-wave execution)
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    draft_model = EAGLE3_LLAMA3_1_INSTRUCT_8B_WEIGHTS_PATH
    attention_backend = "ascend"
    page_size = 128
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    max_running_requests = 16
    env_overrides = (
        (envs.SGLANG_REQ_RUNNING_TIMEOUT, 3),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


if __name__ == "__main__":
    unittest.main()

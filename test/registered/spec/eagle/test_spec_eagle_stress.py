"""Perf + stress: throughput, retract-under-pressure, abort storms, timeouts.

These need memory headroom / measure load behavior, so they run on the large
(Hopper) runner.
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
    SpecPerfKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base, EagleLlama2Base

register_cuda_ci(est_time=807, stage="base-b", runner_config="1-gpu-large")


class TestEagle3Perf(Eagle3Base, SpecPerfKit):
    """Decode throughput (max_new_tokens=1) on EAGLE3 spec v2."""

    disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2Retract(EagleLlama2Base, SpecAccuracyKit, SpecFeatureKit):
    """Retract under a small KV budget; must not leak."""

    max_running_requests = 64
    extra_args = ("--max-total-tokens", 4500)  # small KV to trigger retract
    env_overrides = (
        (envs.SGLANG_TEST_RETRACT, True),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagle3Topk16V2Retract(Eagle3Base, SpecAccuracyKit, SpecFeatureKit):
    """EAGLE3 topk=16 tree on spec v2 under retract; must not leak KV. Stresses
    the accepted-path KV move (move_accept_tokens_to_target_kvcache)."""

    spec_topk = 16
    spec_tokens = 64
    disable_overlap = False
    cuda_graph_max_bs_decode = 5
    max_running_requests = 64
    gsm8k_accept_len_thres = 2.4
    extra_args = ("--max-total-tokens", 4500)  # small KV to trigger retract
    env_overrides = (
        (envs.SGLANG_TEST_RETRACT, True),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama2AbortAll(EagleLlama2Base, AbortAllMixin):
    abort_all_max_new_tokens = 4000
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


class TestEagleLlama2WaitingTimeout(EagleLlama2Base, WaitingTimeoutMixin):
    max_running_requests = 1
    env_overrides = (
        (envs.SGLANG_REQ_WAITING_TIMEOUT, 0.001),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


class TestEagleLlama2RunningTimeout(EagleLlama2Base, RunningTimeoutTwoWaveMixin):
    # Regression: https://github.com/sgl-project/sglang/pull/18760
    max_running_requests = 16
    env_overrides = (
        (envs.SGLANG_REQ_RUNNING_TIMEOUT, 3),
        (envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),
    )


if __name__ == "__main__":
    unittest.main()

"""MockEngine thin-shell API tests.

The MockEngine / MockReqHandle / MockOracle classes that this file targeted were removed when
the mock-mode subsystem was redesigned. The new kv_cache_canary.mock_model surface is
class-free: oracle.py exposes only Oracle implementations, sampler.py exposes free functions
(install_oracle_sampler, fill_expected_inputs), and args_modifier.py exposes
apply_mock_model_defaults. There is no engine wrapper that owns admit/step/preempt/abort —
those flows live in sglang's own scheduler. Until a new test harness emerges (if ever), these
end-to-end shell tests are kept as no-ops.
"""

from __future__ import annotations

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="MockEngine / MockReqHandle classes were removed; the new kv_cache_canary.mock_model is class-free (only free functions + Oracle dataclasses)."
)


def test_launch_qwen3_dummy_weights_one_layer() -> None:
    pass


def test_admit_returns_req_handle() -> None:
    pass


def test_step_drives_one_event_loop() -> None:
    pass


def test_step_until_target_steps() -> None:
    pass


def test_force_preempt_pauses_req() -> None:
    pass


def test_resume_returns_req_to_active() -> None:
    pass


def test_abort_removes_req() -> None:
    pass


def test_canary_violations_returns_list() -> None:
    pass


def test_assert_no_canary_violations_raises_on_violation() -> None:
    pass


def test_engine_idle_60s_no_violation() -> None:
    pass


def test_forward_uses_triton_hash_kernel() -> None:
    pass


def test_oracle_expectation_per_req_per_step_can_be_pinned() -> None:
    pass

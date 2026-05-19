"""MockEngine thin-shell API self-unit tests."""

from __future__ import annotations

import time

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
    from sglang.srt.mock_mode.handle import MockReqHandle
    from sglang.srt.mock_mode.oracle import MockOracle
except ImportError:
    MockEngine = None
    MockReqHandle = None
    MockOracle = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_launch_qwen3_dummy_weights_one_layer() -> None:
    start = time.monotonic()
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    elapsed = time.monotonic() - start

    assert elapsed < 20.0
    assert engine is not None
    engine.shutdown()


def test_admit_returns_req_handle() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)

    req = engine.admit(prompt=_fake_prompt(64), max_new_tokens=4)

    assert isinstance(req, MockReqHandle)
    assert len(req.prompt) == 64
    assert req.max_new_tokens == 4
    engine.shutdown()


def test_step_drives_one_event_loop() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    engine.admit(prompt=_fake_prompt(16), max_new_tokens=2)

    result = engine.step()

    assert result.forward_mode in {"prefill", "decode", "mixed", "idle"}
    engine.shutdown()


def test_step_until_target_steps() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=4)

    results = engine.step_until(req, n=2)

    assert len(results) >= 2
    engine.shutdown()


def test_force_preempt_pauses_req() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=4)
    engine.step()

    engine.force_preempt(req)
    active_ids = [r.req_id for r in engine.active_reqs()]

    assert req.req_id not in active_ids
    engine.shutdown()


def test_resume_returns_req_to_active() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=4)
    engine.step()
    engine.force_preempt(req)

    engine.resume(req)
    active_ids = [r.req_id for r in engine.active_reqs()]

    assert req.req_id in active_ids
    engine.shutdown()


def test_abort_removes_req() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=4)
    engine.step()

    engine.abort(req)
    active_ids = [r.req_id for r in engine.active_reqs()]

    assert req.req_id not in active_ids
    engine.shutdown()


def test_canary_violations_returns_list() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    engine.admit(prompt=_fake_prompt(8), max_new_tokens=2)
    engine.step()

    violations = engine.canary_violations()

    assert isinstance(violations, list)
    engine.shutdown()


def test_assert_no_canary_violations_raises_on_violation() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=2)
    engine.pin_expectation(req=req, step=0, expected_input_token=999999)
    engine.step()

    with pytest.raises(AssertionError):
        engine.assert_no_canary_violations()
    engine.shutdown()


def test_engine_idle_60s_no_violation() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_forward_uses_triton_hash_kernel() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    engine.admit(prompt=_fake_prompt(8), max_new_tokens=2)

    engine.step()
    counts = engine.kernel_invocation_counts()

    assert counts.get("triton_hash_kernel", 0) > 0
    engine.shutdown()


def test_oracle_expectation_per_req_per_step_can_be_pinned() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=_fake_prompt(8), max_new_tokens=4)
    bogus_token = 999999
    engine.pin_expectation(req=req, step=0, expected_input_token=bogus_token)

    engine.step()
    violations = engine.canary_violations()

    assert any(v.req_id == req.req_id and v.expected == bogus_token for v in violations)
    engine.shutdown()

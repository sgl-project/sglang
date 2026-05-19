"""Canary kernel and mock-oracle wiring self-unit tests."""

from __future__ import annotations

import os

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
    from sglang.srt.mock_mode.canary_wiring import (
        FAIL_REASON_BIT_INPUT_POSITION_MISMATCH,
        FAIL_REASON_BIT_INPUT_TOKEN_MISMATCH,
    )
except ImportError:
    MockEngine = None
    FAIL_REASON_BIT_INPUT_TOKEN_MISMATCH = None
    FAIL_REASON_BIT_INPUT_POSITION_MISMATCH = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def _launch(**kwargs) -> "MockEngine":
    return MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1, **kwargs)


def test_mock_expected_tokens_filled_from_oracle() -> None:
    engine = _launch()
    engine.admit(prompt=[1, 2, 3, 4], max_new_tokens=1)
    engine.step()

    plan = engine.last_write_plan()
    expected_tokens = engine.last_mock_expected_tokens()

    assert expected_tokens.shape[0] == plan.num_write_entries
    assert expected_tokens.tolist() == [1, 2, 3, 4]
    engine.shutdown()


def test_mock_expected_positions_filled_from_oracle() -> None:
    engine = _launch()
    engine.admit(prompt=[10, 20, 30], max_new_tokens=1)
    engine.step()

    plan = engine.last_write_plan()
    expected_positions = engine.last_mock_expected_positions()

    assert expected_positions.shape[0] == plan.num_write_entries
    assert expected_positions.tolist() == [0, 1, 2]
    engine.shutdown()


def test_mock_mode_on_enables_write_step_comparison() -> None:
    engine = _launch(canary_mock_mode_on=True)
    req = engine.admit(prompt=[1, 2, 3], max_new_tokens=1)
    engine.pin_expectation(req=req, step=0, expected_input_token=999999)

    engine.step()
    violations = engine.canary_violations()

    assert any(
        v.fail_reason_bits & FAIL_REASON_BIT_INPUT_TOKEN_MISMATCH for v in violations
    )
    engine.shutdown()


def test_mock_mode_off_dummy_tensors_safe() -> None:
    engine = _launch(canary_mock_mode_on=False)
    engine.admit(prompt=[1, 2, 3], max_new_tokens=1)

    engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_mock_input_perturb_env_detects_violation() -> None:
    os.environ["SGLANG_MOCK_INPUT_PERTURB_PROB"] = "0.01"
    try:
        engine = _launch(canary_mock_mode_on=True)
        engine.admit(prompt=list(range(1, 65)), max_new_tokens=4)

        deadline_step = 0
        while not engine.canary_violations() and deadline_step < 240:
            engine.step()
            deadline_step += 1
        violations = engine.canary_violations()

        assert any(
            v.fail_reason_bits
            & (
                FAIL_REASON_BIT_INPUT_TOKEN_MISMATCH
                | FAIL_REASON_BIT_INPUT_POSITION_MISMATCH
            )
            for v in violations
        )
        engine.shutdown()
    finally:
        os.environ.pop("SGLANG_MOCK_INPUT_PERTURB_PROB", None)


def test_violation_row_carries_req_id_and_position() -> None:
    engine = _launch(canary_mock_mode_on=True)
    req = engine.admit(prompt=[1, 2, 3], max_new_tokens=1)
    bogus = 999999
    engine.pin_expectation(req=req, step=0, expected_input_token=bogus, position=1)

    engine.step()
    violations = engine.canary_violations()

    assert any(
        v.req_id == req.req_id and v.position == 1 and v.expected == bogus
        for v in violations
    )
    engine.shutdown()


def test_violation_does_not_cascade_to_next_verify() -> None:
    engine = _launch(canary_mock_mode_on=True)
    req = engine.admit(prompt=[1, 2, 3], max_new_tokens=2)
    engine.pin_expectation(req=req, step=0, expected_input_token=999999)

    engine.step()
    first_count = len(engine.canary_violations())
    engine.step()
    second_count = len(engine.canary_violations())

    assert second_count == first_count
    engine.shutdown()

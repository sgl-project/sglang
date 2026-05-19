"""Scripted-PR regression self-e2e for #20711 cross-batch bitwise invariance."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_cross_batch_bitwise_regression() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_20711_fix=False,
    )
    solo = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)
    engine.step_until(solo, n=4)
    solo_tokens = engine.output_history(solo)

    batched = [
        engine.admit(prompt=_fake_prompt(32), max_new_tokens=4) for _ in range(4)
    ]
    engine.step_until_idle(max_steps=20)
    target_in_batch = batched[0]

    assert engine.output_history(target_in_batch) != solo_tokens
    engine.shutdown()


def test_cross_batch_bitwise_with_fix() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        apply_pr_20711_fix=True,
    )
    solo = engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)
    engine.step_until(solo, n=4)
    solo_tokens = engine.output_history(solo)

    batched = [
        engine.admit(prompt=_fake_prompt(32), max_new_tokens=4) for _ in range(4)
    ]
    engine.step_until_idle(max_steps=20)
    target_in_batch = batched[0]

    assert engine.output_history(target_in_batch) == solo_tokens
    engine.assert_no_canary_violations()
    engine.shutdown()

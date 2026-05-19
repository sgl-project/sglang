"""Sampler-override hookpoint self-unit tests."""

from __future__ import annotations

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
    from sglang.srt.mock_mode.sampler_override import sampler_override_is_active
except ImportError:
    MockEngine = None
    sampler_override_is_active = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def test_sampler_override_forces_oracle_token() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    reqs = [engine.admit(prompt=[i, i + 1, i + 2], max_new_tokens=1) for i in range(32)]
    engine.step_until_idle(max_steps=8)

    for req in reqs:
        observed = engine.output_history(req)
        expected = [
            engine.oracle.predict_output_token(req=req, step=s)
            for s in range(len(observed))
        ]
        assert observed == expected
    engine.shutdown()


def test_sampler_path_still_executes() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
    )
    engine.admit(prompt=[1, 2, 3, 4], max_new_tokens=2, top_p=0.8, temperature=0.7)

    engine.step()
    sampler_stats = engine.sampler_stats()

    assert sampler_stats.top_p_invocations > 0
    assert sampler_stats.temperature_invocations > 0
    engine.shutdown()


def test_top_p_intermediate_results_not_asserted() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    engine.admit(prompt=[1, 2, 3, 4], max_new_tokens=2, top_p=0.5)
    engine.step()

    engine.assert_no_canary_violations()

    engine.shutdown()


def test_eos_token_propagates_to_req() -> None:
    engine = MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)
    req = engine.admit(prompt=[1, 2, 3], max_new_tokens=8, eos_at=2)

    engine.step_until_idle(max_steps=10)

    assert engine.is_finished(req)
    engine.shutdown()

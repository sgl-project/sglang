"""Vectorized MockOracle helpers: predict_input_tokens_for_plan / predict_next_tokens_for_active_batch."""

from __future__ import annotations

import pytest
import torch

try:
    from sglang.srt.mock_mode import MockEngine
    from sglang.srt.mock_mode.oracle import MockOracle
except ImportError:
    MockEngine = None
    MockOracle = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="awaits mock_mode subsystem reimplementation; deleted in commit 8dcfc979d3"
)


def _launch() -> "MockEngine":
    return MockEngine.launch(model="Qwen/Qwen3-0.6B", num_hidden_layers=1)


def test_predict_input_tokens_for_plan_batch() -> None:
    engine = _launch()
    req_a = engine.admit(prompt=[1, 2, 3, 4], max_new_tokens=1)
    req_b = engine.admit(prompt=[5, 6, 7], max_new_tokens=1)
    engine.step()
    plan = engine.last_write_plan()

    expected_tokens, expected_positions = engine.oracle.predict_input_tokens_for_plan(
        plan=plan, forward_batch=engine.last_forward_batch()
    )

    assert len(expected_tokens) == plan.num_write_entries
    assert len(expected_positions) == plan.num_write_entries
    engine.shutdown()


def test_predict_input_tokens_for_plan_matches_scalar() -> None:
    engine = _launch()
    req = engine.admit(prompt=[10, 20, 30, 40, 50], max_new_tokens=1)
    engine.step()
    plan = engine.last_write_plan()

    vectorized_tokens, vectorized_positions = (
        engine.oracle.predict_input_tokens_for_plan(
            plan=plan, forward_batch=engine.last_forward_batch()
        )
    )
    scalar_tokens = [
        engine.oracle.predict_input_token(req=req, position=pos)
        for pos in vectorized_positions
    ]

    assert list(vectorized_tokens) == scalar_tokens
    engine.shutdown()


def test_predict_next_tokens_for_active_batch_batch() -> None:
    engine = _launch()
    engine.admit(prompt=[1, 2, 3], max_new_tokens=2)
    engine.admit(prompt=[4, 5, 6], max_new_tokens=2)
    engine.step_until_idle(max_steps=5)

    fb = engine.last_forward_batch()
    next_tokens = engine.oracle.predict_next_tokens_for_active_batch(
        forward_batch=fb, device=torch.device("cuda")
    )

    assert next_tokens.shape == (len(fb.req_pool_indices),)
    assert next_tokens.dtype == torch.int64
    engine.shutdown()


def test_predict_next_tokens_matches_scalar() -> None:
    engine = _launch()
    reqs = [
        engine.admit(prompt=[1, 2, 3], max_new_tokens=2),
        engine.admit(prompt=[4, 5, 6], max_new_tokens=2),
    ]
    engine.step()

    fb = engine.last_forward_batch()
    vectorized = engine.oracle.predict_next_tokens_for_active_batch(
        forward_batch=fb, device=torch.device("cuda")
    )
    scalar = [
        engine.oracle.predict_output_token(req=r, step=len(engine.output_history(r)))
        for r in reqs
    ]

    assert vectorized.tolist() == scalar
    engine.shutdown()


def test_vectorized_handles_chunked_offsets() -> None:
    long_prompt = list(range(1, 8193))
    engine = _launch()
    engine.admit(prompt=long_prompt, max_new_tokens=1)
    engine.step()
    plan = engine.last_write_plan()

    expected_tokens, expected_positions = engine.oracle.predict_input_tokens_for_plan(
        plan=plan, forward_batch=engine.last_forward_batch()
    )

    assert all(t == long_prompt[p] for t, p in zip(expected_tokens, expected_positions))
    engine.shutdown()

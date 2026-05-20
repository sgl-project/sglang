"""E2E: tensor parallel multi-rank under mock model + canary."""

from __future__ import annotations

import time

import pytest

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(
    reason="requires TP > 1 support which is not currently implemented"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_tp2_cross_rank_violation_lockstep_raise() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        tp_size=2,
    )
    req = engine.admit(prompt=_fake_prompt(32), max_new_tokens=2)
    engine.inject_perturbation(rank=0, kind="byte_flip")

    with pytest.raises(RuntimeError) as info:
        engine.step()
    failing_ranks = set(info.value.args[0]["ranks"])

    assert failing_ranks == {0, 1}
    engine.shutdown()


def test_tp4_canary_overhead_within_budget() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        tp_size=4,
        canary_full=True,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=16)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    overhead_pct = engine.canary_overhead_pct()
    threshold = engine.canary_overhead_threshold_pct()

    assert overhead_pct <= threshold
    engine.shutdown()

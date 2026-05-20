"""E2E: Mamba / state-space model under mock model + canary."""

from __future__ import annotations

import time

import pytest

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(
    reason="requires Mamba state mirror which is not currently implemented"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_mamba_extra_buffer_overlap_canary_clean() -> None:
    engine = SteppableEngine.launch(
        model="state-spaces/mamba-130m",
        num_hidden_layers=1,
        enable_overlap=True,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=16)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_mamba_state_kv_layout_real_data_all() -> None:
    engine = SteppableEngine.launch(
        model="state-spaces/mamba-130m",
        num_hidden_layers=1,
        canary_real_data="all",
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until_idle(max_steps=20)
    engine.assert_no_canary_violations()

    engine.shutdown()

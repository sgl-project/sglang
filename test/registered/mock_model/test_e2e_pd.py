"""E2E: prefill-decode split under mock model + canary."""

from __future__ import annotations

import time

import pytest

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(
    reason="requires PD disagg support which is not currently implemented"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_pd_transfer_canary_clean() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        disagg_prefill_decode=True,
        canary_full=True,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=16)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_pd_transfer_checksum_full_real_data() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        disagg_prefill_decode=True,
        canary_real_data="all",
        sweep_every_n=4,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=8)
    engine.inject_perturbation(channel="pd_transfer", kind="byte_flip")

    engine.step_until_idle(max_steps=40)
    violations = engine.canary_violations()

    assert any("REAL_KV" in v.fail_reason_name for v in violations)
    engine.shutdown()


def test_pd_transfer_corrupted_byte_detected() -> None:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        disagg_prefill_decode=True,
        canary_real_data="all",
        sweep_every_n=2,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)
    engine.inject_perturbation(channel="pd_transfer", kind="single_byte_corruption")

    engine.step_until_idle(max_steps=20)
    violations = engine.canary_violations()

    assert any("REAL_KV" in v.fail_reason_name for v in violations)
    engine.shutdown()

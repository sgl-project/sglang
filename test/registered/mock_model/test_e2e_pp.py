"""E2E: pipeline parallel under mock model + canary."""

from __future__ import annotations

import time

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="extra-a-2-gpu")

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_pp2_canary_clean() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=2,
        pp_size=2,
        canary_full=True,
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=16)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_pp2_layer_split_real_kv_source_layout() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=2,
        pp_size=2,
        canary_real_data="all",
    )
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=4)

    engine.step_until_idle(max_steps=20)
    engine.assert_no_canary_violations()

    engine.shutdown()

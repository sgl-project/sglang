"""Regression for PR #25015 EAGLE positions misalign: revert the fix and expect canary fire."""

from __future__ import annotations

import pytest

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_eagle_positions_misalign_regression(capfd, monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DEBUG_REVERT_PR_25015_FIX", "1")
    engine = Engine(
        model_path="Qwen/Qwen3-0.6B",
        mock_model_enabled=True,
        speculative_algorithm="EAGLE",
        kv_canary="raise",
        kv_canary_input_check=True,
    )
    try:
        with pytest.raises(Exception):
            engine.generate(
                input_ids=_fake_prompt(64),
                sampling_params={"max_new_tokens": 4, "temperature": 0.0},
            )
        captured = capfd.readouterr()
        assert "POSITION_MISMATCH" in captured.err
    finally:
        engine.shutdown()


def test_eagle_positions_match_with_fix() -> None:
    engine = Engine(
        model_path="Qwen/Qwen3-0.6B",
        mock_model_enabled=True,
        speculative_algorithm="EAGLE",
        kv_canary="raise",
        kv_canary_input_check=True,
    )
    try:
        engine.generate(
            input_ids=_fake_prompt(64),
            sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        )
    finally:
        engine.shutdown()

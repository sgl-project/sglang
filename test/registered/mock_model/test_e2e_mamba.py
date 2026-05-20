"""End-to-end Mamba smoke under mock_model + canary."""

from __future__ import annotations

import pytest

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")

pytestmark = pytest.mark.skip(
    reason="requires Mamba state mirror which is not currently implemented"
)


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_mamba_no_canary_violation() -> None:
    engine = Engine(
        model_path="state-spaces/mamba-130m",
        mock_model_enabled=True,
        kv_canary="raise",
        kv_canary_input_check=True,
    )
    try:
        engine.generate(
            input_ids=_fake_prompt(32),
            sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        )
    finally:
        engine.shutdown()

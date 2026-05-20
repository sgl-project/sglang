"""E2E: multimodal (image embed token) under mock model + canary."""

from __future__ import annotations

import time

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


_IMAGE_TOKEN = 32_000


def _mixed_prompt(text_len: int, image_count: int) -> list[int]:
    return [_IMAGE_TOKEN] * image_count + list(range(1, text_len + 1))


def test_multimodal_image_embed_canary_clean() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        multimodal=True,
        canary_full=True,
    )
    engine.admit(prompt=_mixed_prompt(text_len=16, image_count=4), max_new_tokens=8)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_multimodal_special_token_position_chain() -> None:
    engine = MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        multimodal=True,
        canary_full=True,
    )
    prompt = _mixed_prompt(text_len=8, image_count=4)
    req = engine.admit(prompt=prompt, max_new_tokens=2)

    engine.step()
    plan = engine.last_write_plan()
    written_positions = engine.last_mock_expected_positions().tolist()

    assert written_positions == list(range(len(prompt)))
    engine.assert_no_canary_violations()
    engine.shutdown()

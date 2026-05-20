"""E2E: spec decoding v2 EAGLE path under mock model + canary."""

from __future__ import annotations

import os
import time

import pytest

try:
    from sglang.srt.mock_mode import MockEngine
except ImportError:
    MockEngine = None

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-1-gpu")

pytestmark = pytest.mark.skip(reason="MockEngine harness not yet implemented.")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def _launch_eagle_v2(**kwargs) -> "MockEngine":
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
    return MockEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        speculative_algorithm="EAGLE",
        canary_real_data="all",
        **kwargs,
    )


def test_eagle_v2_draft_verify_chain_canary_clean() -> None:
    engine = _launch_eagle_v2()
    engine.admit(prompt=_fake_prompt(32), max_new_tokens=16)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        engine.step()
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_eagle_v2_with_chunked_prefill() -> None:
    engine = _launch_eagle_v2(chunked_prefill_size=512)
    engine.admit(prompt=_fake_prompt(4096), max_new_tokens=4)

    engine.step_until_idle(max_steps=200)
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_eagle_v2_with_radix_cache_hit() -> None:
    engine = _launch_eagle_v2(radix_cache=True)
    shared = _fake_prompt(64)
    engine.admit(prompt=shared, max_new_tokens=4)
    engine.step_until_idle(max_steps=20)
    engine.admit(prompt=shared + [101, 102], max_new_tokens=4)

    engine.step_until_idle(max_steps=20)
    engine.assert_no_canary_violations()

    engine.shutdown()


def test_v2_env_var_explicitly_set() -> None:
    _launch_eagle_v2().shutdown()

    assert (
        os.environ.get("SGLANG_ENABLE_SPEC_V2") == "1"
    ), "fixture must explicitly set SGLANG_ENABLE_SPEC_V2=1 before launching"

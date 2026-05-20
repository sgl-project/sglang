"""Scripted-PR regression self-e2e for #25015 EAGLE positions misalign."""

from __future__ import annotations

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-test-1-gpu-large")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def test_eagle_positions_misalign_regression() -> None:
    engine = SteppableEngine.launch(
        model_path="Qwen/Qwen3-0.6B",
        mock_model_enabled=True,
        num_hidden_layers_override=1,
        speculative_algorithm="EAGLE",
        kv_canary="raise",
        apply_pr_25015_fix=False,
    )
    req = engine.admit(prompt=_fake_prompt(64), max_new_tokens=4)

    engine.step()
    engine.step_until(req, n=2)
    violations = engine.canary_violations()

    assert any("POSITION_MISMATCH" in v.fail_reason_name for v in violations)
    engine.shutdown()


def test_eagle_positions_match_with_fix() -> None:
    engine = SteppableEngine.launch(
        model_path="Qwen/Qwen3-0.6B",
        mock_model_enabled=True,
        num_hidden_layers_override=1,
        speculative_algorithm="EAGLE",
        kv_canary="raise",
        apply_pr_25015_fix=True,
    )
    req = engine.admit(prompt=_fake_prompt(64), max_new_tokens=4)

    engine.step()
    engine.step_until(req, n=4)
    engine.assert_no_canary_violations()

    engine.shutdown()

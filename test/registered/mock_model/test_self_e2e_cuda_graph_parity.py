"""CUDA-graph parity self-e2e: graph capture vs eager violation set byte-equal."""

from __future__ import annotations

from sglang.srt.steppable_engine import SteppableEngine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="extra-a-test-1-gpu-large")


def _fake_prompt(length: int) -> list[int]:
    return list(range(1, length + 1))


def _run_pr_scenario(*, pr_id: str, cuda_graph: bool) -> set:
    engine = SteppableEngine.launch(
        model="Qwen/Qwen3-0.6B",
        num_hidden_layers=1,
        cuda_graph=cuda_graph,
        scripted_pr_scenario=pr_id,
    )
    req = engine.admit(prompt=_fake_prompt(64), max_new_tokens=4)
    engine.step_until(req, n=4)
    violations = engine.canary_violations()
    engine.shutdown()
    return {(v.req_id, v.position, v.fail_reason_bits) for v in violations}


def test_graph_capture_vs_eager_same_violation_set_pr_25015() -> None:
    captured = _run_pr_scenario(pr_id="pr_25015", cuda_graph=True)
    eager = _run_pr_scenario(pr_id="pr_25015", cuda_graph=False)
    assert captured == eager


def test_graph_capture_vs_eager_same_violation_set_pr_24230() -> None:
    captured = _run_pr_scenario(pr_id="pr_24230", cuda_graph=True)
    eager = _run_pr_scenario(pr_id="pr_24230", cuda_graph=False)
    assert captured == eager


def test_graph_capture_vs_eager_same_violation_set_pr_24401() -> None:
    captured = _run_pr_scenario(pr_id="pr_24401", cuda_graph=True)
    eager = _run_pr_scenario(pr_id="pr_24401", cuda_graph=False)
    assert captured == eager


def test_graph_capture_vs_eager_same_violation_set_pr_22819() -> None:
    captured = _run_pr_scenario(pr_id="pr_22819", cuda_graph=True)
    eager = _run_pr_scenario(pr_id="pr_22819", cuda_graph=False)
    assert captured == eager


def test_graph_capture_vs_eager_same_violation_set_pr_20711() -> None:
    captured = _run_pr_scenario(pr_id="pr_20711", cuda_graph=True)
    eager = _run_pr_scenario(pr_id="pr_20711", cuda_graph=False)
    assert captured == eager

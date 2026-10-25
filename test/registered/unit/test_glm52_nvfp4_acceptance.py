"""Acceptance gates must reject incomplete or regressed benchmark results."""

import importlib.util
import json
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_path = Path(__file__).resolve().parents[3] / "scripts/bench_glm52_nvfp4_layersplit.py"
_spec = importlib.util.spec_from_file_location("glm52_acceptance", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
compare_results = _module.compare_results

_perf_path = _path.with_name("bench_glm52_nvfp4_layersplit_perf.py")
_perf_spec = importlib.util.spec_from_file_location("glm52_performance", _perf_path)
_perf = importlib.util.module_from_spec(_perf_spec)
_perf_spec.loader.exec_module(_perf)


def _results():
    runs = {
        phase: [
            dict(score=0.94, latency=100.0, output_throughput=1000.0) for _ in range(3)
        ]
        for phase in ("baseline", "layer_split")
    }
    allocations = {}
    for phase, layers in (("baseline", 10), ("layer_split", 4)):
        allocations[phase] = [
            {
                "device": f"cuda:{rank}",
                "layers": 10,
                "capacity_tokens": 131072,
                "layer_split": phase == "layer_split",
                "main_bytes": 416 * 131072 * layers,
                "index_bytes": 132 * 131072 * layers,
                "remote_main_bytes": 0,
                "remote_index_bytes": 0,
                "global_scale_bytes": 40,
                "total_bytes": 548 * 131072 * layers + 40,
            }
            for rank in range(4)
        ]
    return runs, allocations


def test_complete_results_pass():
    assert compare_results(*_results())["passed"]


@pytest.mark.parametrize(
    "problem", ["missing_run", "missing_pool", "accounting", "capacity"]
)
def test_incomplete_or_unmatched_results_fail(problem):
    runs, allocations = _results()
    if problem == "missing_run":
        runs["baseline"].pop()
    elif problem == "missing_pool":
        allocations["layer_split"].pop()
    elif problem == "accounting":
        allocations["layer_split"][0]["total_bytes"] += 1
    else:
        allocations["layer_split"][0]["capacity_tokens"] *= 2
    assert not compare_results(runs, allocations)["passed"]


@pytest.mark.parametrize(
    "metric,value",
    [
        ("score", 0.934),
        ("latency", 106.0),
        ("output_throughput", 949.0),
        ("output_throughput", 0.0),
    ],
)
def test_accuracy_or_performance_regressions_fail(metric, value):
    runs, allocations = _results()
    for sample in runs["layer_split"]:
        sample[metric] = value
    assert not compare_results(runs, allocations)["passed"]


def _synthetic_results():
    runs = {}
    for phase in ("baseline", "layer_split"):
        runs[phase] = {}
        for workload in _perf.WORKLOADS:
            runs[phase][workload["name"]] = [
                {
                    "completed": workload["requests"],
                    "total_input_tokens": workload["requests"] * workload["input"],
                    "total_output_tokens": workload["requests"] * workload["output"],
                    "output_throughput": 1000.0 if phase == "baseline" else 900.0,
                    "duration": 100.0,
                    "median_ttft_ms": 20.0,
                    "median_tpot_ms": 2.0,
                    "errors": [],
                }
                for _ in range(3)
            ]
    return runs, _results()[1]


def test_synthetic_comparison_reports_slowdown_without_accuracy_claim():
    result = _perf.summarize_performance(*_synthetic_results())
    assert result["accuracy_evaluated"] is False
    assert result["workloads"]["8k_cc32"]["layer_split_change_percent"][
        "output_throughput"
    ] == pytest.approx(-10.0)


@pytest.mark.parametrize(
    "problem", ["missing_run", "short_output", "wrong_input", "request_error", "nan"]
)
def test_synthetic_comparison_rejects_incomplete_measurements(problem):
    runs, allocations = _synthetic_results()
    samples = runs["layer_split"]["8k_cc32"]
    if problem == "missing_run":
        samples.pop()
    elif problem == "short_output":
        samples[0]["total_output_tokens"] -= 1
    elif problem == "wrong_input":
        samples[0]["total_input_tokens"] -= 1
    elif problem == "request_error":
        samples[0]["errors"] = ["HTTP 500"]
    else:
        samples[0]["output_throughput"] = float("nan")
    with pytest.raises(RuntimeError):
        _perf.summarize_performance(runs, allocations)


@pytest.mark.parametrize("test_count", [1299, 1319])
def test_full_gsm8k_keeps_all_held_out_questions(tmp_path, monkeypatch, test_count):
    from sglang.test.kits import eval_accuracy_kit
    from sglang.test.simple_eval_mixed_prefix_gsm8k import GSM8KEval

    for split, count in (("train", 20), ("test", test_count)):
        path = tmp_path / f"gsm8k_{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"question": f"{split} question {i}", "answer": str(i)})
                + "\n"
                for i in range(count)
            )
        )
    monkeypatch.setattr(
        eval_accuracy_kit,
        "download_and_cache_file",
        lambda _url, filename: filename,
    )
    if test_count != 1319:
        with pytest.raises(ValueError, match="1319 test examples"):
            eval_accuracy_kit.prepare_full_gsm8k_data(tmp_path / "combined.jsonl")
        return
    path = eval_accuracy_kit.prepare_full_gsm8k_data(tmp_path / "combined.jsonl")
    evaluation = GSM8KEval(num_examples=1319, num_shots=20, data_path=path)
    assert len(evaluation._lines) == 1319
    assert evaluation._lines[0]["question"] == "test question 0"
    assert evaluation._lines[-1]["question"] == "test question 1318"
    assert "train question 19" in evaluation._few_shot_prompt
    assert "test question" not in evaluation._few_shot_prompt

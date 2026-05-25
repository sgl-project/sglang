import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _load_eval_suite_module():
    path = Path(__file__).parents[3] / "test/manual/dsv4/eval_dsv4_indexcache_suite.py"
    spec = importlib.util.spec_from_file_location("eval_dsv4_indexcache_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_suite = _load_eval_suite_module()


def test_dsv4_index_cache_eval_suite_defaults_to_paper_relevant_tasks():
    tasks = eval_suite.selected_tasks([], ["long-context", "reasoning"])

    assert [task.name for task in tasks] == [
        "ruler",
        "mrcr_v2",
        "graphwalks",
        "longbench_v2",
        "aa_lcr",
        "aime25",
        "gpqa",
        "livecodebench",
        "ifbench",
    ]


def test_dsv4_index_cache_eval_suite_rejects_short_context_smokes():
    with pytest.raises(ValueError, match="intentionally blocked"):
        eval_suite.selected_tasks(["mmlu"], [])

    with pytest.raises(ValueError, match="intentionally blocked"):
        eval_suite.selected_tasks(["gsm8k"], [])


def test_dsv4_index_cache_eval_suite_builds_longbench_command():
    args = Namespace(
        num_threads=64,
        max_tokens=32768,
        temperature=1.0,
        top_p=0.95,
        num_examples=10,
        dataset_path="/tmp/longbench",
        min_context_length=75000,
        max_context_length=200000,
        endpoint_label="baseline",
        repeat_index=1,
    )
    task = eval_suite.TASKS["longbench_v2"]

    cmd = eval_suite.build_sglang_eval_cmd(task, "http://endpoint", args)

    assert cmd[:3] == ["python", "-m", "sglang.test.run_eval"]
    assert "--eval-name" in cmd
    assert "longbench_v2" in cmd
    assert "--min-context-length" in cmd
    assert "75000" in cmd
    assert "--dataset-path" in cmd


def test_dsv4_index_cache_eval_suite_dry_run_records_sgl_eval_command(monkeypatch):
    monkeypatch.setattr(eval_suite.shutil, "which", lambda _: "/usr/bin/sgl-eval")
    args = Namespace(
        sgl_eval_bin="sgl-eval",
        temperature=1.0,
        top_p=0.95,
        max_tokens=32768,
        num_threads=64,
        out_dir=Path("/tmp/sgl-eval-out"),
        num_examples=None,
        endpoint_label="searched_1_2",
        repeat_index=2,
        dry_run=True,
        timeout=7200,
        require_metrics=False,
    )
    task = eval_suite.TASKS["ruler"]

    result = eval_suite.run_eval_task(task, "http://endpoint", args)

    assert result["returncode"] == 0
    assert result["metrics"] == {}
    assert result["primary_metric"] is None
    assert result["cmd"][:3] == ["sgl-eval", "run", "ruler"]
    assert "http://endpoint/v1" in result["cmd"]
    assert "/tmp/sgl-eval-out/searched_1_2/ruler/repeat_2" in result["cmd"]


def test_dsv4_index_cache_eval_suite_rejects_zero_repeats(tmp_path):
    output = tmp_path / "eval.json"

    with pytest.raises(SystemExit, match="--repeats"):
        eval_suite.parse_args(
            [
                "--endpoint",
                "baseline=http://endpoint",
                "--repeats",
                "0",
                "--output",
                str(output),
                "--dry-run",
            ]
        )


def test_dsv4_index_cache_eval_suite_rejects_duplicate_endpoint_labels(tmp_path):
    output = tmp_path / "eval.json"

    with pytest.raises(SystemExit, match="duplicate endpoint labels"):
        eval_suite.parse_args(
            [
                "--endpoint",
                "baseline=http://baseline-a",
                "--endpoint",
                "baseline=http://baseline-b",
                "--output",
                str(output),
                "--dry-run",
            ]
        )


def test_dsv4_index_cache_eval_suite_requires_baseline_endpoint_label(tmp_path):
    output = tmp_path / "eval.json"

    with pytest.raises(SystemExit, match="--baseline-label"):
        eval_suite.parse_args(
            [
                "--endpoint",
                "searched_1_2=http://searched-half",
                "--baseline-label",
                "baseline",
                "--output",
                str(output),
                "--dry-run",
            ]
        )


def test_dsv4_index_cache_eval_suite_rejects_speculative_endpoint():
    with pytest.raises(RuntimeError, match="quality eval endpoint 'searched_1_2'"):
        eval_suite.validate_server_info_for_base_path(
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 1,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 3,
            },
            "searched_1_2",
        )


def test_dsv4_index_cache_eval_suite_accepts_base_path_endpoint():
    eval_suite.validate_server_info_for_base_path(
        {
            "speculative_algorithm": None,
            "speculative_num_steps": 0,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
        },
        "baseline",
    )


def test_dsv4_index_cache_eval_suite_dry_run_writes_server_checks(tmp_path):
    output = tmp_path / "eval.json"

    eval_suite.main(
        [
            "--endpoint",
            "baseline=http://baseline",
            "--endpoint",
            "searched_1_4=http://searched-quarter",
            "--task",
            "ruler",
            "--repeats",
            "2",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    result = json.loads(output.read_text())

    assert result["server_checks"] == {
        "baseline": {
            "server_info_checked": False,
            "speculative_decode": "dry run; /server_info not queried",
        },
        "searched_1_4": {
            "server_info_checked": False,
            "speculative_decode": "dry run; /server_info not queried",
        },
    }
    assert [(row["endpoint"], row["repeat"]) for row in result["results"]] == [
        ("baseline", 1),
        ("baseline", 2),
        ("searched_1_4", 1),
        ("searched_1_4", 2),
    ]
    assert {row["task"] for row in result["results"]} == {"ruler"}
    assert result["primary_metric_summary"] == {}
    assert result["primary_metric_comparison"] == {}
    assert result["quality_gate"] == {
        "max_primary_metric_drop": None,
        "passed": True,
        "failures": [],
    }


def test_dsv4_index_cache_eval_suite_extracts_stdout_metrics():
    metrics = eval_suite.extract_metrics(
        "{'event': 'ignored'}\n"
        "==========================================\n"
        "model - metrics={'score': 0.82, 'latency': 12.5} score=0.82\n"
    )

    assert metrics["score"] == 0.82
    assert metrics["latency"] == 12.5
    assert eval_suite.choose_primary_metric(metrics) == {
        "name": "score",
        "value": 0.82,
    }


def test_dsv4_index_cache_eval_suite_extracts_json_file_metrics(tmp_path):
    out_dir = tmp_path / "eval-out"
    out_dir.mkdir()
    (out_dir / "results.json").write_text(
        json.dumps({"metrics": {"accuracy": 0.71, "details": {"f1": 0.66}}})
    )

    metrics = eval_suite.extract_metrics("", out_dir)

    assert metrics["accuracy"] == 0.71
    assert metrics["details.f1"] == 0.66


def test_dsv4_index_cache_eval_suite_summarizes_primary_metrics():
    summary = eval_suite.summarize_primary_metrics(
        [
            {
                "endpoint": "baseline",
                "task": "ruler",
                "primary_metric": {"name": "score", "value": 0.8},
            },
            {
                "endpoint": "baseline",
                "task": "ruler",
                "primary_metric": {"name": "score", "value": 0.9},
            },
            {
                "endpoint": "searched_1_2",
                "task": "ruler",
                "primary_metric": {"name": "score", "value": 0.85},
            },
        ]
    )

    assert summary["baseline"]["ruler"]["metric"] == "score"
    assert summary["baseline"]["ruler"]["mean"] == pytest.approx(0.85)
    assert summary["baseline"]["ruler"]["min"] == 0.8
    assert summary["baseline"]["ruler"]["max"] == 0.9
    assert summary["baseline"]["ruler"]["n"] == 2
    assert summary["searched_1_2"]["ruler"]["mean"] == pytest.approx(0.85)


def test_dsv4_index_cache_eval_suite_compares_primary_metrics():
    summary = {
        "baseline": {
            "ruler": {"metric": "score", "mean": 0.9, "min": 0.9, "max": 0.9, "n": 1}
        },
        "searched_1_2": {
            "ruler": {
                "metric": "score",
                "mean": 0.87,
                "min": 0.87,
                "max": 0.87,
                "n": 1,
            }
        },
    }

    comparison = eval_suite.compare_primary_metrics(summary, "baseline")

    assert comparison["searched_1_2"]["ruler"] == {
        "metric": "score",
        "mean": 0.87,
        "baseline_mean": 0.9,
        "delta": pytest.approx(-0.03),
        "status": "compared",
    }


def test_dsv4_index_cache_eval_suite_enforces_quality_drop():
    comparisons = {
        "searched_1_4": {
            "ruler": {
                "metric": "score",
                "mean": 0.84,
                "baseline_mean": 0.9,
                "delta": -0.06,
                "status": "compared",
            }
        }
    }

    with pytest.raises(RuntimeError, match="searched_1_4/ruler"):
        eval_suite.enforce_quality_drop(comparisons, 0.05)

    eval_suite.enforce_quality_drop(comparisons, 0.07)


def test_dsv4_index_cache_eval_suite_writes_output_before_quality_gate_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "eval.json"

    def fake_run_eval_task(task, base_url, args):
        metric = 0.9 if args.endpoint_label == "baseline" else 0.82
        return {
            "cmd": ["fake"],
            "returncode": 0,
            "elapsed_sec": 0.1,
            "output": "",
            "metrics": {"score": metric},
            "primary_metric": {"name": "score", "value": metric},
        }

    monkeypatch.setattr(eval_suite, "run_eval_task", fake_run_eval_task)

    with pytest.raises(RuntimeError, match="quality metric drop"):
        eval_suite.main(
            [
                "--endpoint",
                "baseline=http://baseline",
                "--endpoint",
                "searched_1_4=http://searched-quarter",
                "--task",
                "ruler",
                "--max-primary-metric-drop",
                "0.05",
                "--output",
                str(output),
                "--dry-run",
            ]
        )

    result = json.loads(output.read_text())

    assert result["primary_metric_comparison"]["searched_1_4"]["ruler"][
        "delta"
    ] == pytest.approx(-0.08)
    assert result["quality_gate"]["passed"] is False
    assert result["quality_gate"]["failures"] == [
        "searched_1_4/ruler: delta -0.08 < allowed -0.05"
    ]

import importlib.util
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
    )
    task = eval_suite.TASKS["ruler"]

    result = eval_suite.run_eval_task(task, "http://endpoint", args)

    assert result["returncode"] == 0
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

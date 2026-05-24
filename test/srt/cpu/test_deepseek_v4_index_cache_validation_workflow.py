import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _load_workflow_module():
    path = (
        Path(__file__).parents[3] / "test/manual/dsv4/run_dsv4_indexcache_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_dsv4_indexcache_validation", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_workflow_module()


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        baseline_endpoint="http://baseline",
        indexcache_endpoint="http://indexcache",
        searched_half_endpoint="http://searched-half",
        searched_quarter_endpoint="http://searched-quarter",
        search_endpoint="http://search",
        calibration_jsonl=tmp_path / "calibration.jsonl",
        num_c4_layers=21,
        pp_block_c4_layers=7,
        pattern_command_template="modal deploy endpoint.py --pattern {pattern}",
        calibration_limit=8,
        profile_dir=tmp_path / "profiles",
        profile_prefix="dsv4-indexcache",
        profile_steps=4,
        profile_prompt_tokens=128000,
        profile_max_tokens=256,
        min_indexcache_prompt_tokens=75000,
        eval_num_threads=64,
        eval_repeats=3,
        eval_max_tokens=32768,
        eval_min_context_length=75000,
        max_primary_metric_drop=0.02,
        require_eval_metrics=True,
        output_dir=tmp_path / "out",
        dry_run=False,
        eagle_off_confirmed=True,
        indexcache_profile_env_confirmed=True,
    )


def test_dsv4_index_cache_validation_workflow_uses_searched_half_and_quarter(tmp_path):
    cmd = workflow.search_cmd(_args(tmp_path))

    assert "--retention" in cmd
    assert cmd.count("--retention") == 2
    assert "1/2" in cmd
    assert "1/4" in cmd
    assert "--pp-block-c4-layers" in cmd
    assert "--command-template" in cmd
    assert "modal deploy endpoint.py --pattern {pattern}" in cmd
    assert "--min-indexcache-prompt-tokens" in cmd
    assert "75000" in cmd


def test_dsv4_index_cache_validation_workflow_runs_paper_relevant_eval_suite(tmp_path):
    cmd = workflow.eval_cmd(_args(tmp_path))

    assert "baseline=http://baseline" in cmd
    assert "searched_1_2=http://searched-half" in cmd
    assert "searched_1_4=http://searched-quarter" in cmd
    assert "long-context" in cmd
    assert "reasoning" in cmd
    assert "--repeats" in cmd
    assert "3" in cmd
    assert "--min-context-length" in cmd
    assert "75000" in cmd
    assert "--baseline-label" in cmd
    assert "baseline" in cmd
    assert "--max-primary-metric-drop" in cmd
    assert "0.02" in cmd
    assert "--require-metrics" in cmd
    assert "mmlu" not in cmd
    assert "gsm8k" not in cmd


def test_dsv4_index_cache_validation_workflow_labels_searched_eval_endpoints(
    tmp_path,
):
    endpoints = workflow.quality_eval_endpoints(_args(tmp_path))

    assert endpoints == [
        ("baseline", "http://baseline"),
        ("searched_1_2", "http://searched-half"),
        ("searched_1_4", "http://searched-quarter"),
    ]


def test_dsv4_index_cache_validation_workflow_labels_artifacts(tmp_path):
    args = _args(tmp_path)
    artifacts = workflow.validation_artifacts(args)

    assert artifacts == {
        "profile": tmp_path / "out" / "profile.json",
        "searched_patterns": tmp_path / "out" / "searched_patterns.json",
        "quality_eval": tmp_path / "out" / "quality_eval.json",
        "validation_summary": tmp_path / "out" / "validation_summary.json",
    }


def test_dsv4_index_cache_validation_workflow_passes_eagle_confirmation_to_profile(
    tmp_path,
):
    cmd = workflow.profile_cmd(_args(tmp_path))

    assert "--eagle-off-confirmed" in cmd


def test_dsv4_index_cache_validation_workflow_passes_profile_env_confirmation(
    tmp_path,
):
    cmd = workflow.profile_cmd(_args(tmp_path))

    assert "--indexcache-profile-env-confirmed" in cmd


def test_dsv4_index_cache_validation_workflow_passes_context_floor_to_profile(
    tmp_path,
):
    cmd = workflow.profile_cmd(_args(tmp_path))

    assert "--min-indexcache-prompt-tokens" in cmd
    assert "75000" in cmd


def test_dsv4_index_cache_validation_workflow_requires_eagle_off_for_real_runs(
    tmp_path,
):
    args = _args(tmp_path)
    args.eagle_off_confirmed = False

    with pytest.raises(SystemExit, match="--eagle-off-confirmed"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_requires_profile_env_for_real_runs(
    tmp_path,
):
    args = _args(tmp_path)
    args.indexcache_profile_env_confirmed = False

    with pytest.raises(SystemExit, match="--indexcache-profile-env-confirmed"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_requires_searched_half_endpoint(
    tmp_path,
):
    args = _args(tmp_path)
    args.searched_half_endpoint = None

    with pytest.raises(SystemExit, match="--searched-half-endpoint"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_requires_searched_quarter_endpoint(
    tmp_path,
):
    args = _args(tmp_path)
    args.searched_quarter_endpoint = None

    with pytest.raises(SystemExit, match="--searched-quarter-endpoint"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_rejects_duplicate_searched_endpoints(
    tmp_path,
):
    args = _args(tmp_path)
    args.searched_quarter_endpoint = args.searched_half_endpoint + "/"

    with pytest.raises(SystemExit, match="must be distinct"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_requires_pattern_placeholder(
    tmp_path,
):
    args = _args(tmp_path)
    args.pattern_command_template = "modal deploy endpoint.py"

    with pytest.raises(SystemExit, match=r"\{pattern\}"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_allows_dry_run_without_eagle_confirmation(
    tmp_path,
):
    args = _args(tmp_path)
    args.dry_run = True
    args.eagle_off_confirmed = False
    args.indexcache_profile_env_confirmed = False
    args.searched_half_endpoint = None
    args.searched_quarter_endpoint = None

    workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_rejects_short_profile_prompt(
    tmp_path,
):
    args = _args(tmp_path)
    args.profile_prompt_tokens = 20000

    with pytest.raises(SystemExit, match="--profile-prompt-tokens"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_rejects_short_eval_context(
    tmp_path,
):
    args = _args(tmp_path)
    args.eval_min_context_length = 20000

    with pytest.raises(SystemExit, match="--eval-min-context-length"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_dry_run_writes_summary(tmp_path):
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"text": "calibration"}\n')
    output_dir = tmp_path / "out"

    workflow.main(
        [
            "--baseline-endpoint",
            "http://baseline",
            "--indexcache-endpoint",
            "http://indexcache",
            "--search-endpoint",
            "http://search",
            "--calibration-jsonl",
            str(calibration),
            "--num-c4-layers",
            "21",
            "--pp-block-c4-layers",
            "7",
            "--pattern-command-template",
            "deploy --pattern {pattern}",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    summary = json.loads((output_dir / "validation_summary.json").read_text())

    assert summary["uniform_1_4"] == "not run; only searched 1/4 is generated"
    assert summary["quality_eval_endpoints"] == {
        "baseline": "http://baseline",
        "searched_1_2": "http://indexcache",
        "searched_1_4": "http://indexcache",
    }
    assert summary["context_gate"] == {
        "min_indexcache_prompt_tokens": 75000,
        "profile_prompt_tokens": 128000,
        "eval_min_context_length": 75000,
    }
    assert summary["artifacts"] == {
        "profile": {
            "path": str(output_dir / "profile.json"),
            "exists": False,
        },
        "searched_patterns": {
            "path": str(output_dir / "searched_patterns.json"),
            "exists": False,
        },
        "quality_eval": {
            "path": str(output_dir / "quality_eval.json"),
            "exists": False,
        },
        "validation_summary": {
            "path": str(output_dir / "validation_summary.json"),
            "exists": False,
        },
    }
    assert [phase["phase"] for phase in summary["phases"]] == [
        "profile",
        "search",
        "eval",
    ]

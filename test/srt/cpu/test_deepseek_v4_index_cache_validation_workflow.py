import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _load_workflow_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/run_dsv4_indexcache_validation.py"
    )
    spec = importlib.util.spec_from_file_location("run_dsv4_indexcache_validation", path)
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
        search_endpoint="http://search",
        calibration_jsonl=tmp_path / "calibration.jsonl",
        num_c4_layers=21,
        pp_block_c4_layers=7,
        pattern_command_template=None,
        calibration_limit=8,
        profile_dir=tmp_path / "profiles",
        profile_prefix="dsv4-indexcache",
        profile_steps=4,
        profile_prompt_tokens=128000,
        profile_max_tokens=256,
        eval_num_threads=64,
        eval_max_tokens=32768,
        output_dir=tmp_path / "out",
        dry_run=False,
        eagle_off_confirmed=True,
    )


def test_dsv4_index_cache_validation_workflow_uses_searched_half_and_quarter(tmp_path):
    cmd = workflow.search_cmd(_args(tmp_path))

    assert "--retention" in cmd
    assert cmd.count("--retention") == 2
    assert "1/2" in cmd
    assert "1/4" in cmd
    assert "--pp-block-c4-layers" in cmd


def test_dsv4_index_cache_validation_workflow_runs_paper_relevant_eval_suite(tmp_path):
    cmd = workflow.eval_cmd(_args(tmp_path))

    assert "baseline=http://baseline" in cmd
    assert "indexcache=http://indexcache" in cmd
    assert "long-context" in cmd
    assert "reasoning" in cmd
    assert "mmlu" not in cmd
    assert "gsm8k" not in cmd


def test_dsv4_index_cache_validation_workflow_passes_eagle_confirmation_to_profile(
    tmp_path,
):
    cmd = workflow.profile_cmd(_args(tmp_path))

    assert "--eagle-off-confirmed" in cmd


def test_dsv4_index_cache_validation_workflow_requires_eagle_off_for_real_runs(
    tmp_path,
):
    args = _args(tmp_path)
    args.eagle_off_confirmed = False

    with pytest.raises(SystemExit, match="--eagle-off-confirmed"):
        workflow.validate_args(args)


def test_dsv4_index_cache_validation_workflow_allows_dry_run_without_eagle_confirmation(
    tmp_path,
):
    args = _args(tmp_path)
    args.dry_run = True
    args.eagle_off_confirmed = False

    workflow.validate_args(args)

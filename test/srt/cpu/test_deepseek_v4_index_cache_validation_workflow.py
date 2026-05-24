import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


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

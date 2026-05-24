import importlib.util
import sys
from pathlib import Path

import pytest


def _load_eval_suite_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/eval_dsv4_indexcache_suite.py"
    )
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

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_base_path_module():
    path = Path(__file__).parents[3] / "test/manual/dsv4/indexcache_base_path.py"
    spec = importlib.util.spec_from_file_location("indexcache_base_path", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base_path = _load_base_path_module()


def test_dsv4_index_cache_base_path_detects_speculative_server_info():
    server_info = {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": "1",
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 3,
        "enable_multi_layer_eagle": True,
    }

    assert base_path.speculative_config_paths(server_info) == [
        "speculative_algorithm=EAGLE",
        "speculative_num_steps=1",
        "speculative_eagle_topk=1",
        "speculative_num_draft_tokens=3",
        "enable_multi_layer_eagle=True",
    ]


def test_dsv4_index_cache_base_path_ignores_disabled_speculative_knobs():
    server_info = {
        "speculative_algorithm": None,
        "speculative_num_steps": 0,
        "speculative_eagle_topk": None,
        "speculative_num_draft_tokens": "0",
        "enable_multi_layer_eagle": False,
    }

    assert base_path.speculative_config_paths(server_info) == []
    assert base_path.validate_server_info_for_base_path(server_info) == []


def test_dsv4_index_cache_base_path_reports_nested_speculative_paths():
    server_info = {
        "decode": [
            {
                "worker": {
                    "speculative_algorithm": None,
                    "speculative_num_draft_tokens": "4",
                }
            }
        ]
    }

    with pytest.raises(
        RuntimeError,
        match=r"decode\[0\].worker.speculative_num_draft_tokens=4",
    ):
        base_path.validate_server_info_for_base_path(server_info, "candidate endpoint")

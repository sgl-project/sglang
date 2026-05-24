import importlib.util
import sys
from pathlib import Path

import pytest


def _load_search_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/search_dsv4_indexcache_pattern.py"
    )
    spec = importlib.util.spec_from_file_location("search_dsv4_indexcache_pattern", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


search_mod = _load_search_module()


def test_dsv4_index_cache_search_keeps_pp_block_anchors_full():
    priority = {5: 0, 3: 1, 1: 2, 2: 3, 6: 4, 7: 5}

    result = search_mod.greedy_search_pattern(
        num_c4_layers=8,
        retention="1/2",
        pp_block_c4_layers=4,
        score_pattern=lambda pattern: sum(
            priority.get(i, 10) for i, value in enumerate(pattern) if value == "S"
        ),
    )
    history = result["history"]

    assert [step["flip"] for step in history] == [5, 3, 1, 2]
    assert result["initial_pattern"] == "FFFFFFFF"
    assert result["final_pattern"] == "FSSSFSFF"
    assert result["target_f_layers"] == 4
    assert result["protected_c4_indices"] == [0, 4]
    assert result["final_pattern"][0] == "F"
    assert result["final_pattern"][4] == "F"


def test_dsv4_index_cache_search_can_target_quarter_retention():
    def score(pattern: str) -> float:
        return pattern.rfind("S")

    result = search_mod.greedy_search_pattern(
        num_c4_layers=8,
        retention="1/4",
        pp_block_c4_layers=0,
        score_pattern=score,
    )

    assert result["final_pattern"].count("F") == 2
    assert result["final_pattern"][0] == "F"
    assert result["target_f_layers"] == 2


def test_dsv4_index_cache_search_rejects_impossible_protected_retention():
    with pytest.raises(ValueError, match="C4 anchors are protected"):
        search_mod.greedy_search_pattern(
            num_c4_layers=8,
            retention="1/4",
            pp_block_c4_layers=2,
            score_pattern=lambda _: 0.0,
        )

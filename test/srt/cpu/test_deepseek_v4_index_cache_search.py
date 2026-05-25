import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _load_search_module():
    path = (
        Path(__file__).parents[3] / "test/manual/dsv4/search_dsv4_indexcache_pattern.py"
    )
    spec = importlib.util.spec_from_file_location(
        "search_dsv4_indexcache_pattern", path
    )
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
    assert history[0]["candidates"][0] == {
        "pattern": "FFFFFSFF",
        "flip": 5,
        "loss": 0,
        "loss_delta": 0,
    }
    assert history[0]["candidates"] == sorted(
        history[0]["candidates"],
        key=lambda item: (item["loss_delta"], item["flip"]),
    )
    assert result["initial_pattern"] == "FFFFFFFF"
    assert result["baseline_loss"] == 0
    assert result["search_method"] == "greedy_training_free"
    assert result["uniform_candidate"] is False
    assert result["final_pattern"] == "FSSSFSFF"
    assert result["final_pattern_counts"] == {"F": 4, "S": 4}
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
    assert result["uniform_candidate"] is False
    assert result["target_f_layers"] == 2


def test_dsv4_index_cache_search_ties_break_by_layer_index():
    result = search_mod.greedy_search_pattern(
        num_c4_layers=4,
        retention="1/2",
        pp_block_c4_layers=0,
        score_pattern=lambda _: 1.0,
    )

    assert [step["flip"] for step in result["history"]] == [1, 2]


def test_dsv4_index_cache_search_records_loss_delta_from_all_full_baseline():
    losses = {
        "FFFF": 10.0,
        "FSFF": 11.0,
        "FFSF": 10.5,
        "FFFS": 11.5,
        "FSSF": 12.0,
        "FFSS": 10.8,
    }

    result = search_mod.greedy_search_pattern(
        num_c4_layers=4,
        retention="1/2",
        pp_block_c4_layers=0,
        score_pattern=losses.__getitem__,
    )

    assert result["baseline_loss"] == 10.0
    assert [step["flip"] for step in result["history"]] == [2, 3]
    assert result["history"][0]["loss_delta"] == 0.5
    assert result["history"][0]["candidates"][0] == {
        "pattern": "FFSF",
        "flip": 2,
        "loss": 10.5,
        "loss_delta": 0.5,
    }


def test_dsv4_index_cache_search_rejects_impossible_protected_retention():
    with pytest.raises(ValueError, match="C4 anchors are protected"):
        search_mod.greedy_search_pattern(
            num_c4_layers=8,
            retention="1/4",
            pp_block_c4_layers=2,
            score_pattern=lambda _: 0.0,
        )


def test_dsv4_index_cache_search_requires_pattern_deployment_template():
    args = type("Args", (), {"command_template": None})()

    with pytest.raises(SystemExit, match="--command-template is required"):
        search_mod.validate_args(args)


def test_dsv4_index_cache_search_requires_pattern_placeholder():
    args = type(
        "Args",
        (),
        {
            "command_template": "modal deploy endpoint.py",
            "num_c4_layers": 21,
            "pp_block_c4_layers": 0,
            "startup_timeout": 1,
            "request_timeout": 1,
            "min_indexcache_prompt_tokens": 75000,
        },
    )()

    with pytest.raises(SystemExit, match=r"\{pattern\}"):
        search_mod.validate_args(args)


def test_dsv4_index_cache_search_rejects_invalid_runtime_args():
    args = type(
        "Args",
        (),
        {
            "command_template": "deploy --pattern {pattern}",
            "num_c4_layers": 0,
            "pp_block_c4_layers": 0,
            "startup_timeout": 1,
            "request_timeout": 1,
            "min_indexcache_prompt_tokens": 75000,
        },
    )()

    with pytest.raises(SystemExit, match="--num-c4-layers"):
        search_mod.validate_args(args)


def test_dsv4_index_cache_search_surfaces_failed_candidate_deploy():
    proc = subprocess.Popen(["sh", "-c", "exit 7"])
    proc.wait(timeout=5)

    with pytest.raises(RuntimeError, match="status 7"):
        search_mod.wait_for_endpoint("http://127.0.0.1:9", timeout=30, proc=proc)


def test_dsv4_index_cache_search_rejects_short_tokenized_calibration(
    monkeypatch,
):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"meta_info": {"input_token_logprobs": [[-1.0], [-2.0]]}}

    monkeypatch.setattr(search_mod.requests, "post", lambda *_, **__: Response())

    with pytest.raises(RuntimeError, match="below IndexCache floor"):
        search_mod.score_endpoint(
            "http://endpoint",
            ["short prompt"],
            timeout=1,
            min_prompt_tokens=3,
        )


def test_dsv4_index_cache_search_rejects_speculative_candidate_server():
    with pytest.raises(RuntimeError, match="speculative_algorithm=EAGLE"):
        search_mod.validate_server_info_for_base_path(
            {
                "speculative_algorithm": "EAGLE",
                "speculative_num_steps": 1,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 3,
            }
        )


def test_dsv4_index_cache_search_accepts_base_path_candidate_server():
    search_mod.validate_server_info_for_base_path(
        {
            "speculative_algorithm": None,
            "speculative_num_steps": 0,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
        }
    )

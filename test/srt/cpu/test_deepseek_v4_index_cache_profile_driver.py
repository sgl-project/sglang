import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _load_profile_driver_module():
    path = (
        Path(__file__).parents[3]
        / "test/manual/dsv4/profile_dsv4_indexcache_endpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "profile_dsv4_indexcache_endpoint", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile_driver = _load_profile_driver_module()


def test_dsv4_index_cache_profile_driver_finds_and_summarizes_traces(tmp_path):
    trace_path = tmp_path / "dsv4-indexcache-rank0.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "dsv4_indexcache.csa_indexer.layer_2", "dur": 1000},
                    {"name": "dsv4_indexcache.cuda_graph.decode.replay", "dur": 0},
                ]
            }
        )
    )

    traces = profile_driver.find_trace_files(tmp_path, "dsv4-indexcache")

    assert traces == [trace_path]
    summary = profile_driver.summarize_trace(traces[0])
    assert summary["categories"]["csa_indexer"]["total_ms"] == 1.0
    assert summary["cuda_graph_paths"] == {"decode.replay": 1}


def test_dsv4_index_cache_profile_driver_requires_eagle_off_for_real_runs():
    args = Namespace(
        dry_run=False,
        eagle_off_confirmed=False,
        indexcache_profile_env_confirmed=True,
        prompt_tokens=128000,
        min_indexcache_prompt_tokens=75000,
    )

    with pytest.raises(SystemExit, match="--eagle-off-confirmed"):
        profile_driver.validate_args(args)


def test_dsv4_index_cache_profile_driver_allows_dry_run_without_eagle_confirmation():
    args = Namespace(
        dry_run=True,
        eagle_off_confirmed=False,
        indexcache_profile_env_confirmed=False,
        prompt_tokens=128000,
        min_indexcache_prompt_tokens=75000,
    )

    profile_driver.validate_args(args)


def test_dsv4_index_cache_profile_driver_rejects_short_prompt_profiles():
    args = Namespace(
        dry_run=True,
        eagle_off_confirmed=False,
        indexcache_profile_env_confirmed=False,
        prompt_tokens=20000,
        min_indexcache_prompt_tokens=75000,
    )

    with pytest.raises(SystemExit, match="--prompt-tokens"):
        profile_driver.validate_args(args)


def test_dsv4_index_cache_profile_driver_requires_marker_env_for_real_runs():
    args = Namespace(
        dry_run=False,
        eagle_off_confirmed=True,
        indexcache_profile_env_confirmed=False,
        prompt_tokens=128000,
        min_indexcache_prompt_tokens=75000,
    )

    with pytest.raises(SystemExit, match="--indexcache-profile-env-confirmed"):
        profile_driver.validate_args(args)


def test_dsv4_index_cache_profile_driver_rejects_missing_real_trace_summaries():
    args = Namespace(dry_run=False)

    with pytest.raises(RuntimeError, match="server-side profile directory"):
        profile_driver.validate_trace_summaries(args, [])


def test_dsv4_index_cache_profile_driver_rejects_missing_required_regions():
    args = Namespace(dry_run=False)
    summaries = [
        {
            "categories": {
                "csa_indexer": {},
                "core_attention_c4": {},
                "ffn_moe": {},
                "cuda_graph": {},
            }
        }
    ]

    with pytest.raises(RuntimeError, match="raw_to_page_translation"):
        profile_driver.validate_trace_summaries(args, summaries)


def test_dsv4_index_cache_profile_driver_rejects_missing_cuda_graph_paths():
    args = Namespace(dry_run=False)
    summaries = [
        {
            "categories": {
                "csa_indexer": {},
                "raw_to_page_translation": {},
                "core_attention_c4": {},
                "ffn_moe": {},
                "cuda_graph": {},
            },
            "cuda_graph_paths": {},
        }
    ]

    with pytest.raises(RuntimeError, match="replay/fallback path stats"):
        profile_driver.validate_trace_summaries(args, summaries)


def test_dsv4_index_cache_profile_driver_accepts_required_profile_regions():
    args = Namespace(dry_run=False)
    summaries = [
        {
            "categories": {
                "csa_indexer": {},
                "raw_to_page_translation": {},
                "core_attention_c4": {},
                "ffn_moe": {},
                "cuda_graph": {},
            },
            "cuda_graph_paths": {"decode.replay": 1},
        }
    ]

    profile_driver.validate_trace_summaries(args, summaries)


def test_dsv4_index_cache_profile_driver_accepts_server_info_without_spec_decode():
    server_info = {
        "speculative_algorithm": None,
        "speculative_num_steps": 0,
        "speculative_eagle_topk": None,
        "speculative_num_draft_tokens": 0,
    }

    assert profile_driver.validate_server_info_for_base_path(server_info) == []


def test_dsv4_index_cache_profile_driver_rejects_eagle_server_info():
    server_info = {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 3,
    }

    with pytest.raises(RuntimeError, match="speculative_algorithm=EAGLE"):
        profile_driver.validate_server_info_for_base_path(server_info)


def test_dsv4_index_cache_profile_driver_rejects_nested_spec_server_info():
    server_info = {
        "decode": [
            {
                "speculative_algorithm": None,
                "speculative_num_draft_tokens": "4",
            }
        ]
    }

    with pytest.raises(
        RuntimeError, match=r"decode\[0\].speculative_num_draft_tokens=4"
    ):
        profile_driver.validate_server_info_for_base_path(server_info)


def test_dsv4_index_cache_profile_driver_records_profile_dir_visibility_note():
    note = profile_driver.profile_dir_note(Path("/tmp/profiles"))

    assert "/tmp/profiles" in note
    assert "remote endpoints" in note

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


def test_dsv4_index_cache_profile_driver_rejects_missing_objective_buckets():
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
            "cuda_graph_paths": {"decode.indexcache_on.replay": 1},
            "objective_buckets": {
                "csa_indexer": {"count": 1},
                "raw_to_page_translation": {"count": 1},
                "sparse_core_attention": {"count": 1},
                "ffn_moe": {"count": 1},
            },
        }
    ]

    with pytest.raises(RuntimeError, match="objective timing buckets"):
        profile_driver.validate_trace_summaries(args, summaries)


def test_dsv4_index_cache_profile_driver_rejects_missing_indexcache_on_graph_path():
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

    with pytest.raises(RuntimeError, match="indexcache_on"):
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
            "cuda_graph_paths": {"decode.indexcache_on.replay": 1},
            "objective_buckets": {
                "csa_indexer": {"count": 1},
                "raw_to_page_translation": {"count": 1},
                "sparse_core_attention": {"count": 1},
                "ffn_moe": {"count": 1},
                "cuda_graph": {"count": 1},
            },
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


def test_dsv4_index_cache_profile_driver_dry_run_writes_manifest(tmp_path):
    output = tmp_path / "profile.json"

    profile_driver.main(
        [
            "--endpoint",
            "http://indexcache",
            "--profile-dir",
            str(tmp_path / "profiles"),
            "--prompt-tokens",
            "128000",
            "--min-indexcache-prompt-tokens",
            "75000",
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    result = json.loads(output.read_text())

    assert result["endpoint"] == "http://indexcache"
    assert result["min_indexcache_prompt_tokens"] == 75000
    assert result["request_results"] == []
    assert result["trace_files"] == []
    assert result["profile_validation"] == {
        "passed": True,
        "failures": [],
    }
    assert result["server_checks"] == {
        "server_info_checked": False,
        "speculative_decode": "dry run; /server_info not queried",
    }
    assert result["indexcache_profile_env"] == (
        "dry run; profiler marker env not exercised"
    )
    assert result["eagle"] == "dry run; speculative decoding not exercised"


def test_dsv4_index_cache_profile_driver_writes_manifest_before_validation_failure(
    tmp_path, monkeypatch
):
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    trace_path = profile_dir / "dsv4-indexcache-rank0.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "dsv4_indexcache.csa_indexer.layer_2", "dur": 1000},
                    {
                        "name": "dsv4_indexcache.cuda_graph.decode.indexcache_on.replay",
                        "dur": 0,
                    },
                ]
            }
        )
    )
    output = tmp_path / "profile.json"

    monkeypatch.setattr(
        profile_driver,
        "fetch_server_info",
        lambda *_: {
            "speculative_algorithm": None,
            "speculative_num_steps": 0,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": 0,
        },
    )
    monkeypatch.setattr(profile_driver, "collect_profile", lambda _: [])

    with pytest.raises(RuntimeError, match="raw_to_page_translation"):
        profile_driver.main(
            [
                "--endpoint",
                "http://indexcache",
                "--profile-dir",
                str(profile_dir),
                "--prompt-tokens",
                "128000",
                "--min-indexcache-prompt-tokens",
                "75000",
                "--output",
                str(output),
                "--eagle-off-confirmed",
                "--indexcache-profile-env-confirmed",
            ]
        )

    result = json.loads(output.read_text())

    assert result["profile_validation"]["passed"] is False
    assert any(
        "raw_to_page_translation" in failure
        for failure in result["profile_validation"]["failures"]
    )
    assert result["trace_files"] == [str(trace_path)]

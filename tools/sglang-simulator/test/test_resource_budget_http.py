"""A real resource SDK can size pools without selecting its latency provider."""

import importlib.util
import json
import re
import sys

import pytest
import requests
from test_simulation_sglang_serving import SGLangServingRunner


@pytest.mark.skipif(
    sys.platform == "win32"
    or importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires the Linux SGLang CPU serving environment",
)
def test_qwen_replay_uses_optional_resource_budget(tmp_path, monkeypatch):
    pytest.importorskip("simopus.sdk")
    import simopus

    revision = simopus.__revision__
    assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    monkeypatch.setenv("SGLANG_SIMULATOR_BOOTSTRAP", "1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    table = tmp_path / "replay.json"
    # This is an explicit replay fallback, not a claimed measured cost row.
    table.write_text("{}", encoding="utf-8")
    config = {
        "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
        "predictor": {
            "name": "replay",
            "database_path": str(table),
            "miss_fallback_seconds": 0.001,
            "miss_strategy": "zero",
        },
        "memory_budget": {
            "provider": "simopus",
            "provider_revision": revision,
            "model_id": "Qwen/Qwen3-8B",
            "model_revision": None,
            "policy": {
                "schema_version": 1,
                "mem_fraction_static": 0.8,
                "runtime_reserve_bytes": 2 * (1 << 30),
            },
        },
        "scheduler": {
            "tp_size": 2,
            "ep_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "cp_size": 1,
            "data_type": "BF16",
            "kv_cache_data_type": "BF16",
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    runner = SGLangServingRunner(
        path,
        tmp_path,
        max_total_tokens=None,
        extra_args=[
            "--skip-tokenizer-init",
            "--skip-server-warmup",
            "--page-size",
            "1",
        ],
    )
    try:
        requests.post(runner.base_url + "/start_profile", timeout=30).raise_for_status()
        response = requests.post(
            runner.base_url + "/generate",
            json={
                "input_ids": [7] * 128,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                    "custom_params": {
                        "simulation": {"total_request": 1, "created_time": 0}
                    },
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        assert result["meta_info"]["completion_tokens"] == 16, result
        assert result["meta_info"]["finish_reason"]["type"] == "length", result
        requests.post(runner.base_url + "/start_profile", timeout=30).raise_for_status()
        metrics = json.loads((runner.output_dir / "metrics.json").read_text())
        response = requests.get(runner.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        capacity = response.json()["max_total_num_tokens"]
    finally:
        runner.shutdown()

    assert metrics["completed"] == 1 and metrics["total_output"] == 16
    assert metrics["replay_exact_match_steps"] == 0
    assert metrics["replay_zero_fallback_steps"] == 16
    assert "simopus" not in metrics
    budget = metrics["memory_budget"]
    assert budget["provider"] == "simopus"
    assert budget["provider_revision"] == revision
    assert budget["model_id"] == "Qwen/Qwen3-8B"
    assert budget["topology"] == {
        "schema_version": 1,
        "tp_size": 2,
        "ep_size": 1,
        "pp_size": 1,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", budget["resource_digest"])
    assert capacity == budget["max_total_tokens"] > 0
    assert budget["policy"] == config["memory_budget"]["policy"]
    assert budget["provenance"]["simopus_revision"] == revision
    assert budget["provenance"]["requested_revision"] is None

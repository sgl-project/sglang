"""Opt-in HTTP E2E with a real InferCast provider and matching measured data."""

import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def provider_inputs():
    systems = os.environ.get("INFERCAST_E2E_SYSTEMS_ROOT")
    model = os.environ.get("INFERCAST_E2E_MODEL_PATH")
    if not systems or not model:
        pytest.skip("set INFERCAST_E2E_SYSTEMS_ROOT and INFERCAST_E2E_MODEL_PATH")
    import infercast

    revision = infercast.__revision__
    assert isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    assert Path(systems).is_dir()
    assert (Path(model) / "config.json").is_file()
    return str(Path(systems).resolve()), str(Path(model).resolve()), revision


@contextmanager
def simulator(
    inputs,
    root,
    *,
    chunked=False,
    mode="OFFLINE",
    mixed=False,
    automatic_budget=False,
    config_override=None,
):
    systems, model, revision = inputs
    config = (
        deepcopy(config_override)
        if config_override is not None
        else json.loads(
            (
                ROOT / "examples/sim_configs/infercast_deepseek_r1_mi355x_eager.json"
            ).read_text(encoding="utf-8")
        )
    )
    if config["predictor"]["name"] == "infercast":
        config["predictor"].update(systems_root=systems, provider_revision=revision)
    if automatic_budget:
        identity = config.get("memory_budget", config["predictor"])
        config["memory_budget"] = {
            "provider": "infercast",
            "provider_revision": revision,
            "model_id": identity["model_id"],
            "model_revision": identity.get("model_revision"),
            "policy": {
                "schema_version": 1,
                "mem_fraction_static": 0.8,
                "runtime_reserve_bytes": 2 * (1 << 30),
            },
        }
    else:
        config.pop("memory_budget", None)
    config_path = root / "simulator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = root / "output"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        SGLANG_USE_CPU_ENGINE="1",
        SGLANG_SIMULATOR_BOOTSTRAP="1",
        SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
        SGLANG_SIMULATOR_OUTPUT_MODE=mode,
        SGLANG_SIMULATOR_OUTPUT_DIR=str(output),
    )
    command = [
        sys.executable,
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        model,
        "--sim-config-path",
        str(config_path),
        "--port",
        str(port),
        "--skip-tokenizer-init",
        "--skip-server-warmup",
        "--max-prefill-tokens",
        "65536" if mixed else "4096",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "65536" if mixed else "64" if chunked else "4096",
        "--page-size",
        "1",
    ]
    if not automatic_budget:
        command += ["--max-total-tokens", "65536"]
    if mixed:
        command += ["--enable-mixed-chunk", "--schedule-policy", "fcfs"]
    url = f"http://127.0.0.1:{port}"
    with (root / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(120):
                assert proc.poll() is None, (root / "server.log").read_text()[-8000:]
                try:
                    if requests.get(url + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                pytest.fail("simulator did not become ready")
            requests.post(url + "/start_profile", timeout=30).raise_for_status()
            yield url, output, config
        except requests.RequestException as error:
            context = (root / "server.log").read_text(
                encoding="utf-8", errors="replace"
            )[-8000:]
            pytest.fail(f"{error}\nSimulator log:\n{context}")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


@pytest.mark.parametrize("automatic_budget", [False, True])
def test_simulator_uses_generated_config_with_ambient_config(
    tmp_path, monkeypatch, automatic_budget
):
    ambient = tmp_path / "ambient.json"
    ambient.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(ambient))
    captured = {}

    def spawn(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=1, poll=lambda: None, wait=lambda **kwargs: 0)

    response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(os, "killpg", lambda *_args: None, raising=False)

    with simulator(
        (str(tmp_path), str(tmp_path), "a" * 40),
        tmp_path,
        automatic_budget=automatic_budget,
    ) as (_, _, config):
        assert captured["env"]["SGLANG_SIMULATOR_CONFIG_PATH"] == str(
            tmp_path / "simulator.json"
        )
        assert os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] == str(ambient)
        if automatic_budget:
            assert "--max-total-tokens" not in captured["command"]
            assert config["memory_budget"] == {
                "provider": "infercast",
                "provider_revision": "a" * 40,
                "model_id": config["predictor"]["model_id"],
                "model_revision": config["predictor"]["model_revision"],
                "policy": {
                    "schema_version": 1,
                    "mem_fraction_static": 0.8,
                    "runtime_reserve_bytes": 2 * (1 << 30),
                },
            }
        else:
            limit = captured["command"].index("--max-total-tokens")
            assert captured["command"][limit + 1] == "65536"
            assert "memory_budget" not in config


def generate(url, prompts, *, output=16, total=None, created_times=None):
    lengths = [output] * len(prompts) if isinstance(output, int) else output
    arrivals = [0] * len(prompts) if created_times is None else created_times
    assert len(lengths) == len(arrivals) == len(prompts)

    def send(request):
        prompt, length, created_time = request
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {
                    "ignore_eos": True,
                    "max_new_tokens": length,
                    "custom_params": {
                        "simulation": {
                            "total_request": total or len(prompts),
                            "created_time": created_time,
                        }
                    },
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        assert isinstance(result, dict), result
        assert "error" not in result, result
        assert result["meta_info"]["finish_reason"]["type"] == "length", result
        assert result["meta_info"]["completion_tokens"] == length, result

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        list(pool.map(send, zip(prompts, lengths, arrivals)))


def read_profile(url, output, inputs, config, *, count, output_length=16):
    lengths = (
        [output_length] * count if isinstance(output_length, int) else output_length
    )
    assert len(lengths) == count
    requests.post(url + "/start_profile", timeout=30).raise_for_status()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["completed"] == count
    assert metrics["total_output"] == sum(lengths)
    for key in ("mean_ttft_ms", "mean_tpot_ms", "duration"):
        assert math.isfinite(metrics[key]) and metrics[key] > 0
    provenance = metrics["infercast"]
    assert provenance["contract"] == "infercast-forward-latency"
    assert provenance["provider_revision"] == inputs[2]
    for key in ("execution_profile", "decode_execution_profile"):
        assert provenance[key] == config["predictor"][key]
    iterations = [
        json.loads(line)
        for line in (output / "iteration.jsonl").read_text().splitlines()
    ]
    assert len(iterations) == provenance["calls"]["total"]
    for item in iterations:
        assert math.isfinite(item["predicted_forward_latency"])
        assert item["predicted_forward_latency"] > 0
    stats = [
        json.loads(line) for line in (output / "request.jsonl").read_text().splitlines()
    ]
    assert len(stats) == count
    for item, length in zip(stats, lengths):
        assert len(item["gen_token_latencies"]) == length
        assert all(
            math.isfinite(value) and value >= 0 for value in item["gen_token_latencies"]
        )
        assert item["last_event_time"] >= item["created_time"]
    return metrics, iterations, stats


@pytest.mark.parametrize("case", ["cold", "chunk", "prefix", "automatic_budget"])
def test_infercast_scheduler_forwards(provider_inputs, tmp_path, case, monkeypatch):
    commands = []
    if case == "automatic_budget":
        original_spawn = subprocess.Popen

        def spawn(command, **kwargs):
            commands.append(command)
            return original_spawn(command, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", spawn)
    with simulator(
        provider_inputs,
        tmp_path,
        chunked=case == "chunk",
        automatic_budget=case == "automatic_budget",
    ) as (
        url,
        output,
        config,
    ):
        if case == "prefix":
            generate(url, [[7] * 64], output=1)
            requests.post(url + "/start_profile", timeout=30).raise_for_status()
        length = 65 if case == "prefix" else 128
        generate(url, [[7] * length])
        metrics, iterations, _ = read_profile(
            url, output, provider_inputs, config, count=1
        )
        if case == "automatic_budget":
            response = requests.get(url + "/server_info", timeout=30)
            response.raise_for_status()
            assert (
                response.json()["max_total_num_tokens"]
                == metrics["memory_budget"]["max_total_tokens"]
            )
    assert metrics["infercast"]["calls"]["context"] == (2 if case == "chunk" else 1)
    assert metrics["infercast"]["calls"]["decode"] == 15
    assert metrics["prefix_cache_reused_ratio"] == pytest.approx(
        64 / length if case == "prefix" else 0
    )
    if case == "automatic_budget":
        assert len(commands) == 1
        assert "--max-total-tokens" not in commands[0]
        budget = metrics["memory_budget"]
        assert budget["provider"] == "infercast"
        assert budget["provider_revision"] == provider_inputs[2]
        assert budget["model_id"] == config["predictor"]["model_id"]
        assert re.fullmatch(r"[0-9a-f]{64}", budget["resource_digest"])
        assert budget["topology"] == {
            "schema_version": 1,
            "tp_size": 8,
            "ep_size": 2,
            "pp_size": 1,
        }
        assert budget["policy"] == config["memory_budget"]["policy"]
        assert type(budget["max_total_tokens"]) is int
        assert budget["max_total_tokens"] > 0
        assert len(budget["ranks"]) == 8
        provenance = budget["provenance"]
        assert provenance["infercast_revision"] == provider_inputs[2]
        assert provenance["requested_revision"] == config["predictor"]["model_revision"]
        assert (
            provenance["config_sha256"]
            == hashlib.sha256(
                (Path(provider_inputs[1]) / "config.json").read_bytes()
            ).hexdigest()
        )
    if case == "chunk":
        assert any(
            extend > 1 and past > 0
            for item in iterations
            for extend, past in item["requests"]
        )


def test_infercast_ragged_mixed_batch(provider_inputs, tmp_path):
    with simulator(provider_inputs, tmp_path, mixed=True) as (
        url,
        output,
        config,
    ):
        # Prime the exact prefixes of a measured ordered ragged context row.
        for token, prefix in ((7, 63), (8, 575), (9, 2175)):
            generate(url, [[token] * prefix], output=1)
            requests.post(url + "/start_profile", timeout=30).raise_for_status()
        # Only B arrives at zero. Its first forward releases A1/A2/A3, which
        # combine with B's next decode token in the measured MIXED vector.
        lengths = [2, 1, 1, 1]
        generate(
            url,
            [[10] * 8383, [7] * 71, [8] * 579, [9] * 2178],
            output=lengths,
            created_times=[0, 1e-6, 2e-6, 3e-6],
        )
        metrics, iterations, _ = read_profile(
            url, output, provider_inputs, config, count=4, output_length=lengths
        )
    assert metrics["infercast"]["calls"]["mixed"] > 0
    assert any(
        item["forward_mode"] == "MIXED"
        and item["requests"] == [[8, 63], [4, 575], [3, 2175], [1, 8383]]
        for item in iterations
    )


def test_infercast_blocking_successive_arrivals(provider_inputs, tmp_path):
    with simulator(provider_inputs, tmp_path, mode="BLOCKING") as (url, output, config):
        generate(url, [[7] * 128], total=2)
        # The next arrival is wall-clock based, after CPU/provider work for the first.
        generate(url, [[8] * 128], total=2)
        _, _, stats = read_profile(url, output, provider_inputs, config, count=2)
    assert stats[1]["created_time"] > stats[0]["created_time"]


def test_generic_qwen_replay_with_infercast_budget(provider_inputs, tmp_path):
    # The empty table uses the existing explicit fallback, without measured data.
    table = tmp_path / "empty-replay.json"
    table.write_text("{}", encoding="utf-8")
    model = ROOT / "test/assets/qwen3-8b"
    config = {
        "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
        "predictor": {
            "name": "replay",
            "database_path": str(table),
            "miss_fallback_seconds": 0.001,
            "miss_strategy": "zero",
        },
        "memory_budget": {"model_id": "Qwen/Qwen3-8B", "model_revision": None},
        "scheduler": {
            "tp_size": 2,
            "ep_size": 1,
            "pp_size": 1,
            "dp_size": 1,
            "cp_size": 1,
            "data_type": "BF16",
            "kv_cache_data_type": "BF16",
            "backend_name": "sglang",
            "backend_version": "0.5.17",
        },
    }
    inputs = provider_inputs[0], str(model), provider_inputs[2]
    with simulator(inputs, tmp_path, automatic_budget=True, config_override=config) as (
        url,
        output,
        _,
    ):
        generate(url, [[7] * 128])
        requests.post(url + "/start_profile", timeout=30).raise_for_status()
        metrics = json.loads((output / "metrics.json").read_text())
        response = requests.get(url + "/server_info", timeout=30)
        response.raise_for_status()
        native_tokens = response.json()["max_total_num_tokens"]
    assert metrics["completed"] == 1 and metrics["total_output"] == 16
    assert metrics["replay_exact_match_steps"] == 0
    assert metrics["replay_zero_fallback_steps"] == 16
    assert "infercast" not in metrics
    budget = metrics["memory_budget"]
    assert budget["provider"] == "infercast"
    assert budget["provider_revision"] == provider_inputs[2]
    assert budget["model_id"] == "Qwen/Qwen3-8B"
    assert budget["topology"] == {
        "schema_version": 1,
        "tp_size": 2,
        "ep_size": 1,
        "pp_size": 1,
    }
    assert native_tokens == budget["max_total_tokens"] > 0
    assert budget["provenance"]["infercast_revision"] == provider_inputs[2]
    assert budget["provenance"]["requested_revision"] is None

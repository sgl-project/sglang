"""Opt-in HTTP E2E with a real InferCast provider and matching measured data."""

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
from pathlib import Path

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
def simulator(inputs, root, *, chunked=False, mode="OFFLINE", mixed=False):
    systems, model, revision = inputs
    config = json.loads(
        (
            ROOT / "examples/sim_configs/infercast_deepseek_r1_mi355x_eager.json"
        ).read_text(encoding="utf-8")
    )
    config["predictor"].update(systems_root=systems, provider_revision=revision)
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
        "--max-total-tokens",
        "65536",
        "--max-prefill-tokens",
        "65536" if mixed else "4096",
        "--max-running-requests",
        "8",
        "--chunked-prefill-size",
        "65536" if mixed else "64" if chunked else "4096",
        "--page-size",
        "1",
    ]
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


@pytest.mark.parametrize("case", ["cold", "chunk", "prefix"])
def test_infercast_scheduler_forwards(provider_inputs, tmp_path, case):
    with simulator(provider_inputs, tmp_path, chunked=case == "chunk") as (
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
    assert metrics["infercast"]["calls"]["context"] == (2 if case == "chunk" else 1)
    assert metrics["infercast"]["calls"]["decode"] == 15
    assert metrics["prefix_cache_reused_ratio"] == pytest.approx(
        64 / length if case == "prefix" else 0
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

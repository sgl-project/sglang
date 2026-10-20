"""GPU example: start collection, generate responses, export the traces."""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture
def collection_server(tmp_path):
    simulator_root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model"
    model.mkdir()
    config = json.loads(
        (simulator_root / "test/assets/qwen3-8b/config.json").read_text()
    )
    # Small dummy model: this tests lifecycle and data contracts, not accuracy.
    config.update(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_window_layers=2,
        vocab_size=1024,
        bos_token_id=1,
        eos_token_id=2,
    )
    (model / "config.json").write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    output = tmp_path / "baseline"
    command = [
        "sglang-simulator-collect",
        f"--collection-output-dir={output}",
        f"--model-path={model}",
        "--load-format=dummy",
        "--skip-tokenizer-init",
        "--disable-overlap-schedule",
        "--device=cuda",
        "--cuda-graph-backend-decode=disabled",
        "--cuda-graph-backend-prefill=disabled",
        "--chunked-prefill-size=128",
        "--max-total-tokens=4096",
        "--mem-fraction-static=0.1",
        "--host=127.0.0.1",
        f"--port={port}",
    ]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                assert server.poll() is None, log_path.read_text()
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise AssertionError(log_path.read_text())

            yield url, output, log_path
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def test_baseline_collection(collection_server):
    url, output, log_path = collection_server

    def post(path, payload=None):
        response = requests.post(url + path, json=payload or {}, timeout=60)
        assert response.ok, response.text + "\n" + log_path.read_text()

    post("/start_profile")
    for inputs in ([10] * 256, [20] * 192, [[30] * 128, [40] * 200]):
        post(
            "/generate",
            {
                "input_ids": inputs,
                "sampling_params": {"max_new_tokens": 4, "ignore_eos": True},
            },
        )
    post("/stop_profile")

    def read(kind):
        return [
            json.loads(line)
            for line in (output / f"TP0.{kind}.jsonl").read_text().splitlines()
        ]

    requests_trace, raw, steps = (
        read("request"),
        read("raw_request"),
        read("schedule_batch"),
    )
    request_ids = {r["rid"] for r in raw}
    assert len(requests_trace) == len(raw) == len(request_ids) == 4
    assert sorted(r["input_length"] for r in requests_trace) == [128, 192, 200, 256]
    assert all(r["output_length"] == 4 for r in requests_trace)
    assert all(r["completed"] and len(r["output_ids"]) == 4 for r in raw)
    assert {r["rid"] for r in requests_trace} == request_ids
    assert {r["rid"] for step in steps for r in step["request_infos"]} == request_ids
    assert {step["forward_mode"] for step in steps} == {1, 2}  # Prefill and decode.
    assert all(step["iter_latency"] > 0 for step in steps)

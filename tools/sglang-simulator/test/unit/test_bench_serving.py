import argparse
import asyncio
import io
import json
from dataclasses import fields

import aiohttp
from sglang_simulator.simulation import bench_serving

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow


async def _collect(generator):
    return [request async for request in generator]


def _row(timestamp=None, extra_request_body=None):
    return DatasetRow(
        prompt=[1, 2, 3],
        prompt_len=3,
        output_len=7,
        timestamp=timestamp,
        extra_request_body=extra_request_body,
    )


def test_offline_trace_injects_normalized_metadata_without_mutating_contract():
    rows = [_row(1200.0, {"rid": "a"}), _row(1000.0, {"rid": "b"})]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = False

    result = asyncio.run(
        _collect(
            bench_serving.simulator_get_request(
                rows,
                request_rate=4,
                use_trace_timestamps=True,
                slowdown_factor=2,
            )
        )
    )

    assert [row.timestamp for row in result] == [1000.0, 1200.0]
    assert result[0].extra_request_body == {
        "rid": "b",
        "simulation": {"created_time_ms": 0.0, "total_request": 2},
    }
    assert result[1].extra_request_body["simulation"] == {
        "created_time_ms": 400.0,
        "total_request": 2,
    }


def test_offline_request_rate_generates_logical_time_without_sleep(monkeypatch):
    rows = [_row(), _row(), _row()]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = False
    monkeypatch.setattr(bench_serving.np.random, "exponential", lambda _: 0.25)

    result = asyncio.run(
        _collect(bench_serving.simulator_get_request(rows, request_rate=4))
    )

    assert [
        row.extra_request_body["simulation"]["created_time_ms"] for row in result
    ] == [0.0, 250.0, 500.0]
    assert all(
        row.extra_request_body["simulation"]["total_request"] == 3 for row in result
    )


def test_v0516_cli_trace_flag_survives_missing_benchmark_forwarding():
    rows = [_row(1000.0), _row(1250.0)]
    bench_serving._SIMULATOR_MODE = "offline"
    bench_serving._USE_TRACE_TIMESTAMPS = True

    result = asyncio.run(
        _collect(
            bench_serving.simulator_get_request(
                rows,
                request_rate=float("inf"),
                use_trace_timestamps=False,
            )
        )
    )

    assert [
        row.extra_request_body["simulation"]["created_time_ms"] for row in result
    ] == [0.0, 250.0]


def test_json_hijack_merges_metadata_into_existing_sampling_params(monkeypatch):
    captured = {}

    async def original_request(self, method, url, **kwargs):
        captured.update(kwargs["json"])
        return "response"

    monkeypatch.setattr(aiohttp.ClientSession, "_request", original_request)
    monkeypatch.setattr(bench_serving, "_ORIGINAL_AIOHTTP_REQUEST", None)
    bench_serving.install_aiohttp_json_hijack()
    payload = {
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 7, "temperature": 0},
        "simulation": {"created_time_ms": 10, "total_request": 1},
    }

    result = asyncio.run(
        aiohttp.ClientSession._request(
            object(), "POST", "http://127.0.0.1:30000/generate", json=payload
        )
    )

    assert result == "response"
    assert "simulation" not in captured
    assert captured["sampling_params"]["max_new_tokens"] == 7
    assert captured["sampling_params"]["custom_params"]["simulation"] == {
        "created_time_ms": 10,
        "total_request": 1,
    }


def test_simulator_cli_argument_is_removed_before_official_parser():
    mode, remaining = bench_serving._extract_simulator_args(
        ["--backend", "sglang", "--simulator-mode", "blocking", "--num-prompts", "2"]
    )
    assert mode == "blocking"
    assert remaining == ["--backend", "sglang", "--num-prompts", "2"]


def test_simulator_argument_parser_adds_autobench_to_hard_coded_choices():
    parser_class = bench_serving._simulator_argument_parser(argparse.ArgumentParser)
    parser = parser_class()
    parser.add_argument(
        "--dataset-name",
        choices=["sharegpt", "random"],
        default="sharegpt",
    )

    args = parser.parse_args(["--dataset-name", "autobench"])
    assert args.dataset_name == "autobench"


def _client_metrics(value=123):
    return serving.BenchmarkMetrics(
        **{field.name: value for field in fields(serving.BenchmarkMetrics)}
    )


def test_missing_backend_fields_are_not_filled_from_offline_client(
    tmp_path, monkeypatch
):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"completed": 50, "p90_ttft_ms": 400})
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(
        bench_serving,
        "_ORIGINAL_CALCULATE_METRICS",
        lambda *args, **kwargs: (_client_metrics(), []),
    )

    metrics, _ = bench_serving.simulator_calculate_metrics()

    assert metrics.completed == 50
    assert metrics.p90_ttft_ms == 400
    assert metrics.p95_ttft_ms == -1
    assert metrics.max_concurrent_requests == -1
    assert metrics.total_input_text == -1
    assert metrics.total_input_vision == -1
    assert metrics.total_output_retokenized == -1


def test_duration_stream_uses_simulated_duration(tmp_path, monkeypatch):
    (tmp_path / "metrics.json").write_text(json.dumps({"duration": 10.7124}))
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))
    target = io.StringIO()
    stream = bench_serving._DurationReplacingStream(target)

    stream.write("Benchmark duration (s):                  1.41      ")

    assert target.getvalue() == "Benchmark duration (s):                  10.71     "


def test_run_benchmark_replaces_return_and_output_file_duration(
    tmp_path, monkeypatch, capsys
):
    output_file = tmp_path / "client.jsonl"
    (tmp_path / "metrics.json").write_text(json.dumps({"duration": 10.7124}))
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path))

    def fake_run(_):
        print("Benchmark duration (s):                  1.41      ")
        output_file.write_text(json.dumps({"duration": 1.41, "completed": 50}) + "\n")
        return {"duration": 1.41, "completed": 50}

    monkeypatch.setattr(bench_serving, "_ORIGINAL_RUN_BENCHMARK", fake_run)
    result = bench_serving.simulator_run_benchmark(
        argparse.Namespace(
            backend="sglang",
            dataset_name="sharegpt",
            use_trace_timestamps=False,
            output_file=str(output_file),
        )
    )

    assert "10.71" in capsys.readouterr().out
    assert result["duration"] == 10.7124
    assert json.loads(output_file.read_text())["duration"] == 10.7124

"""Real CPU simulator integration; these are mechanism tests, not measured GPU GT."""

import json
import os
import subprocess
import sys
from pathlib import Path

from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"


def _config(tmp_path):
    config = {
        "platform": {"accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80}},
        "predictor": {
            "name": "aisimulate",
            "estimation_mode": "op_level",
            "database_mode": "SOL",
            "estimator_config": {"correction": {"enabled": False}},
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "backend_version": "0.5.10"},
    }
    path = tmp_path / "sim.json"
    path.write_text(json.dumps(config))
    return path


def test_native_runner_mixed_and_warm_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(_config(tmp_path)))
    # bench_runner resolves its output paths at import time.
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path / "output"))
    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            skip_tokenizer_init=True,
            max_total_tokens=4096,
            page_size=1,
            chunked_prefill_size=128,
            enable_mixed_chunk=True,
        )
    )
    dataset = SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[1000 + i] * 129,
                input_length=129,
                output_length=16,
                custom_params={"created_time": t},
            )
            for i, t in enumerate([0, 0.001, 0.02, 0.022])
        ]
    )
    try:
        cold = runner.benchmark(
            BenchmarkConfig(request_rate=10, ignore_request_timestamp=False),
            dataset=dataset,
        )
        cold_steps = runner.get_iteration_stats()
        warm = runner.benchmark(
            BenchmarkConfig(request_rate=10, ignore_request_timestamp=False),
            dataset=dataset,
        )
        warm_steps = runner.get_iteration_stats()
    finally:
        runner.shutdown()
    assert cold["completed"] == warm["completed"] == 4
    assert cold["total_output"] == warm["total_output"] == 64
    assert warm["kv_cache_device_hit_ratio"] > 0.95
    assert any(step["forward_mode"] == "mixed" for step in cold_steps)
    assert any(
        step["forward_mode"] == "extend" and [1, 128] in step["requests"]
        for step in warm_steps
    )


def test_native_auto_capacity_startup(tmp_path):
    env = os.environ.copy()
    env["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_config(tmp_path))
    env["SGLANG_SIMULATOR_OUTPUT_DIR"] = str(tmp_path / "output")
    code = f"""from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
from sglang.srt.server_args import ServerArgs
runner=SGLangBenchmarkRunner(server_args=ServerArgs(model_path={str(ASSETS / "qwen3-8b")!r},load_format="dummy",device="cpu",skip_tokenizer_init=True))
runner.shutdown()
print("AUTO_CAPACITY_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AUTO_CAPACITY_OK" in result.stdout

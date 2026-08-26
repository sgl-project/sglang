import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
BENCH_SERVING = SGLANG_ROOT / "benchmark" / "simulator" / "bench_serving.py"
EXAMPLES = Path(__file__).parent.parent / "examples"
SIM_CONFIGS = {
    "aic_sol": EXAMPLES / "sim_configs" / "aic_sol.json",
    "aic_silicon": EXAMPLES / "sim_configs" / "aic_silicon.json",
    "ml": EXAMPLES / "sim_configs" / "ml.json",
    "replay": EXAMPLES / "sim_configs" / "replay.json",
}


class SGLangServingRunner:
    def __init__(self, config_path: Path, tmp_path: Path, mode: str = "offline"):
        self.mode = mode
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

        self.output_dir = tmp_path / "output"
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            SGLANG_USE_CPU_ENGINE="1",
            SGLANG_SIMULATOR_CONFIG_PATH=str(config_path),
            SGLANG_SIMULATOR_OUTPUT_MODE=mode.upper(),
            SGLANG_SIMULATOR_OUTPUT_DIR=str(self.output_dir),
        )
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.sglang.launch_server",
            "--model-path",
            str(ASSETS / "qwen3-8b"),
            "--sim-config-path",
            str(config_path),
            "--port",
            str(self.port),
            "--tokenizer-path",
            str(EXAMPLES / "assets" / "tokenizer"),
            "--max-total-tokens",
            "8192",
            "--max-running-requests",
            "8",
            "--disable-overlap-schedule",
        ]
        self.server_proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
        for _ in range(120):
            if self.server_proc.poll() is not None:
                raise RuntimeError("SGLang Simulator server exited during startup")
            try:
                if requests.get(self.base_url, timeout=1).status_code < 500:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.shutdown()
        raise RuntimeError("SGLang Simulator server did not become ready")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def benchmark(
        self,
        output_file: Path,
        workload: str = "sharegpt",
        request_rate=None,
        seed=42,
    ) -> dict:
        cmd = [
            sys.executable,
            str(BENCH_SERVING),
            f"--simulator-mode={self.mode}",
            "--backend=sglang",
            f"--base-url={self.base_url}",
            f"--model={ASSETS / 'qwen3-8b'}",
            f"--tokenizer={EXAMPLES / 'assets' / 'tokenizer'}",
            "--num-prompts=3",
            "--disable-tqdm",
            "--profile",
            f"--output-file={output_file}",
        ]
        if request_rate is not None:
            cmd.extend([f"--request-rate={request_rate}", f"--seed={seed}"])

        if workload == "sharegpt":
            cmd.extend(
                [
                    "--dataset-name=sharegpt",
                    f"--dataset-path={EXAMPLES / 'workloads' / 'sharegpt-example.json'}",
                    "--sharegpt-output-len=4",
                ]
            )
        else:
            assert workload == "timestamp_trace"
            cmd.extend(
                [
                    "--dataset-name=autobench",
                    f"--dataset-path={EXAMPLES / 'workloads' / 'timestamp-trace-example.jsonl'}",
                    "--use-trace-timestamps",
                ]
            )

        subprocess.run(cmd, check=True)
        assert output_file.is_file()
        return json.loads(
            (self.output_dir / "metrics.json").read_text(encoding="utf-8")
        )

    def shutdown(self):
        if self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()


def assert_decode_metrics(metrics):
    assert metrics["completed"] == 3
    assert metrics["total_output"] == 12
    assert metrics["mean_ttft_ms"] >= 0
    assert metrics["mean_tpot_ms"] > 0
    assert metrics["mean_itl_ms"] > 0
    assert metrics["input_throughput"] > 0


@pytest.mark.parametrize("config_name", SIM_CONFIGS)
def test_benchmark(config_name, tmp_path):
    runner = SGLangServingRunner(SIM_CONFIGS[config_name], tmp_path)
    try:
        metrics = runner.benchmark(tmp_path / "benchmark.json")
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)


def test_timestamp_trace(tmp_path):
    runner = SGLangServingRunner(SIM_CONFIGS["replay"], tmp_path)
    try:
        metrics = runner.benchmark(
            tmp_path / "benchmark.json", workload="timestamp_trace"
        )
    finally:
        runner.shutdown()

    assert_decode_metrics(metrics)

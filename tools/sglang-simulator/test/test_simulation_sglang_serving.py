import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

ASSETS = Path(__file__).parent / "assets"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
os.environ["SGLANG_SIMULATOR_OUTPUT_MODE"] = "OFFLINE"


class SGLangServingRunner:
    def __init__(self, server_args: dict):
        cmd = [sys.executable, "-m", "sglang_simulator.simulation.sglang.launch_server"]

        for key, value in server_args.items():
            flag = "--" + key.replace("_", "-")
            if value is True:
                cmd.append(flag)
            elif value is not False:
                cmd.extend([flag, str(value)])

        self.server_proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        duration = 0
        while duration < 120:
            try:
                response = requests.get(url="http://localhost:30000")
                if response.status_code < 500:
                    return
            except Exception:
                pass
            time.sleep(1)
            duration += 1
        raise RuntimeError("Fail to start llm server.")

    def benchmark(self):
        start = time.time()
        output_file = "/tmp/sglang_simulator_serving_benchmark.json"
        cmd = [
            sys.executable,
            "-m",
            "sglang_simulator.simulation.bench_serving",
            "--simulator-mode=offline",
            "--backend=sglang",
            "--base-url=http://127.0.0.1:30000",
            "--warmup-requests=0",
            f"--model={ASSETS / 'qwen3-8b'}",
            "--dataset-name=autobench",
            f"--dataset-path={ASSETS / 'autobench.jsonl'}",
            "--use-trace-timestamps",
            "--num-prompts=3",
            "--disable-tqdm",
            "--profile",
            f"--output-file={output_file}",
        ]
        subprocess.run(cmd, check=True)

        with open(output_file) as file:
            metrics = json.load(file)

        with open(output_file, "w"):
            pass

        metrics["time_cost"] = time.time() - start
        return metrics

    def flush_cache(self):
        requests.get(url="http://localhost:30000/flush_cache")

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None


def test_benchmark():
    runner = SGLangServingRunner(
        server_args={
            "model_path": ASSETS / "qwen3-8b",
            "sim_config_path": ASSETS / "config.json",
            "skip_tokenizer_init": True,
            "max_total_tokens": 8192,
            "max_running_requests": 8,
            "disable_overlap_schedule": True,
        }
    )

    try:
        metrics = runner.benchmark()
    finally:
        runner.shutdown()

    assert metrics["completed"] == 3


if __name__ == "__main__":
    test_benchmark()

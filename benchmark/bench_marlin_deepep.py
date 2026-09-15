"""Compare actual Marlin HTTP serving with none and DeepEP communication.

Run from the repository root with PYTHONPATH=python and a local, pinned AWQ
checkpoint. Servers run sequentially on the same GPUs. Results include every
repetition; the script does not assume that DeepEP is faster.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def server_command(args, backend):
    world = len(args.gpus.split(","))
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--quantization",
        "awq_marlin",
        "--dtype",
        "bfloat16",
        "--tp-size",
        str(world),
        "--ep-size",
        str(world),
        "--moe-runner-backend",
        "marlin",
        "--moe-a2a-backend",
        backend,
        "--deepep-mode",
        "auto",
        "--attention-backend",
        "triton",
        "--disable-shared-experts-fusion",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "--max-running-requests",
        "128",
        "--max-total-tokens",
        "65536",
        "--chunked-prefill-size",
        "2048",
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.75",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.dp_attention:
        command += ["--enable-dp-attention", "--dp-size", str(world)]
    return command


def wait_ready(server, base_url):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Server exited before readiness; see server log.")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Server did not become ready within 600 seconds.")


def benchmark(args, backend, env):
    output = args.results / f"{backend}.jsonl"
    if output.exists():
        raise FileExistsError(f"Use a new results directory: {output} exists.")
    for concurrency in args.concurrency:
        for repeat in range(args.repetitions):
            command = [
                sys.executable,
                "-m",
                "sglang.benchmark.serving",
                "--backend",
                "sglang",
                "--base-url",
                f"http://127.0.0.1:{args.port}",
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.input_len),
                "--random-output-len",
                str(args.output_len),
                "--random-range-ratio",
                "1",
                "--num-prompts",
                str(concurrency * args.waves),
                "--max-concurrency",
                str(concurrency),
                "--warmup-requests",
                "8",
                "--seed",
                "42",
                "--output-file",
                str(output),
                "--disable-tqdm",
            ]
            log = args.results / f"{backend}-c{concurrency}-r{repeat}.log"
            with log.open("w") as stream:
                subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1200,
                )
            result = json.loads(output.read_text().splitlines()[-1])
            if result["completed"] != concurrency * args.waves:
                raise RuntimeError(f"Incomplete requests in {log}")
            if result["total_output_tokens"] != result["completed"] * args.output_len:
                raise RuntimeError(f"Unexpected generation lengths in {log}")
            summary = {
                key: result[key]
                for key in (
                    "max_concurrency",
                    "completed",
                    "duration",
                    "output_throughput",
                    "total_throughput",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                )
            }
            print(
                json.dumps({"backend": backend, "repeat": repeat, **summary}),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local pinned AWQ checkpoint")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--backends", nargs="+", choices=["none", "deepep"], default=["none", "deepep"]
    )
    parser.add_argument("--dp-attention", action="store_true")
    parser.add_argument(
        "--capacity",
        type=int,
        default=64,
        help="DeepEP per-rank decode capacity; must cover the per-rank batch limit",
    )
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if max(args.concurrency) > 128:
        parser.error("This benchmark configures a 128-request server limit.")
    args.results.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": str(args.capacity),
    }
    for backend in args.backends:
        command = server_command(args, backend)
        (args.results / f"{backend}-command.json").write_text(
            json.dumps(command, indent=2)
        )
        with (args.results / f"{backend}-server.log").open("w") as stream:
            server = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_ready(server, f"http://127.0.0.1:{args.port}")
                benchmark(args, backend, env)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGINT)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()


if __name__ == "__main__":
    main()

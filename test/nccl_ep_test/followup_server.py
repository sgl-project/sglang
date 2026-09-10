"""Run cost-ordered server gates; never provision or connect to a server.

PYTHONPATH=python:test python -m nccl_ep_test.followup_server env --reports /tmp/ep-gates
Then run single, pair, and serve with the same report directory.
"""

import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "gaunernst/DeepSeek-V2-Lite-Chat-FP8"
REVISION = "2f6d5dd458e5d9673719d03d30c488866be52e1a"


def source_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prerequisite(reports, name):
    record = json.loads((reports / name).read_text())
    if not record.get("passed") or record.get("source_head") != source_head():
        raise RuntimeError(f"Required successful gate for this commit: {name}")
    return record


def command(args, log, *, timeout=600, env=None):
    with log.open("w") as stream:
        process = subprocess.Popen(
            args,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, args)
        finally:
            stop_process_group(process)


def stop_process_group(process):
    # The launcher may have already exited while its GPU workers are still alive.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def environment():
    import torch

    def output(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=20)
            return {
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        except FileNotFoundError:
            return {"available": False}

    devices = [
        {
            "name": torch.cuda.get_device_name(i),
            "sm": list(torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ]
    return {
        "devices": devices,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": output(["nvidia-smi"]),
        "topology": output(["nvidia-smi", "topo", "-m"]),
        "toolkit": output(["nvcc", "--version"]),
        "p2p": (
            [
                torch.cuda.can_device_access_peer(0, 1),
                torch.cuda.can_device_access_peer(1, 0),
            ]
            if len(devices) >= 2
            else None
        ),
    }


def server_args(port, *, graph=True):
    args = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--revision",
        REVISION,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--moe-a2a-backend",
        "nccl_ep",
        "--moe-runner-backend",
        "triton",
        "--fp8-gemm-backend",
        "triton",
        "--attention-backend",
        "triton",
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--ep-size",
        "2",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--moe-dense-tp-size",
        "1",
        "--disable-shared-experts-fusion",
        "--disable-overlap-schedule",
        "--nccl-ep-mode",
        "low_latency",
        "--nccl-ep-num-max-dispatch-tokens-per-rank",
        "64",
        "--cuda-graph-bs-decode",
        "1",
        "8",
        "16",
        "32",
        "--chunked-prefill-size",
        "128",
        "--page-size",
        "1",
        "--context-length",
        "512",
        "--mem-fraction-static",
        "0.6",
        "--max-running-requests",
        "64",
        "--enable-metrics",
    ]
    if graph:
        args.append("--enable-nccl-ep-cuda-graph")
    return args


def serving_smoke(reports, port, *, graph):
    label = "graph" if graph else "eager"
    base = f"http://127.0.0.1:{port}"
    # Avoid accidentally directing tests at an unrelated existing server.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))

    def request(path, body=None, timeout=120):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode()

    responses = []
    with (reports / f"serve-{label}.log").open("w") as log:
        process = subprocess.Popen(
            server_args(port, graph=graph),
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect serve-{label}.log")
                try:
                    request("/health", timeout=2)
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Server startup exceeded 600 seconds")
                    time.sleep(1)
            info = json.loads(request("/server_info"))
            assert info["moe_a2a_backend"] == "nccl_ep"
            assert info["moe_runner_backend"] == "triton"
            assert info["disable_shared_experts_fusion"]
            assert info["enable_nccl_ep_cuda_graph"] == graph
            assert 0 < info["chunked_prefill_size"] <= 64
            save(reports / f"serve-{label}-info.json", info)

            def generate(index):
                started = time.perf_counter()
                body = {
                    "text": [
                        "What is 2 plus 2?",
                        "Write one short sentence about the moon.",
                        "Complete: one, two,",
                        "Name a color.",
                    ][index % 4],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": [8, 16, 4, 12][index % 4],
                    },
                    "return_logprob": True,
                }
                result = json.loads(request("/generate", body))
                assert isinstance(result["text"], str)
                meta = result["meta_info"]
                assert meta["completion_tokens"] > 0
                probabilities = meta.get("output_token_logprobs")
                assert probabilities and all(
                    math.isfinite(item[0]) for item in probabilities
                )
                return {
                    "request": body,
                    "result": result,
                    "wall_seconds": time.perf_counter() - started,
                }

            responses.extend(generate(i) for i in range(2))
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses.extend(pool.map(generate, range(2, 6)))
            responses.extend(generate(i) for i in range(6, 8))
            metrics = request("/metrics")
            (reports / f"serve-{label}-metrics.txt").write_text(metrics)
            mode = "decode_cuda_graph" if graph else "decode_none"
            passes = sum(
                float(match.group(1))
                for match in re.finditer(
                    r'^sglang:cuda_graph_passes_total\{[^\n}]*mode="'
                    + mode
                    + r'"[^\n}]*\}\s+([0-9.eE+-]+)',
                    metrics,
                    re.MULTILINE,
                )
            )
            assert passes > 0, f"No actual {mode} passes observed"
            return {"mode": label, "requests": responses, "decode_passes": passes}
        finally:
            stop_process_group(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("env", "single", "pair", "serve"))
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    result = {"phase": args.phase, "source_head": source_head(), "passed": False}
    target = reports / f"{args.phase}.json"
    save(target, result)
    try:
        if args.phase == "env":
            result.update(environment())
            from .environment import binding_check, prepare_jit

            result["bindings"] = binding_check()
            result["jit"] = prepare_jit()
        elif args.phase == "single":
            prerequisite(reports, "env.json")
            selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=selected)
            command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "test/registered/unit/layers/moe/test_nccl_ep_triton.py",
                    "test/registered/unit/layers/moe/test_nccl_ep_shared_experts.py",
                ],
                reports / "single-tests.log",
                env=env,
            )
            command(
                [
                    sys.executable,
                    "-m",
                    "nccl_ep_test.single_gpu",
                    "--require-sm",
                    "120",
                    "--experts",
                    "32",
                    "--capacity",
                    "128",  # Two ranks * 64 source tokens in the eager fallback.
                    "--report",
                    str(reports / "sm120-compute.json"),
                ],
                reports / "single-compute.log",
                env=env,
            )
            result["compute"] = json.loads((reports / "sm120-compute.json").read_text())
        elif args.phase == "pair":
            prerequisite(reports, "single.json")
            env_check = environment()
            if (
                len(env_check["devices"]) != 2
                or env_check["p2p"] != [True, True]
                or any(device["sm"] != [12, 0] for device in env_check["devices"])
            ):
                raise RuntimeError(
                    "Two visible SM120 GPUs and bidirectional P2P are required"
                )
            command(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    "-m",
                    "nccl_ep_test.pair_followups",
                    "--report-dir",
                    str(reports),
                ],
                reports / "pair.log",
                timeout=900,
            )
            result["ranks"] = [
                json.loads((reports / f"pair-rank{rank}.json").read_text())
                for rank in range(2)
            ]
            assert all(record["passed"] for record in result["ranks"])
        else:
            prerequisite(reports, "pair.json")
            from .environment import binding_check, prepare_jit

            result["bindings"] = binding_check()
            result["jit"] = prepare_jit()  # Export paths to the server subprocesses.
            result["serving"] = [
                serving_smoke(reports, args.port, graph=graph)
                for graph in (False, True)
            ]
            result["model_revision"] = REVISION
            result["full_model_accuracy_benchmark"] = False
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save(target, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

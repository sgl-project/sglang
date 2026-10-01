"""Pilot serving comparison inside an isolated, pinned ROCm environment.

Requires the patched SGLang source, launch.sh, and downloaded model in --work.
Starts only its own localhost server and terminates its own process group.
"""

import argparse
import json
import os
import runpy
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def wait_ready(process, url, timeout=3600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("Server did not become ready within 60 minutes")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("/work"))
    parser.add_argument("--modes", nargs="+", default=["baseline", "static", "planned"])
    parser.add_argument("--port", type=int, default=30927)
    parser.add_argument("--accuracy-questions", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--dummy-smoke", action="store_true")
    args = parser.parse_args()
    work = args.work.resolve()
    result = work / ("dummy-smoke-results" if args.dummy_smoke else "serving-results")
    result.mkdir(exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    model = str(work / "model")
    common_env = os.environ.copy()
    common_env.update(
        {
            "MODEL_PATH": model,
            "PORT": str(args.port),
            "TP": "4",
            "HIP_VISIBLE_DEVICES": "0,1,2,3",
            "SGLANG_OPT_USE_BF16_ROUTER_GEMM": "0",
            "SGLANG_FORCE_MXFP8_BLOCK_CONVERT": "1",
        }
    )
    manifest = {
        "scope": "pilot synthetic serving comparison; 64-question accuracy smoke check is not full qualification",
        "model_revision": "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba",
        "modes": args.modes,
        "tp": 4,
        "gpus": [0, 1, 2, 3],
        "seed": 20260927,
        "concurrency": [1, 2, 10, 15, 20],
        "repetitions": args.repetitions,
        "accuracy_questions": args.accuracy_questions,
    }
    if args.dummy_smoke:
        manifest["scope"] = (
            "dummy-weight startup and request smoke checks; no model accuracy or performance claim"
        )
    (result / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for mode in args.modes:
        if mode not in ("baseline", "static", "planned"):
            raise ValueError(mode)
        mode_dir = result / mode
        mode_dir.mkdir(exist_ok=True)
        env = dict(common_env, MODE=mode)
        command = [
            "bash",
            str(work / "launch.sh"),
            "--context-length",
            "65536",
            "--cuda-graph-max-bs-decode",
            "32",
            "--max-running-requests",
            "32",
            "--watchdog-timeout",
            "1200",
            "--random-seed",
            "20260927",
        ]
        if args.dummy_smoke:
            command.extend(["--load-format", "dummy"])
        (mode_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (mode_dir / "server.log").open("w") as log:
            print(f"Starting {mode}", flush=True)
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=work,
            )
            try:
                wait_ready(process, url)
                with urllib.request.urlopen(
                    url + "/get_server_info", timeout=30
                ) as response:
                    (mode_dir / "server-info.json").write_bytes(response.read())
                print(f"Ready: {mode}", flush=True)
                if args.dummy_smoke:
                    for lengths in ([17], [257, 17], [9000]):
                        request = urllib.request.Request(
                            url + "/generate",
                            data=json.dumps(
                                {
                                    "input_ids": [
                                        [1000 + i % 100 for i in range(length)]
                                        for length in lengths
                                    ],
                                    "sampling_params": {
                                        "max_new_tokens": 16,
                                        "ignore_eos": True,
                                        "temperature": 0,
                                    },
                                }
                            ).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(request, timeout=1200) as response:
                            output = json.loads(response.read())
                        if not isinstance(output, list) or len(output) != len(lengths):
                            raise RuntimeError(
                                f"Unexpected smoke response for {mode}: {output}"
                            )
                        if any(
                            item["meta_info"]["completion_tokens"] != 16
                            for item in output
                        ):
                            raise RuntimeError(
                                f"Incomplete smoke generation for {mode}"
                            )
                        print(
                            f"Dummy request passed: {mode} lengths={lengths}",
                            flush=True,
                        )
                    (mode_dir / "smoke.json").write_text(
                        json.dumps({"passed": True, "scope": manifest["scope"]}) + "\n"
                    )
                    continue
                if args.accuracy_questions:
                    test = (
                        work
                        / "sglang/test/registered/amd/accuracy/mi35x/test_minimax_m3_tp4_eval_mi35x.py"
                    )
                    namespace = runpy.run_path(str(test))
                    accuracy, invalid, elapsed = namespace["run_gsm8k_benchmark"](
                        url,
                        model,
                        num_questions=args.accuracy_questions,
                        parallel=8,
                        max_tokens=4096,
                    )
                    accuracy_result = {
                        "accuracy": accuracy,
                        "invalid": invalid,
                        "elapsed_s": elapsed,
                        "questions": args.accuracy_questions,
                    }
                    (mode_dir / "accuracy.json").write_text(
                        json.dumps(accuracy_result, indent=2) + "\n"
                    )
                    print(f"Accuracy {mode}: {accuracy_result}", flush=True)
                    if invalid or accuracy < 0.9:
                        raise RuntimeError(f"Accuracy smoke gate failed for {mode}")
                for repeat in range(args.repetitions):
                    for workload, input_len, ratio in (
                        ("uniform8k", 8192, 1.0),
                        ("mixed32k", 32768, 0.03125),
                    ):
                        for concurrency in (1, 2, 10, 15, 20):
                            name = f"{workload}-c{concurrency}-r{repeat}"
                            benchmark = [
                                "python3",
                                "-m",
                                "sglang.benchmark.serving",
                                "--backend",
                                "sglang",
                                "--base-url",
                                url,
                                "--model",
                                model,
                                "--dataset-name",
                                "random",
                                "--random-input-len",
                                str(input_len),
                                "--random-output-len",
                                "256",
                                "--random-range-ratio",
                                str(ratio),
                                "--num-prompts",
                                str(max(8, concurrency * 2)),
                                "--max-concurrency",
                                str(concurrency),
                                "--request-rate",
                                "inf",
                                "--warmup-requests",
                                str(concurrency),
                                "--seed",
                                "20260927",
                                "--disable-tqdm",
                                "--output-details",
                                "--output-file",
                                str(mode_dir / f"{name}.jsonl"),
                            ]
                            print(f"Benchmark {mode}: {name}", flush=True)
                            with (mode_dir / f"{name}.log").open("w") as bench_log:
                                subprocess.run(
                                    benchmark,
                                    env=env,
                                    stdout=bench_log,
                                    stderr=subprocess.STDOUT,
                                    check=True,
                                    timeout=1200,
                                    cwd=work,
                                )
            finally:
                stop(process)
        print(f"Finished {mode}", flush=True)


if __name__ == "__main__":
    main()

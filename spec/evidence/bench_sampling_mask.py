"""Compare steady decode throughput before and after sampling-mask overlap support.

Run from an environment containing this checkout's serving dependencies. Both
servers use detached worktrees, the same model snapshot, and the selected GPU.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer

REVISIONS = {
    "before": "756d0e0a851c0aee706670affa90c3c47e317d15",
    "after": "cb30f7c6a969b5d19633d5f51c387426f4c00b9a",
}
DECODE_LINE = re.compile(
    r"Decode batch.*?#running-req: (\d+),.*?cuda graph: (True|False), "
    r"gen throughput \(token/s\): ([\d.]+)"
)
BATCH_SIZES = (1, 16, 64)
INPUT_LEN, OUTPUT_LEN = 128, 512


def decode_rates(log, batch_size):
    """Equal-token, 16-step intervals: trim startup/tail before filtering."""
    rows = DECODE_LINE.findall(log)
    eligible = [
        (graph, float(rate))
        for size, graph, rate in rows[2:-1]
        if int(size) == batch_size
    ]
    if not eligible:
        raise ValueError("No complete steady full-batch decode intervals")
    if any(
        graph != "True" or not math.isfinite(rate) or rate <= 0
        for graph, rate in eligible
    ):
        raise ValueError("Steady decode requires CUDA graphs and positive finite rates")
    return [rate for _, rate in eligible]


def validate_outputs(outputs, batch_size):
    if not isinstance(outputs, list) or len(outputs) != batch_size:
        raise ValueError("Response does not contain the requested batch")
    lengths = []
    for output in outputs:
        meta = output["meta_info"]
        if meta["finish_reason"]["type"] != "length":
            raise ValueError(f"Request failed: {meta['finish_reason']}")
        tokens = output["output_ids"]
        masks = meta["output_token_sampling_mask"]
        logprobs = meta["output_token_sampling_logprobs"]
        if not (
            len(tokens) == len(masks) == len(logprobs) == OUTPUT_LEN
            and meta["completion_tokens"] == OUTPUT_LEN
            and meta["output_token_sampling_mask_length"] == OUTPUT_LEN
        ):
            raise ValueError("Token, mask, and logprob counts are not aligned")
        for token, mask, logprob in zip(tokens, masks, logprobs):
            if not mask or token not in mask or not math.isfinite(logprob):
                raise ValueError("Invalid sampling support or selected-token logprob")
            lengths.append(len(mask))
    return {"min_mask_tokens": min(lengths), "max_mask_tokens": max(lengths)}


def wait_ready(process, url):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("Server startup exceeded 900 seconds")


def run_trial(url, log_path, prompts):
    requests.post(url + "/flush_cache", timeout=60).raise_for_status()
    offset = log_path.stat().st_size
    start = time.perf_counter()
    response = requests.post(
        url + "/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "temperature": 1.0,
                "top_k": 4096,
                "top_p": 1.0,
                "min_p": 0.0,
                "max_new_tokens": OUTPUT_LEN,
                "ignore_eos": True,
            },
            "return_sampling_mask": True,
            "stream": False,
        },
        timeout=1800,
    )
    response.raise_for_status()
    outputs = response.json()
    validation = validate_outputs(outputs, len(prompts))
    with log_path.open("rb") as log:
        log.seek(offset)
        trial_log = log.read().decode(errors="replace")
    rates = decode_rates(trial_log, len(prompts))
    return {
        "status": "ok",
        "decode_tokens_per_second": statistics.harmonic_mean(rates),
        "interval_rates": rates,
        "http_and_validation_seconds": time.perf_counter() - start,
        "server_log_byte_start": offset,
        "server_log_byte_end": log_path.stat().st_size,
        **validation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--model-revision", help="Hugging Face model commit; resolved once if omitted"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model_revision = args.model_revision or HfApi().model_info(args.model).sha
    model_path = snapshot_download(
        args.model,
        revision=model_revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rng = random.Random(42)
    special_ids = set(tokenizer.all_special_ids)
    vocab = [i for i in range(tokenizer.vocab_size) if i not in special_ids]
    prompts = [rng.choices(vocab, k=INPUT_LEN) for _ in range(max(BATCH_SIZES))]
    (output_dir / "prompts.json").write_text(json.dumps(prompts))
    package_names = (
        "torch",
        "sglang-kernel",
        "flashinfer-python",
        "transformers",
        "apache-tvm-ffi",
        "nvidia-cutlass-dsl",
    )
    installed = {
        d.metadata["Name"].lower(): d.version
        for d in importlib.metadata.distributions()
    }
    report = {
        "model": args.model,
        "model_revision": model_revision,
        "revisions": REVISIONS,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: installed.get(name) for name in package_names},
        "gpu": args.gpu,
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,power.limit",
                "--format=csv",
            ],
            text=True,
        ),
        "trials": [],
        "servers": {},
    }

    def save():
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    url = f"http://127.0.0.1:{args.port}"
    failed = False
    for label, revision in REVISIONS.items():
        with tempfile.TemporaryDirectory(prefix=f"sampling-mask-{label}-") as directory:
            tree = Path(directory) / "checkout"
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "worktree",
                    "add",
                    "--detach",
                    str(tree),
                    revision,
                ],
                check=True,
            )
            process = None
            try:
                env = os.environ.copy()
                env.update(
                    CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(tree / "python")
                )
                command = [
                    str(Path(sys.executable).with_name("sglang")),
                    "serve",
                    "--model-path",
                    model_path,
                    "--dtype",
                    "bfloat16",
                    "--tp-size",
                    "1",
                    "--sampling-backend",
                    "flashinfer",
                    "--attention-backend",
                    "flashinfer",
                    "--mem-fraction-static",
                    "0.7",
                    "--max-total-tokens",
                    "65536",
                    "--max-running-requests",
                    "64",
                    "--chunked-prefill-size",
                    "8192",
                    "--cuda-graph-max-bs-decode",
                    "64",
                    "--cuda-graph-max-bs-prefill",
                    "128",
                    "--disable-radix-cache",
                    "--decode-log-interval",
                    "16",
                    "--random-seed",
                    "42",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ]
                command += (
                    ["--disable-overlap-schedule"]
                    if label == "before"
                    else ["--sampling-mask-max-tokens", "8192"]
                )
                report["servers"][label] = {
                    "command": shlex.join(command),
                    "environment": {
                        name: env[name]
                        for name in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")
                    },
                }
                save()
                log_path = output_dir / f"{label}.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=tree,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    wait_ready(process, url)
                    report["servers"][label]["server_info"] = requests.get(
                        url + "/server_info", timeout=30
                    ).json()
                    save()
                    for batch_size in BATCH_SIZES:
                        for repeat in range(4):
                            trial = {
                                "revision": label,
                                "batch_size": batch_size,
                                "repeat": repeat,
                                "warmup": repeat == 0,
                            }
                            print(
                                f"{label}: batch={batch_size}, {'warmup' if repeat == 0 else f'run={repeat}'}",
                                flush=True,
                            )
                            try:
                                trial.update(
                                    run_trial(url, log_path, prompts[:batch_size])
                                )
                            except Exception as exc:
                                trial.update(status="failed", error=str(exc))
                                failed = True
                            report["trials"].append(trial)
                            save()
                            print(
                                json.dumps(
                                    {
                                        k: v
                                        for k, v in trial.items()
                                        if k != "interval_rates"
                                    }
                                ),
                                flush=True,
                            )
                            if trial["status"] != "ok":
                                break
            finally:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30)
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo),
                        "worktree",
                        "remove",
                        "--force",
                        str(tree),
                    ],
                    check=True,
                )
    report["summary"] = []
    for batch_size in BATCH_SIZES:
        summary = {"batch_size": batch_size}
        for label in REVISIONS:
            rates = [
                t["decode_tokens_per_second"]
                for t in report["trials"]
                if t["revision"] == label
                and t["batch_size"] == batch_size
                and not t["warmup"]
                and t["status"] == "ok"
            ]
            if len(rates) != 3:
                failed = True
                continue
            summary[label] = {
                "median": statistics.median(rates),
                "min": min(rates),
                "max": max(rates),
            }
        if "before" in summary and "after" in summary:
            summary["change_percent"] = (
                summary["after"]["median"] / summary["before"]["median"] - 1
            ) * 100
        report["summary"].append(summary)
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())

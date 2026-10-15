"""Matched TP4 input-logprob/Top-N serving ABBA matrix.

Reserve four idle GPUs before running; this script does not terminate foreign jobs.
Requires a local model and the runtime dependencies of the selected checkout.
Example:
  python benchmark/kernels/bench_input_logprobs_serving.py --model qwen \
    --model-path /path/to/Qwen3-32B --output /path/to/new-results

The four logical cases are generation without logprobs, target plus explicit
probes, prompt Top-N (which includes target scores), and probes plus Top-N.
Raw responses, timings and commands are retained. No tail samples are discarded.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT
MODELS = {"qwen": None, "glm52": None, "glm53": None}


def empty_gpus():
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if apps:
        raise RuntimeError(f"GPU processes present at server boundary: {apps}")


def command(model, port):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODELS[model],
        "--tp-size",
        "4",
        "--max-running-requests",
        "1",
        "--max-queued-requests",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--skip-server-warmup",
        "--random-seed",
        "314159",
        "--watchdog-timeout",
        "300",
    ]
    if model == "qwen":
        cmd += ["--mem-fraction-static", "0.5", "--attention-backend", "triton"]
    else:
        cmd += [
            "--mem-fraction-static",
            "0.75",
            "--trust-remote-code",
            "--quantization",
            "modelopt_fp4",
            "--disable-shared-experts-fusion",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--bf16-gemm-backend",
            "torch",
            "--attention-backend",
            "dsa",
            "--dsa-prefill-backend",
            "flashmla_kv",
            "--dsa-decode-backend",
            "flashmla_kv",
            "--dsa-topk-backend",
            "sgl-kernel",
            "--page-size",
            "64",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-flashinfer-autotune",
        ]
    return cmd


def cells(args, vocab):
    result = []
    for length in args.lengths:
        ids = random.Random(314159 + length).choices(
            range(1000, min(vocab - 1000, 140000)), k=length
        )
        configurations = [("baseline", 0, False), ("logprob", 0, True)]
        for n in args.top_n:
            configurations += [("topn", n, False), ("logprob_topn", n, True)]
        for name, n, probes in configurations:
            payload = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
            }
            if name != "baseline":
                payload.update(
                    return_logprob=True, logprob_start_len=0, top_logprobs_num=n
                )
            if probes:
                payload["token_ids_logprob"] = [
                    0,
                    1,
                    vocab // 4 - 1,
                    vocab // 4,
                    vocab // 2 - 1,
                    vocab // 2,
                    3 * vocab // 4 - 1,
                    3 * vocab // 4,
                    vocab - 1,
                ]
            result.append(
                {
                    "id": f"L{length}_{name}_N{n}",
                    "name": name,
                    "length": length,
                    "top_n": n,
                    "explicit_probes": probes,
                    "payload": payload,
                }
            )
    return result


def validate(data, cell):
    assert len(data["output_ids"]) == 1
    meta = data["meta_info"]
    assert meta["prompt_tokens"] == cell["length"]
    if cell["name"] != "baseline":
        assert len(meta["input_token_logprobs"]) == cell["length"]
    if cell["explicit_probes"]:
        rows = meta["input_token_ids_logprobs"]
        assert len(rows) == cell["length"]
        assert sum(len(r) for r in rows if r) == (cell["length"] - 1) * 9
    if cell["top_n"]:
        rows = meta["input_top_logprobs"]
        assert len(rows) == cell["length"]
        assert sum(len(r) for r in rows if r) == (cell["length"] - 1) * cell["top_n"]


def run_block(args, index, mode, matrix, output):
    empty_gpus()
    block = output / f"{index}_{mode}"
    block.mkdir()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        PYTHONPATH=f"{TREE / 'python'}:{os.environ.get('PYTHONPATH', '')}",
        SGLANG_ENABLE_DISTRIBUTED_INPUT_LOGPROBS=str(mode),
        SGLANG_JIT_DEEPGEMM_PRECOMPILE="0",
    )
    cmd = command(args.model, args.port)
    logpath = block / "server.log"
    rows = []
    with logpath.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=TREE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 720
            with requests.Session() as session:
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"server exited {proc.returncode}: {logpath}"
                        )
                    try:
                        if session.get(
                            f"http://127.0.0.1:{args.port}/ready", timeout=2
                        ).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError(f"startup exceeded 720s: {logpath}")
                print(
                    json.dumps({"block": index, "mode": mode, "event": "ready"}),
                    flush=True,
                )
                # Each cell gets identical warmups, then interleaved measurement rounds.
                order = matrix if index % 2 == 0 else list(reversed(matrix))
                for iteration in range(-args.warmups, args.samples):
                    rotated = (
                        order[iteration % len(order) :]
                        + order[: iteration % len(order)]
                    )
                    for cell in rotated:
                        started = time.perf_counter()
                        response = session.post(
                            f"http://127.0.0.1:{args.port}/generate",
                            json=cell["payload"],
                            timeout=300,
                        )
                        seconds = time.perf_counter() - started
                        response.raise_for_status()
                        data = response.json()
                        validate(data, cell)
                        (block / f"{cell['id']}_{iteration}.json").write_text(
                            json.dumps(data)
                        )
                        if iteration >= 0:
                            rows.append(
                                {
                                    "cell": cell["id"],
                                    "iteration": iteration,
                                    "seconds": seconds,
                                    "response_bytes": len(response.content),
                                    "output_ids": data["output_ids"],
                                }
                            )
                    print(
                        json.dumps(
                            {"block": index, "mode": mode, "iteration": iteration}
                        ),
                        flush=True,
                    )
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
            (block / "result.json").write_text(
                json.dumps(
                    {"rows": rows, "returncode": proc.returncode, "command": cmd},
                    indent=2,
                )
            )
    assert proc.returncode == 0, proc.returncode
    if args.route_marker:
        assert (args.route_marker in logpath.read_text()) == bool(mode)
    empty_gpus()
    return rows


def main():
    global TREE
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, default=TREE)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--route-marker", help="Optional task-local route instrumentation marker"
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    parser.add_argument("--top-n", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--port", type=int, default=30582)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    TREE = args.worktree.resolve()
    MODELS[args.model] = args.model_path
    assert len(args.gpus.split(",")) == 4, "This benchmark is TP4"
    assert args.samples > 0 and args.warmups > 0 and all(n > 0 for n in args.top_n)
    vocab = json.loads((Path(MODELS[args.model]) / "config.json").read_text())[
        "vocab_size"
    ]
    matrix = cells(args, vocab)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TREE, text=True
    ).strip()
    if not args.dry_run:
        assert not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=TREE, text=True
        ).strip()
    manifest = {
        "source": source,
        "model": MODELS[args.model],
        "args": vars(args) | {"output": str(args.output), "worktree": str(TREE)},
        "order": [0, 1, 1, 0],
        "cells": matrix,
        "api_note": "Prompt Top-N includes target-token logprobs; logprob cases additionally request nine explicit IDs.",
        "timing": "HTTP whole-response wall time, excludes JSON parsing and server startup; concurrency 1",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "cells": [c["id"] for c in matrix],
                    "command": command(args.model, args.port),
                }
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    blocks = [
        run_block(args, i, mode, matrix, args.output)
        for i, mode in enumerate([0, 1, 1, 0])
    ]
    summary = []
    for cell in matrix:
        entry = {"cell": cell["id"]}
        for mode, indices in [("gathered", [0, 3]), ("distributed", [1, 2])]:
            samples = [
                r["seconds"]
                for i in indices
                for r in blocks[i]
                if r["cell"] == cell["id"]
            ]
            entry[mode] = {
                "samples": samples,
                "median": statistics.median(samples),
                "mean": statistics.mean(samples),
            }
        entry["median_speedup"] = (
            entry["gathered"]["median"] / entry["distributed"]["median"]
        )
        summary.append(entry)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "COMPLETE", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()

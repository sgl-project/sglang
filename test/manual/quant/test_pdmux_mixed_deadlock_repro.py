#!/usr/bin/env python3
"""Minimal reproducer: SGLang PDMux mixed-batch deadlock on hybrid GDN models.

Symptom
-------
On a single A100/H100 with `--enable-pdmux` and a Qwen3.5/Qwen3.8 hybrid
(GDN + full-attention) model, once the server has completed one >=4K-token
prompt through the split-prefill path, the NEXT request pattern that runs a
decode batch and a split-prefill batch concurrently deadlocks the scheduler
event loop forever. py-spy shows the loop blocked at:

    synchronize (torch/cuda/streams.py)
    event_loop_pdmux (sglang/srt/multiplex/multiplexing_mixin.py:205)

i.e. `decode_stream.synchronize()` waiting for decode-forward kernels that
never complete. The watchdog does not fire. No client abort is required.

Reproduction matrix (single A100-SXM4-40GB, SGLang <commit>, budget 256)
------------------------------------------------------------------------
prior request                      | following mixed batch
-----------------------------------+---------------------
1K-prompt request(s), any decode   | 6/6 pass
4K-prompt request (512 decode)     | 2/2 deadlock
4K-prompt request, aborted at 5 s  | 2/2 deadlock
4K prompt via /flush_cache path    | deadlock independent of flush

Requirements: one A100/H100 (>=40 GB), the Qwen3.8-27B-FP8 checkpoint
(or any hybrid qwen3_5-family checkpoint that fits), py-spy for the stack
dump. Runtime is ~8 min plus checkpoint download.

Usage
-----
    python test/manual/quant/test_pdmux_mixed_deadlock_repro.py --model Qwen/Qwen3.8-27B-FP8

The script launches its own server, so run it on an idle GPU. With the
stale-field fix in this PR applied, PDMux starts on current main; without
the workspace fix the mixed batch deadlocks; with both it runs clean.
Manual (not CI): needs a >=40GB GPU and downloads the checkpoint.
See sgl-project/sglang#37904 for the full evidence trail.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

PORT = 30000
BASE_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_SECONDS = 1800
MIXED_TIMEOUT_SECONDS = 300
PROMPT_1K = "test " * 1023
PROMPT_4K = "test " * 4095


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--split-budget", type=int, default=256)
    parser.add_argument("--mixed-timeout", type=float, default=MIXED_TIMEOUT_SECONDS)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="sensitize+trigger rounds; b1 hits ~30-70%/try pre-fix, b2 6/6",
    )
    parser.add_argument("--workdir", default=".")
    return parser.parse_args()


ARGS: argparse.Namespace


def wait_ready(process: subprocess.Popen[str]) -> None:
    url = f"http://127.0.0.1:{ARGS.port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    raise TimeoutError("server readiness timeout")


def stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, 30),
        (signal.SIGTERM, 15),
        (signal.SIGKILL, 10),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def generate(prompt: str, max_new_tokens: int, timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        BASE_URL + "/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def hash_of(result: dict[str, object]) -> str:
    return hashlib.sha256(str(result["text"]).encode("utf-8")).hexdigest()


def mixed_case(batch_size: int) -> None:
    """One decode batch plus `batch_size` concurrent split-prefill batches.

    b1 (one prefill) reproduced the deadlock on roughly a third to two
    thirds of single attempts pre-fix; b2 (two prefills) was 6/6 across
    the pre-fix baseline campaign. The main loop therefore tries both,
    several times, re-priming the >=4K sensitizing request each round.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size + 1) as executor:
        background = executor.submit(
            generate, "Count upward: 1, 2, 3,", 256, ARGS.mixed_timeout
        )
        time.sleep(0.25)
        prefills = [
            executor.submit(generate, PROMPT_1K, 32, ARGS.mixed_timeout)
            for _ in range(batch_size)
        ]
        for future in prefills:
            future.result(timeout=ARGS.mixed_timeout + 30)
        background.result(timeout=ARGS.mixed_timeout + 30)


def dump_stacks() -> None:
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        print("[repro] py-spy not found; skipping stack dump (pip install py-spy)")
        return
    pids = subprocess.run(
        ["pgrep", "-f", "sglang::scheduler"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    for pid in pids:
        dump = subprocess.run(
            [py_spy, "dump", "--pid", pid], capture_output=True, text=True, check=False
        )
        print(f"[repro] stack of scheduler pid {pid}:\n{dump.stdout}")


def main() -> None:
    global ARGS
    ARGS = parse_args()
    workdir = pathlib.Path(ARGS.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config = (workdir / "pdmux-repro.yaml").resolve()
    config.write_text(
        "sm_group_num: 3\n"
        "manual_divisions:\n"
        "  - [72, 36, 1]\n"
        f"split_forward_token_budget: {ARGS.split_budget}\n"
        "decode_bs_divisor: 36\n",
        encoding="utf-8",
    )
    argv = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        ARGS.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(ARGS.port),
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--mem-fraction-static",
        "0.90",
        "--max-running-requests",
        "12",
        "--max-total-tokens",
        "24576",
        "--max-mamba-cache-size",
        "12",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--disable-overlap-schedule",
        "--random-seed",
        "7",
        "--watchdog-timeout",
        "1800",
        "--skip-server-warmup",
        "--enable-pdmux",
        "--sm-group-num",
        "3",
        "--pdmux-config-path",
        str(config),
    ]
    if ARGS.revision:
        argv.extend(["--revision", ARGS.revision])

    log_path = workdir / "pdmux-repro-server.log"
    print(f"[repro] starting server, log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            wait_ready(process)
            print("[repro] step 1: 1K-prompt reference (expected to pass)")
            generate(PROMPT_1K, 32, ARGS.mixed_timeout)
            for attempt in range(1, ARGS.attempts + 1):
                print(
                    f"[repro] attempt {attempt}/{ARGS.attempts}: 4K-prompt request, 512 decode tokens"
                )
                victim = generate(PROMPT_4K, 512, 600)
                print(
                    f"[repro]   victim finished, tokens={victim['meta_info']['completion_tokens']}"
                )
                for label, batch_size in (("b1", 1), ("b2", 2)):
                    print(
                        f"[repro] attempt {attempt} mixed {label}: decode + {batch_size}x 1K prefill"
                    )
                    try:
                        mixed_case(batch_size)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[repro] mixed {label} timed out: {type(exc).__name__}")
                        time.sleep(2)
                        dump_stacks()
                        print(
                            "[repro] DEADLOCK REPRODUCED: the scheduler event loop is blocked\n"
                            "        in decode_stream.synchronize(); attach nsys or the event\n"
                            "        tracer from the issue for the device-side wait chain."
                        )
                        raise SystemExit(2)
            print(
                "[repro] all attempts completed - deadlock did NOT reproduce.\n"
                "        The race is timing sensitive; please re-run and share\n"
                "        GPU model + driver + torch version."
            )
        finally:
            stop_process_group(process)


if __name__ == "__main__":
    main()

"""
End-to-end regression test for DeepEP Waterfill (DeepSeek-V3/R1).

This script runs the same accuracy + serving performance tests we used during
the Waterfill development:
  - GSM8K accuracy (200 questions, 5-shot)
  - MMLU accuracy (nsub=60, ntrain=5)
  - Serving benchmark (random dataset, output_len=1) for a fixed case list

It is designed to run inside the `sglang_dev` docker container (or any
environment where `python3 -m sglang.launch_server` is available).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

DEFAULT_HOST_URL = "http://127.0.0.1"
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_PORT = 30000

DEFAULT_TP = 8
DEFAULT_EP = 8

# Default serving cases (reduced set for faster regression runs)
DEFAULT_CASES = "256:64,1024:32,4096:16,16384:8"


@dataclass(frozen=True)
class BenchCase:
    input_len: int
    max_concurrency: int
    num_prompts: int

    @property
    def key(self) -> str:
        return f"in{self.input_len}_c{self.max_concurrency}_n{self.num_prompts}"


def _run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), "(cwd=" + str(cwd) + ")", flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=check)


def _read_last_jsonl(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _round_up_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return x
    return ((x + m - 1) // m) * m


def _round_down_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return x
    return (x // m) * m


def _clamp_num_prompts(num_prompts: int, *, conc: int, max_v: int) -> int:
    # Align to concurrency so that we always have full waves.
    n = max(num_prompts, 1)
    n = _round_up_to_multiple(n, conc)
    if max_v > 0 and n > max_v:
        n = _round_down_to_multiple(max_v, conc)
        if n <= 0:
            n = max_v
    return max(n, 1)


def parse_cases(
    cases_str: str, *, requests_per_concurrency: int, max_num_prompts: int
) -> List[BenchCase]:
    cases: List[BenchCase] = []
    for raw in cases_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.replace("=", ":").split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid --cases item: {raw!r}")
        in_len = int(parts[0])
        conc = int(parts[1])
        if len(parts) == 3:
            num_prompts = int(parts[2])
        else:
            num_prompts = conc * requests_per_concurrency
        num_prompts = _clamp_num_prompts(num_prompts, conc=conc, max_v=max_num_prompts)
        cases.append(
            BenchCase(input_len=in_len, max_concurrency=conc, num_prompts=num_prompts)
        )

    cases.sort(key=lambda c: (c.input_len, c.max_concurrency))
    return cases


def wait_for_server(host_url: str, port: int, timeout_s: int = 1200) -> None:
    url = f"{host_url}:{port}/health"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Server not ready after {timeout_s}s: {url}")


def start_server(
    *,
    repo_dir: str,
    model_path: str,
    bind_host: str,
    port: int,
    tp: int,
    ep: int,
    enable_waterfill: bool,
    disable_shared_experts_fusion: bool,
    log_path: str,
) -> Tuple[subprocess.Popen, object]:
    flags = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(tp),
        "--ep-size",
        str(ep),
        "--moe-a2a-backend",
        "deepep",
        "--disable-radix-cache",
        "--host",
        bind_host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--deepep-mode",
        "normal",
        "--log-level",
        "warning",
    ]
    if enable_waterfill:
        flags.insert(flags.index("--host"), "--enable-deepep-waterfill")
    if disable_shared_experts_fusion:
        flags.insert(flags.index("--host"), "--disable-shared-experts-fusion")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    p = subprocess.Popen(flags, cwd=repo_dir, stdout=f, stderr=subprocess.STDOUT)
    return p, f


def stop_server(proc: subprocess.Popen, log_fh: object) -> None:
    try:
        proc.terminate()
    except Exception:
        pass
    time.sleep(5)
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        log_fh.close()
    except Exception:
        pass


def ensure_mmlu_data(data_root: str) -> str:
    """
    Ensures MMLU data exists and returns the path to the 'data' directory.

    Output layout:
      {data_root}/data/dev
      {data_root}/data/test
    """
    tar_path = os.path.join(data_root, "data.tar")
    data_dir = os.path.join(data_root, "data")
    test_dir = os.path.join(data_dir, "test")
    dev_dir = os.path.join(data_dir, "dev")
    if os.path.isdir(test_dir) and os.path.isdir(dev_dir):
        return data_dir

    os.makedirs(data_root, exist_ok=True)
    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    print(f"[mmlu] downloading {url} -> {tar_path}", flush=True)
    urllib.request.urlretrieve(url, tar_path)
    print(f"[mmlu] extracting {tar_path} -> {data_root}", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(data_root)

    if not (os.path.isdir(test_dir) and os.path.isdir(dev_dir)):
        raise RuntimeError(f"MMLU data not found after extract: {data_dir}")
    return data_dir


def run_gsm8k(
    *,
    repo_dir: str,
    out_dir: str,
    host_url: str,
    port: int,
    parallel: int,
    num_shots: int,
    num_questions: int,
    tag: str,
) -> str:
    result_file = os.path.join(out_dir, f"gsm8k_{tag}.jsonl")
    raw_file = os.path.join(out_dir, f"gsm8k_{tag}_raw.json")
    _run(
        [
            "python3",
            os.path.join(repo_dir, "benchmark/gsm8k/bench_sglang.py"),
            "--backend",
            "srt",
            "--host",
            host_url,
            "--port",
            str(port),
            "--parallel",
            str(parallel),
            "--num-shots",
            str(num_shots),
            "--num-questions",
            str(num_questions),
            "--result-file",
            result_file,
            "--raw-result-file",
            raw_file,
        ],
        cwd=out_dir,
    )
    return result_file


def run_mmlu(
    *,
    repo_dir: str,
    out_dir: str,
    host_url: str,
    port: int,
    parallel: int,
    ntrain: int,
    nsub: int,
    data_dir: str,
    tag: str,
) -> str:
    result_file = os.path.join(out_dir, f"mmlu_{tag}.jsonl")
    raw_file = os.path.join(out_dir, f"mmlu_{tag}_raw.json")
    _run(
        [
            "python3",
            os.path.join(repo_dir, "benchmark/mmlu/bench_sglang.py"),
            "--backend",
            "srt",
            "--host",
            host_url,
            "--port",
            str(port),
            "--parallel",
            str(parallel),
            "--ntrain",
            str(ntrain),
            "--nsub",
            str(nsub),
            "--data_dir",
            data_dir,
            "--result-file",
            result_file,
            "--raw-result-file",
            raw_file,
        ],
        cwd=out_dir,
    )
    return result_file


def run_bench_serving(
    *,
    sglang_dir: str,
    host: str,
    port: int,
    model_path: str,
    num_prompts: int,
    random_input: int,
    random_output: int,
    max_concurrency: int,
    output_file: str,
) -> dict:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    _run(
        [
            "python3",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--host",
            host,
            "--port",
            str(port),
            "--dataset-name",
            "random",
            "--num-prompts",
            str(num_prompts),
            "--random-input",
            str(random_input),
            "--random-output",
            str(random_output),
            "--max-concurrency",
            str(max_concurrency),
            "--model",
            model_path,
            "--output-file",
            output_file,
        ],
        cwd=sglang_dir,
    )
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_profile_mode_tag(*, mode: str, repo_dir: str) -> str:
    name = Path(repo_dir).name
    if mode == "baseline":
        if name.startswith("sglang_baseline_"):
            return "baseline" + name[len("sglang_baseline_") :]
        if name.startswith("baseline_"):
            return "baseline" + name[len("baseline_") :]
        return "baseline"
    # mode == "waterfill"
    if name == "sglang":
        return "waterfill_current"
    if name.startswith("sglang_wf_"):
        return "waterfill_" + name[len("sglang_wf_") :]
    return "waterfill"


def run_bench_one_batch_server_profile(
    *,
    sglang_dir: str,
    base_url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    profile_steps: int,
    profile_prefix: str,
    profile_output_dir: str,
    result_file: str,
) -> str:
    """
    Run `sglang.bench_one_batch_server` against an already-running server and
    trigger torch profiling via `--profile`.

    Returns the directory that contains the profiler artifacts.
    """
    os.makedirs(profile_output_dir, exist_ok=True)
    before = set(os.listdir(profile_output_dir))

    _run(
        [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            # `ServerArgs` requires --model-path even in --base-url mode.
            # Use a dummy value to bypass model-related validations.
            "--model-path",
            "none",
            "--base-url",
            base_url,
            "--batch-size",
            str(batch_size),
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--seed",
            "1",
            "--profile",
            "--profile-by-stage",
            "--profile-steps",
            str(profile_steps),
            "--profile-prefix",
            profile_prefix,
            "--profile-output-dir",
            profile_output_dir,
            "--result-filename",
            result_file,
            "--no-append-to-github-summary",
        ],
        cwd=sglang_dir,
    )

    # `sglang.profiler.run_profile` always creates a time-stamped subdir under
    # `--profile-output-dir`. Find the newly created one.
    after = set(os.listdir(profile_output_dir))
    new_dirs = []
    for d in sorted(after - before):
        p = os.path.join(profile_output_dir, d)
        if os.path.isdir(p):
            new_dirs.append(p)
    if not new_dirs:
        # Fallback: pick the most recently modified directory.
        all_dirs = [
            os.path.join(profile_output_dir, d)
            for d in os.listdir(profile_output_dir)
            if os.path.isdir(os.path.join(profile_output_dir, d))
        ]
        if not all_dirs:
            raise RuntimeError(
                f"No profiler output directory found under: {profile_output_dir}"
            )
        all_dirs.sort(key=os.path.getmtime)
        return all_dirs[-1]

    new_dirs.sort(key=os.path.getmtime)
    return new_dirs[-1]


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline-sglang-dir", type=str, default="")
    parser.add_argument(
        "--waterfill-sglang-dir",
        type=str,
        default="",
        help="Defaults to this repo root.",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="",
        help="Where to write outputs. Defaults to /lustre/.../bench if it exists; otherwise ./bench.",
    )

    # Server
    parser.add_argument(
        "--model-path", type=str, default=os.environ.get("MODEL_PATH", "")
    )
    parser.add_argument("--host-url", type=str, default=DEFAULT_HOST_URL)
    parser.add_argument("--bind-host", type=str, default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument("--ep", type=int, default=DEFAULT_EP)
    parser.add_argument(
        "--disable-shared-experts-fusion",
        action="store_true",
        help="Pass --disable-shared-experts-fusion to both baseline and waterfill servers.",
    )

    # Accuracy
    # Default: run accuracy. Use --skip-accuracy to opt out.
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip accuracy evaluation (GSM8K + MMLU).",
    )
    parser.add_argument("--gsm8k-parallel", type=int, default=64)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-num-questions", type=int, default=200)
    parser.add_argument("--mmlu-parallel", type=int, default=8)
    parser.add_argument("--mmlu-ntrain", type=int, default=5)
    parser.add_argument("--mmlu-nsub", type=int, default=60)
    parser.add_argument("--mmlu-data-dir", type=str, default="")

    # Serving benchmark
    # Default: run serving benchmark. Use --skip-serving to opt out.
    parser.add_argument(
        "--skip-serving",
        action="store_true",
        help="Skip serving benchmark.",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES)
    parser.add_argument("--requests-per-concurrency", type=int, default=16)
    parser.add_argument("--max-num-prompts", type=int, default=512)

    # Torch profiling (one-batch server benchmark)
    parser.add_argument(
        "--run-torch-profile",
        action="store_true",
        help=(
            "Run a one-batch benchmark with `python -m sglang.bench_one_batch_server "
            "--profile` (bs=16, input_len=1024, output_len=1) to dump torch profiler "
            "traces for baseline and waterfill."
        ),
    )
    parser.add_argument(
        "--torch-profile-root",
        type=str,
        default="",
        help="Directory to store torch profiler traces (defaults to <result-root>/torch_profile).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    waterfill_dir = args.waterfill_sglang_dir or str(repo_root)
    baseline_dir = args.baseline_sglang_dir

    if not args.model_path:
        raise ValueError(
            "--model-path is required (or set env MODEL_PATH). "
            "Example: /lustre/.../model/DeepSeek-V3/"
        )

    default_result_root = (
        "/lustre/raplab/client/xutingz/workspace/bench"
        if os.path.isdir("/lustre/raplab/client/xutingz/workspace/bench")
        else str(Path.cwd() / "bench")
    )
    result_root = args.result_root or default_result_root

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(result_root, f"deepep_waterfill_e2e_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("==========================================")
    print("DeepEP Waterfill E2E Test")
    print("==========================================")
    print(f"out_dir: {out_dir}")
    print(f"baseline_dir: {baseline_dir or '(skip)'}")
    print(f"waterfill_dir: {waterfill_dir}")
    print(f"model_path: {args.model_path}")
    print(f"tp={args.tp}, ep={args.ep}, port={args.port}")
    print(f"disable_shared_experts_fusion={args.disable_shared_experts_fusion}")
    print("")

    summary: dict = {
        "out_dir": out_dir,
        "accuracy": {},
        "serving_benchmark": {},
        "torch_profile": {},
    }

    # ---------------- Accuracy ----------------
    if not args.skip_accuracy:
        mmlu_data_dir = (
            args.mmlu_data_dir
            if args.mmlu_data_dir
            else ensure_mmlu_data(os.path.join(out_dir, "mmlu_data"))
        )

        def _run_accuracy_mode(
            mode: str, repo_dir: str, enable_waterfill: bool
        ) -> None:
            print("\n==========================================", flush=True)
            print(f"[acc] START mode={mode} waterfill={enable_waterfill}", flush=True)
            print("==========================================\n", flush=True)

            _run(
                ["pip", "install", "-e", "python[dev]", "--no-deps", "-q"],
                cwd=repo_dir,
                check=False,
            )

            server_log = os.path.join(out_dir, f"server_{mode}.log")
            p, f = start_server(
                repo_dir=repo_dir,
                model_path=args.model_path,
                bind_host=args.bind_host,
                port=args.port,
                tp=args.tp,
                ep=args.ep,
                enable_waterfill=enable_waterfill,
                disable_shared_experts_fusion=args.disable_shared_experts_fusion,
                log_path=server_log,
            )
            try:
                wait_for_server(args.host_url, args.port, timeout_s=1800)
                gsm_path = run_gsm8k(
                    repo_dir=repo_dir,
                    out_dir=out_dir,
                    host_url=args.host_url,
                    port=args.port,
                    parallel=args.gsm8k_parallel,
                    num_shots=args.gsm8k_num_shots,
                    num_questions=args.gsm8k_num_questions,
                    tag=mode,
                )
                mmlu_path = run_mmlu(
                    repo_dir=repo_dir,
                    out_dir=out_dir,
                    host_url=args.host_url,
                    port=args.port,
                    parallel=args.mmlu_parallel,
                    ntrain=args.mmlu_ntrain,
                    nsub=args.mmlu_nsub,
                    data_dir=mmlu_data_dir,
                    tag=mode,
                )
                summary["accuracy"][mode] = {
                    "gsm8k": _read_last_jsonl(gsm_path),
                    "mmlu": _read_last_jsonl(mmlu_path),
                }
            finally:
                stop_server(p, f)

        if baseline_dir:
            _run_accuracy_mode("baseline", baseline_dir, enable_waterfill=False)
        _run_accuracy_mode("waterfill", waterfill_dir, enable_waterfill=True)

    # ---------------- Serving benchmark ----------------
    if not args.skip_serving:
        cases = parse_cases(
            args.cases,
            requests_per_concurrency=args.requests_per_concurrency,
            max_num_prompts=args.max_num_prompts,
        )
        summary["serving_benchmark"]["cases"] = [
            {
                "input_len": c.input_len,
                "max_concurrency": c.max_concurrency,
                "num_prompts": c.num_prompts,
                "key": c.key,
            }
            for c in cases
        ]
        summary["serving_benchmark"]["rounds"] = args.rounds
        summary["serving_benchmark"]["output_len"] = args.output_len
        summary["serving_benchmark"]["results"] = {"baseline": {}, "waterfill": {}}

        def _run_serving_mode(mode: str, repo_dir: str, enable_waterfill: bool) -> None:
            print("\n==========================================", flush=True)
            print(f"[bench] START mode={mode} waterfill={enable_waterfill}", flush=True)
            print("==========================================\n", flush=True)

            _run(
                ["pip", "install", "-e", "python[dev]", "--no-deps", "-q"],
                cwd=repo_dir,
                check=False,
            )

            server_log = os.path.join(out_dir, f"server_{mode}_serving.log")
            p, f = start_server(
                repo_dir=repo_dir,
                model_path=args.model_path,
                bind_host=args.bind_host,
                port=args.port,
                tp=args.tp,
                ep=args.ep,
                enable_waterfill=enable_waterfill,
                disable_shared_experts_fusion=args.disable_shared_experts_fusion,
                log_path=server_log,
            )
            try:
                wait_for_server(args.host_url, args.port, timeout_s=1800)

                for c in cases:
                    key = c.key
                    summary["serving_benchmark"]["results"][mode].setdefault(key, [])
                    for r in range(1, args.rounds + 1):
                        out_file = os.path.join(out_dir, f"{mode}_{key}_r{r}.json")
                        res = run_bench_serving(
                            sglang_dir=repo_dir,
                            host=args.bind_host,
                            port=args.port,
                            model_path=args.model_path,
                            num_prompts=c.num_prompts,
                            random_input=c.input_len,
                            random_output=args.output_len,
                            max_concurrency=c.max_concurrency,
                            output_file=out_file,
                        )
                        summary["serving_benchmark"]["results"][mode][key].append(res)
            finally:
                stop_server(p, f)

        if baseline_dir:
            _run_serving_mode("baseline", baseline_dir, enable_waterfill=False)
        _run_serving_mode("waterfill", waterfill_dir, enable_waterfill=True)

    # ---------------- Torch profiler ----------------
    if args.run_torch_profile:
        torch_profile_root = (
            args.torch_profile_root
            if args.torch_profile_root
            else os.path.join(result_root, "torch_profile")
        )
        os.makedirs(torch_profile_root, exist_ok=True)

        bs = 16
        in_len = 1024
        out_len = 1
        profile_steps = 5
        summary["torch_profile"]["config"] = {
            "batch_size": bs,
            "input_len": in_len,
            "output_len": out_len,
            "profile_steps": profile_steps,
            "root": torch_profile_root,
        }
        summary["torch_profile"]["results"] = {"baseline": {}, "waterfill": {}}

        def _run_torch_profile_mode(
            mode: str, repo_dir: str, enable_waterfill: bool
        ) -> None:
            print("\n==========================================", flush=True)
            print(
                f"[torch_profile] START mode={mode} waterfill={enable_waterfill}",
                flush=True,
            )
            print("==========================================\n", flush=True)

            _run(
                ["pip", "install", "-e", "python[dev]", "--no-deps", "-q"],
                cwd=repo_dir,
                check=False,
            )

            server_log = os.path.join(out_dir, f"server_{mode}_torch_profile.log")
            p, f = start_server(
                repo_dir=repo_dir,
                model_path=args.model_path,
                bind_host=args.bind_host,
                port=args.port,
                tp=args.tp,
                ep=args.ep,
                enable_waterfill=enable_waterfill,
                disable_shared_experts_fusion=args.disable_shared_experts_fusion,
                log_path=server_log,
            )
            try:
                wait_for_server(args.host_url, args.port, timeout_s=1800)
                base_url = f"{args.host_url}:{args.port}"

                tag = _torch_profile_mode_tag(mode=mode, repo_dir=repo_dir)
                profile_out = os.path.join(
                    torch_profile_root, f"{ts}_{tag}_in{in_len}_bs{bs}_o{out_len}"
                )
                os.makedirs(profile_out, exist_ok=True)

                result_file = os.path.join(
                    out_dir,
                    f"bench_one_batch_{mode}_in{in_len}_bs{bs}_o{out_len}.jsonl",
                )
                trace_dir = run_bench_one_batch_server_profile(
                    sglang_dir=repo_dir,
                    base_url=base_url,
                    batch_size=bs,
                    input_len=in_len,
                    output_len=out_len,
                    profile_steps=profile_steps,
                    profile_prefix=tag,
                    profile_output_dir=profile_out,
                    result_file=result_file,
                )
                summary["torch_profile"]["results"][mode] = {
                    "profile_output_dir": profile_out,
                    "trace_dir": trace_dir,
                    "server_log": server_log,
                    "result_file": result_file,
                }
                print(f"[torch_profile] {mode} trace_dir={trace_dir}", flush=True)
            finally:
                stop_server(p, f)

        if baseline_dir:
            _run_torch_profile_mode("baseline", baseline_dir, enable_waterfill=False)
        _run_torch_profile_mode("waterfill", waterfill_dir, enable_waterfill=True)

    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

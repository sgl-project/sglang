"""Benchmark driver for HiCache buffer_only mode.

Runs a matrix of (model, host-memory-mode) server configs through the two
workloads buffer mode targets, and reports hit rates, latency, and the
buffer-mode pipeline counters side by side:

- multiturn: growing-history conversations (delegates to bench_multiturn.py,
  offline random tokens, round barrier, fixed seed);
- longctx: long shared prefix + divergent continuations, measured warm
  (device tier) and again after /flush_cache (through-storage tier).

Example (dense, cache-vs-buffer):
    python benchmark/hicache/bench_buffer_mode.py \
        --model /path/to/model --modes cache,buffer \
        --workloads multiturn,longctx --buffer-size-gb 2 --cache-ratio 2

SWA / Mamba hybrids run the same way (add --tp for large hybrids); the
unified radix tree is selected automatically for them.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

BUFFER_METRICS = [
    "sglang:hicache_existence_cache_skipped_pages_total",
    "sglang:hicache_backup_dropped_tokens_total",
    "sglang:hicache_pending_write_queue_depth",
    "sglang:hicache_host_used_tokens",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--port", type=int, default=31212)
    parser.add_argument(
        "--launch-module",
        type=str,
        default="sglang.launch_server",
        help="server entry module (e.g. sglang_meta.launch_server for "
        "Meta-internal model families)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="cache,buffer",
        help="comma list of deployments to compare: cache (alias cache_wt), "
        "cache_wb (write_back), buffer",
    )
    parser.add_argument(
        "--workloads", type=str, default="multiturn,longctx", help="comma list"
    )
    parser.add_argument("--buffer-size-gb", type=int, default=2)
    parser.add_argument("--cache-ratio", type=float, default=2.0)
    parser.add_argument(
        "--host-ratio",
        type=float,
        default=0.0,
        help="if > 0, size EVERY mode's host pool as ratio x device pool "
        "(overrides --buffer-size-gb / --cache-ratio) for apples-to-apples runs",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=0,
        help="cap the device KV pool so the working set overflows into the hierarchy",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--attention-backend", type=str, default="")
    parser.add_argument(
        "--storage-backend",
        type=str,
        default="file",
        help="L3 backend: file, or meta_cache_store (spawns a local-mode store "
        "under the run's storage dir; requires --launch-module "
        "sglang_meta.launch_server)",
    )
    parser.add_argument(
        "--sub-question-input-length",
        type=int,
        default=0,
        help="Input tokens per FOLLOW-UP turn (0 = same as --request-length). "
        "Set with a large --request-length for long-context multiturn: "
        "100K first prompt + small follow-ups.",
    )
    parser.add_argument(
        "--prefetch-policy",
        type=str,
        default="wait_complete",
        choices=["best_effort", "wait_complete", "timeout"],
        help="hicache storage prefetch stop policy (server default is "
        "timeout; benches historically used wait_complete)",
    )
    parser.add_argument(
        "--storage-extra-config",
        type=str,
        default="",
        help="JSON merged over the backend's default extra-config "
        '(e.g. \'{"capacity_gb": "120"}\')',
    )
    parser.add_argument(
        "--extra-server-args",
        type=str,
        default="",
        help="space-separated extra args appended to the server command",
    )
    # multiturn shape
    parser.add_argument("--num-clients", type=int, default=12)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--request-length", type=int, default=1024)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--request-rate", type=int, default=4)
    # longctx shape
    parser.add_argument("--prefix-tokens", type=int, default=6144)
    parser.add_argument("--num-prefixes", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--out", type=str, default="bench_buffer_mode_results.json")
    return parser.parse_args()


def scrape_metrics(base_url):
    try:
        text = requests.get(f"{base_url}/metrics", timeout=30).text
    except Exception:
        return {}
    out = {}
    for name in BUFFER_METRICS:
        total, found = 0.0, False
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                total += float(line.rsplit(" ", 1)[1])
                found = True
        if found:
            out[name.split(":")[1]] = total
    return out


class Server:
    def __init__(self, args, mode, storage_dir):
        self.args = args
        self.mode = mode
        self.storage_dir = storage_dir
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.proc = None

    def launch(self):
        a = self.args
        extra_config = {"prefetch_threshold": 64}
        if a.storage_backend == "meta_cache_store":
            # Local-mode store: in-process server on rank 0, data under the
            # run's storage dir (same isolation as the file backend).
            extra_config.update(
                {
                    "cluster": "mks",
                    "local_mode": "true",
                    "data_root": os.path.join(self.storage_dir, "mcs"),
                    "capacity_gb": "120",
                }
            )
        if a.storage_extra_config:
            extra_config.update(json.loads(a.storage_extra_config))
        cmd = [
            sys.executable,
            "-m",
            a.launch_module,
            "--model-path",
            a.model,
            "--port",
            str(a.port),
            "--tp-size",
            str(a.tp),
            "--mem-fraction-static",
            str(a.mem_fraction_static),
            "--page-size",
            str(a.page_size),
            "--enable-hierarchical-cache",
            "--enable-cache-report",
            "--enable-metrics",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-prefetch-policy",
            a.prefetch_policy,
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
        ]
        if self.mode == "buffer":
            cmd += [
                "--hicache-host-memory-mode",
                "buffer_only",
                "--hicache-storage-backend",
                a.storage_backend,
            ]
            if a.host_ratio > 0:
                cmd += ["--hicache-ratio", str(a.host_ratio)]
            else:
                cmd += ["--hicache-size", str(a.buffer_size_gb)]
        else:
            write_policy = "write_back" if self.mode == "cache_wb" else "write_through"
            cmd += [
                "--hicache-ratio",
                str(a.host_ratio if a.host_ratio > 0 else a.cache_ratio),
                "--hicache-storage-backend",
                a.storage_backend,
            ]
            # Override the default write policy appended below.
            cmd = [c if c != "write_through" else write_policy for c in cmd]
        if a.max_total_tokens > 0:
            cmd += ["--max-total-tokens", str(a.max_total_tokens)]
        if a.attention_backend:
            cmd += ["--attention-backend", a.attention_backend]
        if a.extra_server_args:
            cmd += a.extra_server_args.split()
        env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": self.storage_dir,
        }
        log_stem = os.path.splitext(os.path.basename(self.args.out))[0]
        self.log = open(f"/tmp/{log_stem}_{self.mode}_server.log", "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=self.log, env=env)
        deadline = time.time() + 2400
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died during launch; see {self.log.name}")
            try:
                if (
                    requests.get(f"{self.base_url}/health", timeout=5).status_code
                    == 200
                ):
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("server did not become healthy in time")

    def flush(self):
        requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 60}, timeout=90
        ).raise_for_status()

    def kill(self):
        if self.proc is not None:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(self.proc.pid)
            self.proc = None


def run_multiturn(server, args):
    log_file = tempfile.mktemp(suffix=".jsonl")
    cmd = [
        sys.executable,
        os.path.join(BENCH_DIR, "bench_multiturn.py"),
        "--model-path",
        args.model,
        "--port",
        str(args.port),
        "--disable-auto-run",
        "--disable-random-sample",
        "--enable-round-barrier",
        "--num-clients",
        str(args.num_clients),
        "--max-parallel",
        str(args.num_clients),
        "--num-rounds",
        str(args.num_rounds),
        "--request-length",
        str(args.request_length),
        "--output-length",
        str(args.output_length),
        "--request-rate",
        str(args.request_rate),
        "--seed",
        "1",
        "--log-file",
        log_file,
    ]
    if args.sub_question_input_length > 0:
        cmd += [
            "--sub-question-input-length",
            str(args.sub_question_input_length),
        ]
    subprocess.run(cmd, check=True, timeout=7200)
    with open(log_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    record = records[-1]
    result = {
        "hit_rate": record["summary"]["cache_hit_rate"],
        "avg_ttft_s": record["summary"]["average_ttft"],
        "p90_ttft_s": record["summary"].get("p90_ttft", 0),
    }
    for round_key, round_data in record.get("round", {}).items():
        result[f"{round_key}_hit"] = round(round_data["cache_hit_rate"], 4)
        result[f"{round_key}_ttft_s"] = round(round_data["average_ttft"], 3)
    return result


def _gen_token_prompt(tokenizer, n_tokens, seed):
    import random as _random

    rng = _random.Random(seed)
    vocab_size = min(tokenizer.vocab_size - 1000, 32000)
    ids = [rng.randrange(1000, vocab_size) for _ in range(n_tokens * 2)]
    text = tokenizer.decode(ids)
    ids = tokenizer.encode(text)[:n_tokens]
    return tokenizer.decode(ids)


def run_longctx(server, args):
    """Long shared prefix + divergent continuations, warm then post-flush."""
    from sglang.benchmark.utils import get_tokenizer

    tokenizer = get_tokenizer(args.model)
    prefixes = [
        _gen_token_prompt(tokenizer, args.prefix_tokens, seed=900 + g)
        for g in range(args.num_prefixes)
    ]
    tails = [
        _gen_token_prompt(tokenizer, 128, seed=9900 + c)
        for c in range(args.continuations)
    ]

    def one_pass(tag):
        latencies, cached, prompt_tokens = [], 0, 0
        for prefix in prefixes:
            for tail in tails:
                start = time.perf_counter()
                res = requests.post(
                    f"{server.base_url}/generate",
                    json={
                        "text": prefix + tail,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    },
                    timeout=600,
                ).json()
                latencies.append(time.perf_counter() - start)
                meta = res.get("meta_info", {})
                cached += int(meta.get("cached_tokens", 0))
                prompt_tokens += int(meta.get("prompt_tokens", 1))
        return {
            f"{tag}_hit_rate": round(cached / max(prompt_tokens, 1), 4),
            f"{tag}_avg_latency_s": round(sum(latencies) / len(latencies), 3),
            f"{tag}_max_latency_s": round(max(latencies), 3),
        }

    result = one_pass("warm")
    # Give write-backs a moment to settle before dropping device state.
    time.sleep(5)
    server.flush()
    result.update(one_pass("replay"))
    return result


def main():
    args = parse_args()
    results = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        storage_dir = tempfile.mkdtemp(prefix=f"bench_buffer_{mode}_")
        server = Server(args, mode, storage_dir)
        print(f"\n=== launching {mode} server ===", flush=True)
        try:
            server.launch()
            mode_result = {}
            for workload in args.workloads.split(","):
                workload = workload.strip()
                print(f"--- {mode}: running {workload} ---", flush=True)
                if workload == "multiturn":
                    mode_result["multiturn"] = run_multiturn(server, args)
                elif workload == "longctx":
                    mode_result["longctx"] = run_longctx(server, args)
                server.flush()
            mode_result["hicache_metrics"] = scrape_metrics(server.base_url)
            results[mode] = mode_result
        finally:
            server.kill()
            shutil.rmtree(storage_dir, ignore_errors=True)

    print("\n===== results =====")
    print(json.dumps(results, indent=2))
    with open(args.out, "w") as f:
        json.dump(
            {"model": args.model, "args": vars(args), "results": results}, f, indent=2
        )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()

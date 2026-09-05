#!/usr/bin/env bash
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#SBATCH --job-name=dsv4-fp4-topk-compare
#SBATCH --partition=Compute-Group01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=dsv4-fp4-topk-compare-%j.out
#
# Manual-only MI355X benchmark. This script is intentionally not registered by
# any CI workflow, and RUN_MANUAL=1 is required to execute it.
#
# It compares three end-to-end server arms on the same exclusive node:
#   legacy      - this SGLang source, fused top-k disabled, 2 GiB logits pool
#   workspace512 - SGLang PR #38086 source, 512 MiB managed logits workspace
#   fused       - this SGLang source, custom AITER ref, streaming fused top-k
#
# The legacy and fused arms default to the checkout containing this script.
# Point WORKSPACE_SGLANG_SOURCE_DIR at a checkout of PR #38086. The fused arm
# materializes AITER_REF from AITER_SOURCE_DIR without changing that checkout,
# mounts the resulting source read-only, copies it inside the ROCm container,
# and installs it there.
#
# Example:
#   export RUN_MANUAL=1
#   export WORKSPACE_SGLANG_SOURCE_DIR=/path/to/sglang-pr-38086
#   export AITER_SOURCE_DIR=/path/to/aiter
#   export AITER_REF=perf/fp4-streaming-topk
#   sbatch scripts/ci/slurm/manual/dsv4_fp4_topk_compare.sh
#
# Useful overrides:
#   IMAGE, MODEL_PATH, MODEL_STORES, RESULT_ROOT, VARIANT_ORDER
#   CONTEXTS, CONCURRENCIES, REPEATS, WARMUP_REPEATS, OUTPUT_LEN, RUN_COLD
#   LEGACY_SGLANG_SOURCE_DIR, FUSED_SGLANG_SOURCE_DIR
#   AITER_INSTALL_CMD, AITER_UPDATE_SUBMODULES, VRAM_SAMPLE_INTERVAL

set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260903}"
MODEL_BASENAME="${MODEL_BASENAME:-DeepSeek-V4-Pro}"
MODEL_STORES="${MODEL_STORES:-/models /data/models /apps/data/models}"
CONTEXTS="${CONTEXTS:-262144 786432}"
CONCURRENCIES="${CONCURRENCIES:-128 256}"
REPEATS="${REPEATS:-3}"
WARMUP_REPEATS="${WARMUP_REPEATS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
RUN_COLD="${RUN_COLD:-1}"
VARIANT_ORDER="${VARIANT_ORDER:-legacy workspace512 fused}"
PORT="${PORT:-30000}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-6000}"
VRAM_SAMPLE_INTERVAL="${VRAM_SAMPLE_INTERVAL:-0.5}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
AITER_REF="${AITER_REF:-HEAD}"
AITER_INSTALL_CMD="${AITER_INSTALL_CMD:-python3 -m pip install --no-deps --no-build-isolation -e .}"
AITER_UPDATE_SUBMODULES="${AITER_UPDATE_SUBMODULES:-1}"

print_log_tail() {
    local path="$1"
    local lines="${2:-200}"
    python3 - "$path" "$lines" <<'PY'
from collections import deque
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.exists():
    print("".join(deque(path.open(errors="replace"), maxlen=int(sys.argv[2]))), end="")
PY
}

write_client() {
    local path="$1"
    cat >"$path" <<'PY'
import argparse
import asyncio
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

import aiohttp


def flush(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/flush_cache", timeout=120):
        pass


def make_body(input_len, output_len):
    payload = {
        "input_ids": [100 + (i % 1024) for i in range(input_len)],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def append_jsonl(path, value):
    with path.open("a") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


async def one(session, url, body):
    started = time.perf_counter()
    async with session.post(
        url, data=body, headers={"content-type": "application/json"}
    ) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
        value = json.loads(text)
    elapsed = time.perf_counter() - started
    meta = value.get("meta_info", {})
    tokens = int(meta.get("completion_tokens") or 0)
    if tokens <= 0:
        tokens = len(meta.get("output_token_logprobs") or [])
    if tokens <= 0:
        tokens = len(value.get("output_ids") or [])
    if tokens <= 0:
        raise RuntimeError(f"Could not read completion token count: {text[:500]}")
    return elapsed, tokens


async def batch(port, body, concurrency):
    timeout = aiohttp.ClientTimeout(total=7200)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        url = f"http://127.0.0.1:{port}/generate"
        started = time.perf_counter()
        results = await asyncio.gather(
            *(one(session, url, body) for _ in range(concurrency))
        )
        wall = time.perf_counter() - started
    latencies = sorted(item[0] for item in results)
    output_tokens = sum(item[1] for item in results)
    p99_index = max(0, min(len(latencies) - 1, math.ceil(len(latencies) * 0.99) - 1))
    return {
        "wall_seconds": wall,
        "request_count": concurrency,
        "request_throughput": concurrency / wall,
        "output_tokens": output_tokens,
        "output_throughput": output_tokens / wall,
        "request_latency_mean": statistics.fmean(latencies),
        "request_latency_p50": statistics.median(latencies),
        "request_latency_p99": latencies[p99_index],
    }


async def main(args):
    output = Path(args.output)
    warmups = Path(args.warmups)
    output.parent.mkdir(parents=True, exist_ok=True)

    for input_len in args.contexts:
        if args.run_cold:
            cold_body = make_body(input_len, 1)
            flush(args.port)
            compile_warmup = await batch(args.port, cold_body, 1)
            compile_warmup.update(
                {"kind": "cold_compile_warmup", "input_len": input_len}
            )
            append_jsonl(warmups, compile_warmup)
            print(json.dumps(compile_warmup), flush=True)

            flush(args.port)
            cold = await batch(args.port, cold_body, 1)
            cold.update(
                {
                    "kind": "cold_prefill",
                    "input_len": input_len,
                    "concurrency": 1,
                    "repeat": 1,
                }
            )
            append_jsonl(output, cold)
            print(json.dumps(cold), flush=True)

        flush(args.port)
        prefix_body = make_body(input_len, 4)
        prefix_warmup = await batch(args.port, prefix_body, 8)
        prefix_warmup.update({"kind": "prefix_warmup", "input_len": input_len})
        append_jsonl(warmups, prefix_warmup)
        print(json.dumps(prefix_warmup), flush=True)

        body = make_body(input_len, args.output_len)
        for concurrency in args.concurrencies:
            for warmup_repeat in range(1, args.warmup_repeats + 1):
                warmup = await batch(args.port, body, concurrency)
                warmup.update(
                    {
                        "kind": "decode_shape_warmup",
                        "input_len": input_len,
                        "concurrency": concurrency,
                        "warmup_repeat": warmup_repeat,
                    }
                )
                append_jsonl(warmups, warmup)
                print(json.dumps(warmup), flush=True)

            for repeat in range(1, args.repeats + 1):
                result = await batch(args.port, body, concurrency)
                result.update(
                    {
                        "kind": "shared_prefix_decode",
                        "input_len": input_len,
                        "concurrency": concurrency,
                        "repeat": repeat,
                    }
                )
                append_jsonl(output, result)
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup-repeats", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--run-cold", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", required=True)
    asyncio.run(main(parser.parse_args()))
PY
}

write_vram_monitor() {
    local path="$1"
    cat >"$path" <<'PY'
import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
USED = re.compile(
    r"GPU\[(?P<gpu>\d+)\].*?VRAM Total Used Memory \(B\):\s*(?P<bytes>\d+)"
)
TOTAL = re.compile(
    r"GPU\[(?P<gpu>\d+)\].*?VRAM Total Memory \(B\):\s*(?P<bytes>\d+)"
)


def sample():
    result = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    text = ANSI.sub("", result.stdout)
    if result.returncode:
        raise RuntimeError(f"rocm-smi exited {result.returncode}: {text[-500:]}")
    totals = {int(m["gpu"]): int(m["bytes"]) for m in TOTAL.finditer(text)}
    used = [(int(m["gpu"]), int(m["bytes"])) for m in USED.finditer(text)]
    if not used:
        raise RuntimeError(f"could not parse VRAM usage: {text[-500:]}")
    return [(gpu, value, totals.get(gpu)) for gpu, value in used]


def main(args):
    output = Path(args.output)
    stop = Path(args.stop)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp_utc", "monotonic_seconds", "gpu", "used_bytes", "total_bytes"]
        )
        stream.flush()
        while not stop.exists():
            try:
                now = dt.datetime.now(dt.timezone.utc).isoformat()
                elapsed = time.monotonic() - started
                for gpu, used, total in sample():
                    writer.writerow([now, f"{elapsed:.6f}", gpu, used, total or ""])
                stream.flush()
            except Exception as exc:
                print(f"[vram-monitor] {exc}", file=sys.stderr, flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stop", required=True)
    parser.add_argument("--interval", type=float, required=True)
    main(parser.parse_args())
PY
}

install_fused_aiter() {
    : "${AITER_EXPECTED_SHA:?}"
    : "${AITER_INSTALL_CMD:?}"
    local runtime_source="/tmp/aiter-${VARIANT}"

    rm -rf "$runtime_source"
    mkdir -p "$runtime_source"
    cp -a /aiter-source/. "$runtime_source/"

    export AITER_USE_SYSTEM_TRITON="${AITER_USE_SYSTEM_TRITON:-1}"
    export MAX_JOBS="${MAX_JOBS:-32}"
    echo "[aiter-install] source=/aiter-source ref=${AITER_REF} sha=${AITER_EXPECTED_SHA}"
    (
        cd "$runtime_source"
        bash -c "$AITER_INSTALL_CMD"
    ) 2>&1 | tee "$RESULT_DIR/aiter_install.log"

    export PYTHONPATH="$runtime_source:${PYTHONPATH:-}"
    python3 - "$RESULT_DIR/aiter_info.json" "$runtime_source" "$AITER_EXPECTED_SHA" <<'PY'
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

output, source, expected_sha = sys.argv[1:]
import aiter
from aiter.ops.flydsl import flydsl_pa_mqa_topk_fp4_prefill

actual_source = os.path.realpath(aiter.__file__)
expected_source = os.path.realpath(source) + os.sep
if not actual_source.startswith(expected_source):
    raise SystemExit(f"AITER did not import from installed source: {actual_source}")
if not callable(flydsl_pa_mqa_topk_fp4_prefill):
    raise SystemExit("AITER fused FP4 streaming top-k entry point is not callable")
actual_sha = subprocess.check_output(
    ["git", "-C", source, "rev-parse", "HEAD"], text=True
).strip()
if actual_sha != expected_sha:
    raise SystemExit(f"AITER SHA mismatch: expected {expected_sha}, got {actual_sha}")
info = {
    "requested_sha": expected_sha,
    "installed_sha": actual_sha,
    "aiter_file": aiter.__file__,
    "aiter_version": importlib.metadata.version("amd-aiter"),
    "fused_fp4_streaming_topk": True,
}
Path(output).write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, sort_keys=True))
PY
}

inside_main() {
    : "${VARIANT:?}"
    : "${MODEL_PATH:?}"
    : "${SOURCE_SHA:?}"
    RESULT_DIR="/run-results/$VARIANT"
    mkdir -p "$RESULT_DIR"

    export HOME=/tmp
    export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
    export PYTHONDONTWRITEBYTECODE=1
    export TOKENIZERS_PARALLELISM=false
    export SGLANG_CACHE_DIR="/tmp/sglang-cache-$VARIANT"
    export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
    export TRITON_CACHE_DIR="$SGLANG_CACHE_DIR/triton"
    export SGLANG_DEFAULT_THINKING=1
    export SGLANG_DSV4_REASONING_EFFORT=high
    export SGLANG_DSV4_FP4_EXPERTS=true
    export SGLANG_DSV4_FP4_LOGITS_FREE_MEM_FRACTION=0.2
    export SGLANG_USE_AITER=1
    export SGLANG_USE_ROCM700A=0
    export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_SET_CPU_AFFINITY=1
    export SGLANG_EXPOSE_OWN_ENV_VARS=1
    export AITER_BF16_FP8_MOE_BOUND=0
    export GPU_MAX_HW_QUEUES=5

    local policy
    case "$VARIANT" in
        legacy)
            policy="legacy_2gib_pooled_logits"
            export SGLANG_DSV4_FP4_FUSED_TOPK=false
            export SGLANG_DSV4_FP4_LOGITS_BUDGET_MB=2048
            ;;
        workspace512)
            policy="pr_38086_512mib_managed_workspace"
            export SGLANG_DSV4_FP4_FUSED_TOPK=false
            export SGLANG_DSV4_FP4_LOGITS_BUDGET_MB=512
            ;;
        fused)
            policy="aiter_streaming_fused_topk"
            export SGLANG_DSV4_FP4_FUSED_TOPK=true
            export SGLANG_DSV4_FP4_LOGITS_BUDGET_MB=2048
            install_fused_aiter
            ;;
        *)
            echo "ERROR: unknown variant: $VARIANT" >&2
            return 2
            ;;
    esac

    python3 - "$RESULT_DIR/runtime.json" "$VARIANT" "$policy" "$SOURCE_SHA" <<'PY'
import json
import os
import sys
from pathlib import Path

output, variant, policy, source_sha = sys.argv[1:]
data = {
    "variant": variant,
    "policy": policy,
    "source_sha": source_sha,
    "image": os.environ["IMAGE"],
    "contexts": [int(x) for x in os.environ["CONTEXTS"].split()],
    "concurrencies": [int(x) for x in os.environ["CONCURRENCIES"].split()],
    "repeats": int(os.environ["REPEATS"]),
    "warmup_repeats": int(os.environ["WARMUP_REPEATS"]),
    "output_len": int(os.environ["OUTPUT_LEN"]),
    "run_cold": bool(int(os.environ["RUN_COLD"])),
    "configured_logits_workspace_mib": int(
        os.environ["SGLANG_DSV4_FP4_LOGITS_BUDGET_MB"]
    ),
    "fused_topk_requested": os.environ["SGLANG_DSV4_FP4_FUSED_TOPK"] == "true",
}
Path(output).write_text(json.dumps(data, indent=2) + "\n")
PY
    env | LC_ALL=C sort | grep -E \
        '^(AITER_|GPU_MAX_HW_QUEUES|HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|SGLANG_)' \
        >"$RESULT_DIR/runtime_env.txt"

    write_client "$RESULT_DIR/client.py"
    write_vram_monitor "$RESULT_DIR/vram_monitor.py"

    local server_pid=""
    local vram_pid=""
    local vram_stop="$RESULT_DIR/vram_monitor.stop"
    rm -f "$vram_stop"

    stop_vram_monitor() {
        if [[ -n "${vram_pid:-}" ]]; then
            touch "$vram_stop"
            wait "$vram_pid" 2>/dev/null || true
            vram_pid=""
        fi
    }

    cleanup_server() {
        if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM -- "-$server_pid" 2>/dev/null || true
            for _ in $(seq 1 60); do
                kill -0 "$server_pid" 2>/dev/null || break
                sleep 2
            done
            kill -KILL -- "-$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
        fi
        server_pid=""
    }

    cleanup_inside() {
        stop_vram_monitor
        cleanup_server
    }
    trap cleanup_inside EXIT INT TERM

    rocm-smi --showmeminfo vram >"$RESULT_DIR/rocm_smi_before.txt" 2>&1 || true
    python3 "$RESULT_DIR/vram_monitor.py" \
        --output "$RESULT_DIR/vram_samples.csv" \
        --stop "$vram_stop" \
        --interval "$VRAM_SAMPLE_INTERVAL" \
        >"$RESULT_DIR/vram_monitor.stdout.log" \
        2>"$RESULT_DIR/vram_monitor.stderr.log" &
    vram_pid=$!

    local -a server_cmd=(
        python3 -m sglang.launch_server
        --model-path "$MODEL_PATH"
        --served-model-name DeepSeek-V4-Pro
        --trust-remote-code
        --host 0.0.0.0
        --port "$PORT"
        --tensor-parallel-size 8
        --dp-size 8
        --enable-dp-attention
        --moe-dense-tp-size 1
        --enable-dp-lm-head
        --attention-backend dsv4
        --page-size 256
        --swa-full-tokens-ratio 0.10
        --kv-cache-dtype fp8_e4m3
        --enforce-shared-experts-fusion
        --chunked-prefill-size 8192
        --context-length 1048576
        --mem-fraction-static 0.85
        --max-running-requests 256
        --cuda-graph-max-bs-decode 256
        --speculative-algorithm EAGLE
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
        --enable-deepseek-v4-fp4-indexer
        --random-seed 20260904
        --decode-log-interval 40
        --watchdog-timeout 7200
        --enable-metrics
    )
    printf '%q ' "${server_cmd[@]}" >"$RESULT_DIR/server_command.txt"
    printf '\n' >>"$RESULT_DIR/server_command.txt"
    echo "[$VARIANT] starting server ($policy)"
    setsid "${server_cmd[@]}" >"$RESULT_DIR/server.log" 2>&1 &
    server_pid=$!

    local ready=0
    local checks=$((READY_TIMEOUT_SECONDS / 5))
    if ((checks < 1)); then
        checks=1
    fi
    for _ in $(seq 1 "$checks"); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR: $VARIANT server exited during startup" >&2
            print_log_tail "$RESULT_DIR/server.log"
            return 1
        fi
        sleep 5
    done
    if [[ "$ready" != 1 ]]; then
        echo "ERROR: $VARIANT server did not become healthy" >&2
        print_log_tail "$RESULT_DIR/server.log"
        return 1
    fi

    case "$VARIANT" in
        legacy)
            grep -Fq "AITER FP4 streaming top-k: legacy fallback" \
                "$RESULT_DIR/server.log" || {
                echo "ERROR: legacy arm did not select the materialized-logits path" >&2
                print_log_tail "$RESULT_DIR/server.log"
                return 1
            }
            ;;
        workspace512)
            grep -Fq "DSV4 FP4 logits workspace:" "$RESULT_DIR/server.log" || {
                echo "ERROR: workspace512 arm did not report PR #38086 workspace" >&2
                print_log_tail "$RESULT_DIR/server.log"
                return 1
            }
            ;;
        fused)
            grep -Fq "AITER FP4 streaming top-k: enabled" "$RESULT_DIR/server.log" || {
                echo "ERROR: fused arm fell back to materialized logits" >&2
                print_log_tail "$RESULT_DIR/server.log"
                return 1
            }
            ;;
    esac

    curl -fsS "http://127.0.0.1:${PORT}/server_info" \
        >"$RESULT_DIR/server_info.json"
    curl -fsS "http://127.0.0.1:${PORT}/metrics" \
        >"$RESULT_DIR/metrics_start.txt"
    python3 "$RESULT_DIR/client.py" \
        --port "$PORT" \
        --contexts $CONTEXTS \
        --concurrencies $CONCURRENCIES \
        --repeats "$REPEATS" \
        --warmup-repeats "$WARMUP_REPEATS" \
        --output-len "$OUTPUT_LEN" \
        --run-cold "$RUN_COLD" \
        --output "$RESULT_DIR/results.jsonl" \
        --warmups "$RESULT_DIR/warmups.jsonl" \
        2>&1 | tee "$RESULT_DIR/client.log"
    curl -fsS "http://127.0.0.1:${PORT}/metrics" \
        >"$RESULT_DIR/metrics_end.txt"
    rocm-smi --showmeminfo vram >"$RESULT_DIR/rocm_smi_after.txt" 2>&1 || true

    stop_vram_monitor
    cleanup_server
    trap - EXIT INT TERM

    python3 - "$RESULT_DIR/vram_samples.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("VRAM monitor produced no parseable samples")
print(f"[vram-monitor] captured {len(rows)} device samples")
PY
}

wait_for_gpu_drain() {
    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "WARN: host rocm-smi unavailable; skipping between-arm drain check" >&2
        return
    fi
    local attempt max_vram
    for attempt in $(seq 1 90); do
        max_vram="$(
            rocm-smi --showmemuse 2>/dev/null |
                awk '/GPU Memory Allocated \(VRAM%\)/ {if ($NF + 0 > max) max = $NF + 0} END {print max + 0}'
        )"
        if [[ "$max_vram" =~ ^[0-9]+$ ]] && ((max_vram <= 10)); then
            echo "[gpu-drain] GPUs ready (maximum VRAM use ${max_vram}%)"
            return
        fi
        echo "[gpu-drain] maximum VRAM use ${max_vram:-unknown}% (attempt ${attempt}/90)"
        sleep 10
    done
    echo "ERROR: GPUs did not drain below 10% VRAM use" >&2
    return 1
}

contains_variant() {
    local expected="$1"
    local variant
    for variant in $VARIANT_ORDER; do
        if [[ "$variant" == "$expected" ]]; then
            return 0
        fi
    done
    return 1
}

validate_source_for_variant() {
    local variant="$1"
    local source="$2"
    local adapter="$source/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py"

    if [[ ! -d "$source/python/sglang" ]] || ! git -C "$source" rev-parse HEAD >/dev/null 2>&1; then
        echo "ERROR: invalid SGLang source for $variant: $source" >&2
        return 2
    fi
    case "$variant" in
        legacy)
            grep -Fq 'SGLANG_DSV4_FP4_LOGITS_BUDGET_MB", "2048"' "$adapter" &&
                grep -Fq "SGLANG_DSV4_FP4_FUSED_TOPK" \
                    "$source/python/sglang/srt/environ.py" || {
                echo "ERROR: legacy source lacks the 2 GiB pool/fused-path kill switch" >&2
                return 2
            }
            ;;
        workspace512)
            [[ -f "$source/python/sglang/srt/layers/attention/dsv4/fp4_logits_workspace.py" ]] ||
                {
                    echo "ERROR: workspace source does not look like SGLang PR #38086" >&2
                    return 2
                }
            ;;
        fused)
            grep -Fq "aiter_fp4_paged_mqa_topk" "$adapter" &&
                grep -Fq "SGLANG_DSV4_FP4_FUSED_TOPK" \
                    "$source/python/sglang/srt/environ.py" || {
                echo "ERROR: fused source lacks the AITER streaming top-k path" >&2
                return 2
            }
            ;;
        *)
            echo "ERROR: unknown variant in VARIANT_ORDER: $variant" >&2
            return 2
            ;;
    esac
}

record_source_provenance() {
    local variant="$1"
    local source="$2"
    local arm="$3"
    local sha ref diff_sha

    sha="$(git -C "$source" rev-parse HEAD)"
    ref="$(git -C "$source" symbolic-ref --short -q HEAD || true)"
    git -C "$source" status --short >"$arm/source_status.txt"
    git -C "$source" diff HEAD --binary >"$arm/source.patch"
    diff_sha="$(sha256sum "$arm/source.patch" | awk '{print $1}')"
    python3 - "$arm/provenance.json" "$variant" "$source" "$sha" "$ref" "$diff_sha" <<'PY'
import json
import sys
from pathlib import Path

output, variant, source, sha, ref, diff_sha = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {
            "variant": variant,
            "source_dir": source,
            "source_sha": sha,
            "source_ref": ref or None,
            "source_diff_sha256": diff_sha,
        },
        indent=2,
    )
    + "\n"
)
PY
}

AITER_STAGE_PARENT=""
AITER_STAGE_DIR=""
AITER_EXPECTED_SHA=""
ACTIVE_CONTAINER=""
DOCKER_CMD=()

prepare_aiter_checkout() {
    : "${AITER_SOURCE_DIR:?Set AITER_SOURCE_DIR for the fused arm}"
    if [[ ! -d "$AITER_SOURCE_DIR/aiter" ]] ||
        ! git -C "$AITER_SOURCE_DIR" rev-parse HEAD >/dev/null 2>&1; then
        echo "ERROR: invalid AITER source: $AITER_SOURCE_DIR" >&2
        return 2
    fi

    AITER_EXPECTED_SHA="$(
        git -C "$AITER_SOURCE_DIR" rev-parse --verify "${AITER_REF}^{commit}"
    )" || {
        echo "ERROR: AITER_REF is not available in AITER_SOURCE_DIR: $AITER_REF" >&2
        return 2
    }
    AITER_STAGE_PARENT="$(
        mktemp -d "${SLURM_TMPDIR:-/tmp}/dsv4-aiter-${SLURM_JOB_ID}.XXXXXXXX"
    )"
    AITER_STAGE_DIR="$AITER_STAGE_PARENT/source"

    {
        echo "[aiter-stage] source=$AITER_SOURCE_DIR"
        echo "[aiter-stage] ref=$AITER_REF"
        echo "[aiter-stage] sha=$AITER_EXPECTED_SHA"
        git clone --local --no-checkout "$AITER_SOURCE_DIR" "$AITER_STAGE_DIR"
        git -C "$AITER_STAGE_DIR" checkout --detach "$AITER_EXPECTED_SHA"
        if [[ "$AITER_UPDATE_SUBMODULES" == 1 && -f "$AITER_STAGE_DIR/.gitmodules" ]]; then
            while read -r key path; do
                local name="${key#submodule.}"
                name="${name%.path}"
                if git -C "$AITER_SOURCE_DIR/$path" rev-parse HEAD >/dev/null 2>&1; then
                    git -C "$AITER_STAGE_DIR" config \
                        "submodule.$name.url" "$AITER_SOURCE_DIR/$path"
                fi
            done < <(
                git -C "$AITER_STAGE_DIR" config -f .gitmodules \
                    --get-regexp '^submodule\..*\.path$'
            )
            git -c protocol.file.allow=always -C "$AITER_STAGE_DIR" \
                submodule update --init --recursive
        fi
        git -C "$AITER_STAGE_DIR" submodule status --recursive || true
    } >"$RESULT_DIR/aiter_stage.log" 2>&1 || {
        cat "$RESULT_DIR/aiter_stage.log" >&2
        return 2
    }
    cat "$RESULT_DIR/aiter_stage.log"

    grep -Rqs "flydsl_pa_mqa_topk_fp4_prefill" "$AITER_STAGE_DIR/aiter" || {
        echo "ERROR: selected AITER ref lacks flydsl_pa_mqa_topk_fp4_prefill" >&2
        return 2
    }
    git -C "$AITER_SOURCE_DIR" status --short >"$RESULT_DIR/aiter_source_status.txt"
    git -C "$AITER_SOURCE_DIR" diff HEAD --binary >"$RESULT_DIR/aiter_source.patch"
    printf '%s\n' "$AITER_REF" >"$RESULT_DIR/aiter_ref.txt"
    printf '%s\n' "$AITER_EXPECTED_SHA" >"$RESULT_DIR/aiter_sha.txt"
}

host_cleanup() {
    if [[ -n "${ACTIVE_CONTAINER:-}" && "${#DOCKER_CMD[@]}" -gt 0 ]]; then
        "${DOCKER_CMD[@]}" rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    fi
    if [[ -n "${AITER_STAGE_PARENT:-}" && -d "$AITER_STAGE_PARENT" ]]; then
        rm -rf "$AITER_STAGE_PARENT"
    fi
}

resolve_model() {
    if [[ -n "${MODEL_PATH:-}" ]]; then
        if [[ ! -d "$MODEL_PATH" ]]; then
            echo "ERROR: MODEL_PATH is not a directory: $MODEL_PATH" >&2
            return 2
        fi
        MODEL_PATH="$(readlink -f "$MODEL_PATH")"
        case "$MODEL_PATH" in
            */snapshots/*) MODEL_MOUNT="${MODEL_PATH%%/snapshots/*}" ;;
            *) MODEL_MOUNT="$(dirname "$MODEL_PATH")" ;;
        esac
        return
    fi

    local store
    for store in $MODEL_STORES; do
        if [[ -d "$store/$MODEL_BASENAME" ]]; then
            MODEL_PATH="$store/$MODEL_BASENAME"
            MODEL_MOUNT="$store"
            return
        fi
    done
    echo "ERROR: $MODEL_BASENAME not found in MODEL_STORES=$MODEL_STORES" >&2
    return 2
}

run_variant() {
    local variant="$1"
    local container="$2"
    local source arm source_sha source_diff_sha
    case "$variant" in
        legacy) source="$LEGACY_SGLANG_SOURCE_DIR" ;;
        workspace512) source="$WORKSPACE_SGLANG_SOURCE_DIR" ;;
        fused) source="$FUSED_SGLANG_SOURCE_DIR" ;;
        *)
            echo "ERROR: unknown variant: $variant" >&2
            return 2
            ;;
    esac

    arm="$RESULT_DIR/$variant"
    mkdir -p "$arm"
    chmod 0777 "$arm"
    validate_source_for_variant "$variant" "$source"
    record_source_provenance "$variant" "$source" "$arm"
    source_sha="$(git -C "$source" rev-parse HEAD)"
    source_diff_sha="$(
        git -C "$source" diff HEAD --binary | sha256sum | awk '{print $1}'
    )"

    local -a cmd=(
        "${DOCKER_CMD[@]}" run --rm
        --name "$container"
        --init
        --privileged
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --network host
        --ipc host
        --shm-size 128g
        --ulimit memlock=-1
        --ulimit stack=67108864
        --ulimit nofile=1048576:1048576
        --security-opt seccomp=unconfined
        --cap-add SYS_PTRACE
        -e "IMAGE=$IMAGE"
        -e "VARIANT=$variant"
        -e "SOURCE_SHA=$source_sha"
        -e "MODEL_PATH=$MODEL_PATH"
        -e "CONTEXTS=$CONTEXTS"
        -e "CONCURRENCIES=$CONCURRENCIES"
        -e "REPEATS=$REPEATS"
        -e "WARMUP_REPEATS=$WARMUP_REPEATS"
        -e "OUTPUT_LEN=$OUTPUT_LEN"
        -e "RUN_COLD=$RUN_COLD"
        -e "PORT=$PORT"
        -e "READY_TIMEOUT_SECONDS=$READY_TIMEOUT_SECONDS"
        -e "VRAM_SAMPLE_INTERVAL=$VRAM_SAMPLE_INTERVAL"
        -e HOME=/tmp
        -v /sys:/sys:ro
        -v "$source:/sgl-workspace/sglang:ro"
        -v "$MODEL_MOUNT:$MODEL_MOUNT:ro"
        -v "$RESULT_DIR:/run-results"
    )
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        cmd+=(-e "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES")
    fi
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
        cmd+=(-e "HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
    fi
    if [[ "$variant" == fused ]]; then
        cmd+=(
            -e "AITER_REF=$AITER_REF"
            -e "AITER_EXPECTED_SHA=$AITER_EXPECTED_SHA"
            -e "AITER_INSTALL_CMD=$AITER_INSTALL_CMD"
            -v "$AITER_STAGE_DIR:/aiter-source:ro"
        )
    fi
    cmd+=(--entrypoint bash "$IMAGE" /run-results/harness.sh --inside)

    printf '%q ' "${cmd[@]}" >"$arm/docker_command.txt"
    printf '\n' >>"$arm/docker_command.txt"
    local run_status
    set +e
    "${cmd[@]}" 2>&1 | tee "$arm/container.log"
    run_status=${PIPESTATUS[0]}
    set -e

    local source_sha_after source_diff_sha_after
    source_sha_after="$(git -C "$source" rev-parse HEAD)"
    source_diff_sha_after="$(
        git -C "$source" diff HEAD --binary | sha256sum | awk '{print $1}'
    )"
    if [[ "$source_sha_after" != "$source_sha" ||
        "$source_diff_sha_after" != "$source_diff_sha" ]]; then
        echo "ERROR: SGLang source changed while the $variant arm was running" >&2
        run_status=3
    fi
    printf '%s\n' "$run_status" >"$arm/exit_code.txt"
    return "$run_status"
}

summarize_results() {
    python3 - "$RESULT_DIR" "$VARIANT_ORDER" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
variants = sys.argv[2].split()
summary = []


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def max_total_tokens(info):
    if not info:
        return None
    if info.get("max_total_num_tokens") is not None:
        return info["max_total_num_tokens"]
    for state in info.get("internal_states", []):
        if state.get("max_total_num_tokens") is not None:
            return state["max_total_num_tokens"]
    return None


for variant in variants:
    arm = root / variant
    if not arm.exists():
        continue
    runtime = read_json(arm / "runtime.json") or {}
    provenance = read_json(arm / "provenance.json") or {}
    info = read_json(arm / "server_info.json")
    log = (arm / "server.log").read_text(errors="replace") if (arm / "server.log").exists() else ""
    workspace_values = [
        float(value)
        for value in re.findall(r"DSV4 FP4 logits workspace: ([0-9.]+) MiB", log)
    ]

    rows = []
    if (arm / "results.jsonl").exists():
        rows = [
            json.loads(line)
            for line in (arm / "results.jsonl").read_text().splitlines()
            if line.strip()
        ]
    groups = {}
    for row in rows:
        key = (row["kind"], row["input_len"], row["concurrency"])
        groups.setdefault(key, []).append(row)
    measurements = []
    for key, values in sorted(groups.items()):
        measurements.append(
            {
                "kind": key[0],
                "input_len": key[1],
                "concurrency": key[2],
                "repeats": len(values),
                "output_throughput_median": statistics.median(
                    value["output_throughput"] for value in values
                ),
                "request_throughput_median": statistics.median(
                    value["request_throughput"] for value in values
                ),
                "wall_seconds_median": statistics.median(
                    value["wall_seconds"] for value in values
                ),
                "request_latency_mean_median": statistics.median(
                    value["request_latency_mean"] for value in values
                ),
                "request_latency_p50_median": statistics.median(
                    value["request_latency_p50"] for value in values
                ),
                "request_latency_p99_median": statistics.median(
                    value["request_latency_p99"] for value in values
                ),
            }
        )

    peak_by_gpu = {}
    samples = arm / "vram_samples.csv"
    if samples.exists():
        with samples.open(newline="") as stream:
            for row in csv.DictReader(stream):
                gpu = int(row["gpu"])
                used = int(row["used_bytes"])
                peak_by_gpu[gpu] = max(peak_by_gpu.get(gpu, 0), used)
    exit_code = None
    if (arm / "exit_code.txt").exists():
        exit_code = int((arm / "exit_code.txt").read_text().strip())
    summary.append(
        {
            "variant": variant,
            "policy": runtime.get("policy"),
            "exit_code": exit_code,
            "source_sha": provenance.get("source_sha"),
            "source_ref": provenance.get("source_ref"),
            "max_total_num_tokens": max_total_tokens(info),
            "configured_logits_workspace_mib": runtime.get(
                "configured_logits_workspace_mib"
            ),
            "reported_workspace_mib_per_rank": (
                statistics.median(workspace_values) if workspace_values else None
            ),
            "fused_topk_enabled": "AITER FP4 streaming top-k: enabled" in log,
            "peak_vram_bytes_max_gpu": max(peak_by_gpu.values(), default=None),
            "peak_vram_mib_max_gpu": (
                max(peak_by_gpu.values()) / 2**20 if peak_by_gpu else None
            ),
            "peak_vram_bytes_by_gpu": {
                str(gpu): value for gpu, value in sorted(peak_by_gpu.items())
            },
            "measurements": measurements,
        }
    )

(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = ["# DeepSeek-V4 FP4 top-k comparison", ""]
for arm in summary:
    lines.extend(
        [
            f"## {arm['variant']}",
            f"- Policy: {arm['policy']}",
            f"- Exit code: {arm['exit_code']}",
            f"- SGLang SHA: {arm['source_sha']}",
            f"- max_total_num_tokens: {arm['max_total_num_tokens']}",
            f"- Reported workspace MiB/rank: {arm['reported_workspace_mib_per_rank']}",
            f"- Peak VRAM MiB (maximum GPU): {arm['peak_vram_mib_max_gpu']}",
            f"- Fused top-k enabled: {arm['fused_topk_enabled']}",
        ]
    )
    for row in arm["measurements"]:
        lines.append(
            "- "
            f"{row['kind']} ISL={row['input_len']} concurrency={row['concurrency']} "
            f"n={row['repeats']}: output={row['output_throughput_median']:.2f} tok/s, "
            f"latency p50={row['request_latency_p50_median']:.3f}s, "
            f"p99={row['request_latency_p99_median']:.3f}s"
        )
    lines.append("")
(root / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
PY
}

print_usage() {
    cat >&2 <<'EOF'
This is a disabled, manual-only benchmark. Run it through Slurm after setting:

  export RUN_MANUAL=1
  export WORKSPACE_SGLANG_SOURCE_DIR=/checkout/of/sglang-pr-38086
  export AITER_SOURCE_DIR=/checkout/of/aiter
  export AITER_REF=<branch-tag-or-sha>
  sbatch scripts/ci/slurm/manual/dsv4_fp4_topk_compare.sh

No job was submitted by this script.
EOF
}

host_main() {
    if [[ "${RUN_MANUAL:-0}" != 1 ]]; then
        print_usage
        return 2
    fi
    : "${SLURM_JOB_ID:?Submit this manual harness with sbatch}"

    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
    FUSED_SGLANG_SOURCE_DIR="${FUSED_SGLANG_SOURCE_DIR:-$repo_root}"
    LEGACY_SGLANG_SOURCE_DIR="${LEGACY_SGLANG_SOURCE_DIR:-$FUSED_SGLANG_SOURCE_DIR}"
    WORKSPACE_SGLANG_SOURCE_DIR="${WORKSPACE_SGLANG_SOURCE_DIR:-}"
    RESULT_ROOT="${RESULT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/dsv4-fp4-topk-compare-results}"
    RESULT_DIR="$RESULT_ROOT/$SLURM_JOB_ID"
    export AITER_REF AITER_SOURCE_DIR CONTEXTS CONCURRENCIES
    export FUSED_SGLANG_SOURCE_DIR IMAGE LEGACY_SGLANG_SOURCE_DIR
    export OUTPUT_LEN REPEATS RESULT_DIR RUN_COLD VARIANT_ORDER WARMUP_REPEATS
    export WORKSPACE_SGLANG_SOURCE_DIR

    local -A seen=()
    local variant
    for variant in $VARIANT_ORDER; do
        case "$variant" in
            legacy|workspace512|fused) ;;
            *)
                echo "ERROR: invalid variant in VARIANT_ORDER: $variant" >&2
                return 2
                ;;
        esac
        if [[ -n "${seen[$variant]:-}" ]]; then
            echo "ERROR: duplicate variant in VARIANT_ORDER: $variant" >&2
            return 2
        fi
        seen[$variant]=1
    done
    if [[ "${#seen[@]}" -eq 0 ]]; then
        echo "ERROR: VARIANT_ORDER is empty" >&2
        return 2
    fi
    if contains_variant workspace512 && [[ -z "$WORKSPACE_SGLANG_SOURCE_DIR" ]]; then
        echo "ERROR: set WORKSPACE_SGLANG_SOURCE_DIR to a PR #38086 checkout" >&2
        return 2
    fi
    if contains_variant fused && [[ -z "${AITER_SOURCE_DIR:-}" ]]; then
        echo "ERROR: set AITER_SOURCE_DIR and optionally AITER_REF for fused" >&2
        return 2
    fi

    resolve_model
    export MODEL_PATH
    mkdir -p "$RESULT_DIR"
    chmod 0777 "$RESULT_DIR"
    cp "${BASH_SOURCE[0]}" "$RESULT_DIR/harness.sh"
    chmod 0755 "$RESULT_DIR/harness.sh"

    if docker ps >/dev/null 2>&1; then
        DOCKER_CMD=(docker)
    else
        DOCKER_CMD=(sudo -n docker)
        "${DOCKER_CMD[@]}" ps >/dev/null
    fi
    "${DOCKER_CMD[@]}" image inspect "$IMAGE" >"$RESULT_DIR/image_inspect.json" 2>/dev/null ||
        "${DOCKER_CMD[@]}" pull "$IMAGE"
    "${DOCKER_CMD[@]}" image inspect "$IMAGE" >"$RESULT_DIR/image_inspect.json"

    trap host_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if contains_variant fused; then
        prepare_aiter_checkout
    fi

    python3 - "$RESULT_DIR/configuration.json" <<'PY'
import json
import os
from pathlib import Path

keys = [
    "IMAGE",
    "MODEL_PATH",
    "CONTEXTS",
    "CONCURRENCIES",
    "REPEATS",
    "WARMUP_REPEATS",
    "OUTPUT_LEN",
    "RUN_COLD",
    "VARIANT_ORDER",
    "LEGACY_SGLANG_SOURCE_DIR",
    "WORKSPACE_SGLANG_SOURCE_DIR",
    "FUSED_SGLANG_SOURCE_DIR",
    "AITER_SOURCE_DIR",
    "AITER_REF",
]
Path(os.sys.argv[1]).write_text(
    json.dumps({key: os.environ.get(key) for key in keys}, indent=2) + "\n"
)
PY

    wait_for_gpu_drain
    local overall_status=0
    local arm_status container
    for variant in $VARIANT_ORDER; do
        container="dsv4-fp4-topk-${SLURM_JOB_ID}-${variant}"
        ACTIVE_CONTAINER="$container"
        "${DOCKER_CMD[@]}" rm -f "$container" >/dev/null 2>&1 || true

        set +e
        (
            set -euo pipefail
            run_variant "$variant" "$container"
        )
        arm_status=$?
        set -e

        "${DOCKER_CMD[@]}" rm -f "$container" >/dev/null 2>&1 || true
        ACTIVE_CONTAINER=""
        wait_for_gpu_drain || arm_status=$?
        if [[ "$arm_status" -ne 0 ]]; then
            overall_status="$arm_status"
            echo "ERROR: $variant arm failed with status $arm_status" >&2
            if [[ "$CONTINUE_ON_FAILURE" != 1 ]]; then
                break
            fi
        fi
    done

    summarize_results || overall_status=$?
    echo "Results: $RESULT_DIR"
    return "$overall_status"
}

if [[ "${1:-}" == "--inside" ]]; then
    inside_main
else
    host_main
fi

#!/bin/bash
set -euo pipefail

export PYTHONPATH=/workspace/sglang/python
export HOME=/host_scratch/.wf_megamoe_home
export USER=${USER:-xutingz}
export LOGNAME=${LOGNAME:-xutingz}
export HF_HOME=/host_scratch/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache_megamoe_mmlu_tput_${SLURM_JOB_ID:-manual}
export TORCHINDUCTOR_CACHE_DIR=/host_scratch/.cache/torchinductor
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export OPENAI_API_KEY=EMPTY
export PYTHONUNBUFFERED=1

MODEL=${MODEL:-/host_scratch/model/DeepSeek-V4-Flash}
RESULT_ROOT=${RESULT_ROOT:-/host_scratch/megamoe_mmlu_tput_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}
MMLU_FILE=${MMLU_FILE:-/host_scratch/bench/waterfill/mmlu_bench_2k.json}
PORT=${PORT:-30000}
CASE_PORT_STRIDE=${CASE_PORT_STRIDE:-1000}

TP_SIZE=${TP_SIZE:-2}
DP_SIZE=${DP_SIZE:-1}
MOE_DENSE_TP_SIZE=${MOE_DENSE_TP_SIZE:-1}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-2048}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:--1}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8_e4m3}
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-megamoe}
MEGA_MOE_CAP=${MEGA_MOE_CAP:-8320}
MEGA_MOE_CAP_BUCKETS_AUTO=${MEGA_MOE_CAP_BUCKETS_AUTO:-0}
MEGA_MOE_CAP_BUCKETS=${MEGA_MOE_CAP_BUCKETS:-}
EP_NUM_REDUNDANT_EXPERTS=${EP_NUM_REDUNDANT_EXPERTS:-0}
EP_DISPATCH_ALGORITHM=${EP_DISPATCH_ALGORITHM:-}
INIT_EXPERT_LOCATION=${INIT_EXPERT_LOCATION:-}
ENABLE_EPLB=${ENABLE_EPLB:-0}
EPLB_ALGORITHM=${EPLB_ALGORITHM:-}
EPLB_REBALANCE_NUM_ITERATIONS=${EPLB_REBALANCE_NUM_ITERATIONS:-}
EPLB_REBALANCE_LAYERS_PER_CHUNK=${EPLB_REBALANCE_LAYERS_PER_CHUNK:-}
EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD=${EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD:-}
EXPERT_DISTRIBUTION_RECORDER_MODE=${EXPERT_DISTRIBUTION_RECORDER_MODE:-}
EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE=${EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE:-}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
SKIP_SERVER_WARMUP=${SKIP_SERVER_WARMUP:-1}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
ENFORCE_SHARED_EXPERTS_FUSION=${ENFORCE_SHARED_EXPERTS_FUSION:-1}

TPUT_WARMUP_ROUNDS=${TPUT_WARMUP_ROUNDS:-4}
TPUT_MEASURE_ROUNDS=${TPUT_MEASURE_ROUNDS:-8}
TPUT_SAMPLE_SIZE=${TPUT_SAMPLE_SIZE:-1000}
TPUT_CONCURRENCY=${TPUT_CONCURRENCY:-256}
TPUT_MAX_TOKENS=${TPUT_MAX_TOKENS:-1}
TPUT_TIMEOUT_SEC=${TPUT_TIMEOUT_SEC:-300}
TPUT_SEED=${TPUT_SEED:-}
TPUT_CLIENT_TIMEOUT_SEC=${TPUT_CLIENT_TIMEOUT_SEC:-3600}

PROFILE_ENABLE=${PROFILE_ENABLE:-0}
PROFILE_ACTIVITIES=${PROFILE_ACTIVITIES:-GPU}
PROFILE_BY_STAGE=${PROFILE_BY_STAGE:-1}
PROFILE_NUM_STEPS=${PROFILE_NUM_STEPS:-5}
PROFILE_STAGES=${PROFILE_STAGES:-prefill}

CASE_ORDER=${CASE_ORDER:-fused,fused_waterfill}
WAIT_HEALTH_ATTEMPTS=${WAIT_HEALTH_ATTEMPTS:-1200}
WAIT_READY_MODE=${WAIT_READY_MODE:-http}
DEEPGEMM_QUIET_WAIT_SEC=${DEEPGEMM_QUIET_WAIT_SEC:-0}
DEEPGEMM_QUIET_STABLE_SEC=${DEEPGEMM_QUIET_STABLE_SEC:-60}

export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="$MEGA_MOE_CAP"
if [ "$MEGA_MOE_CAP_BUCKETS_AUTO" = "1" ]; then
  unset SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS
else
  export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS="$MEGA_MOE_CAP_BUCKETS"
fi
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS:-0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB:-4.0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS:-0}
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND:-0}
export SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL=${SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL:-0}
export SGLANG_MEGA_MOE_LOG_ALL_RANKS=${SGLANG_MEGA_MOE_LOG_ALL_RANKS:-0}
export SGLANG_MEGA_MOE_LOG_TOPK_STATS_INTERVAL=${SGLANG_MEGA_MOE_LOG_TOPK_STATS_INTERVAL:-0}
export SGLANG_LOG_EXPERT_LOCATION_METADATA=${SGLANG_LOG_EXPERT_LOCATION_METADATA:-0}
export SGLANG_DISABLE_STATIC_WATERFILL=${SGLANG_DISABLE_STATIC_WATERFILL:-0}
export SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD=${SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD:-0}
export SGLANG_WATERFILL_LOG_STATS_INTERVAL=${SGLANG_WATERFILL_LOG_STATS_INTERVAL:-0}
export SGLANG_WATERFILL_LOG_ALL_RANKS=${SGLANG_WATERFILL_LOG_ALL_RANKS:-0}
export SGLANG_WATERFILL_FORCE_LOCAL_SHARED=${SGLANG_WATERFILL_FORCE_LOCAL_SHARED:-0}
export SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS=${SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS:-1}
export SGLANG_WATERFILL_LOCAL_PREF_NUMER=${SGLANG_WATERFILL_LOCAL_PREF_NUMER:-11}
export SGLANG_WATERFILL_LOCAL_PREF_DENOM=${SGLANG_WATERFILL_LOCAL_PREF_DENOM:-10}
export SGLANG_WATERFILL_REMOTE_COST_TOKENS=${SGLANG_WATERFILL_REMOTE_COST_TOKENS:-0}
export SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED=${SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED:-}
export SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD=${SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD:-0}
export SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M=${SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M:-0}
export SGLANG_WATERFILL_REUSE_TOPK_BUFFER=${SGLANG_WATERFILL_REUSE_TOPK_BUFFER:-0}
export SGLANG_WATERFILL_REUSE_TOPK_BUFFER_CACHE_SIZE=${SGLANG_WATERFILL_REUSE_TOPK_BUFFER_CACHE_SIZE:-8}
export SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE=${SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE:-64}
if [ -n "${SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH+x}" ]; then
  export SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH
fi
export SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH_LOG_INTERVAL=${SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH_LOG_INTERVAL:-0}
export SGLANG_WATERFILL_RANK2_SINGLE_BLOCK_COUNT_MAX_TOKENS=${SGLANG_WATERFILL_RANK2_SINGLE_BLOCK_COUNT_MAX_TOKENS:-512}
export SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK=${SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK:-}
export SGLANG_PROFILE_WITH_STACK=${SGLANG_PROFILE_WITH_STACK:-false}
export SGLANG_PROFILE_RECORD_SHAPES=${SGLANG_PROFILE_RECORD_SHAPES:-false}
export SGLANG_MEGA_MOE_PREINIT_SYMM_BUFFERS=${SGLANG_MEGA_MOE_PREINIT_SYMM_BUFFERS:-1}
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=${SGLANG_OPT_FIX_MEGA_MOE_MEMORY:-1}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

mkdir -p "$RESULT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "MODEL=$MODEL"
echo "MMLU_FILE=$MMLU_FILE"
test -s "$MMLU_FILE"

python -m compileall -q \
  /workspace/sglang/python/sglang/srt/layers/moe/mega_moe.py \
  /workspace/sglang/python/sglang/srt/layers/moe/deepep_waterfill.py \
  /workspace/sglang/python/sglang/srt/models/deepseek_v4.py

cleanup_port() {
  local port=$1
  pkill -u "$USER" -f "sglang.launch_server.*--port ${port}" >/dev/null 2>&1 || true
  sleep 3
}

wait_health() {
  local port=$1
  local server_pid=$2
  local server_log=$3

  if [ "$WAIT_READY_MODE" = "log" ]; then
    for i in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
      if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "SERVER_EXITED port=$port"
        tail -n 240 "$server_log" || true
        return 1
      fi
      if grep -q "Uvicorn running" "$server_log"; then
        echo "SERVER_LOG_READY port=$port attempt=$i"
        return 0
      fi
      if [ $((i % 30)) -eq 0 ]; then
        echo "WAIT_LOG_READY port=$port attempt=$i"
        tail -n 30 "$server_log" || true
      fi
      sleep 2
    done
    echo "SERVER_LOG_READY_TIMEOUT port=$port"
    tail -n 240 "$server_log" || true
    return 1
  fi

  for i in $(seq 1 "$WAIT_HEALTH_ATTEMPTS"); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "SERVER_EXITED port=$port"
      tail -n 240 "$server_log" || true
      return 1
    fi
    if curl --connect-timeout 2 --max-time 5 -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "SERVER_READY port=$port attempt=$i"
      return 0
    fi
    if [ $((i % 30)) -eq 0 ]; then
      echo "WAIT_HEALTH port=$port attempt=$i"
      tail -n 30 "$server_log" || true
    fi
    sleep 5
  done
  echo "SERVER_HEALTH_TIMEOUT port=$port"
  tail -n 240 "$server_log" || true
  return 1
}

wait_deepgemm_quiet() {
  local server_log=$1
  local max_wait=$2
  local stable_sec=$3
  if [ "$max_wait" -le 0 ]; then
    return 0
  fi

  echo "WAIT_DEEPGEMM_QUIET max_wait=${max_wait}s stable=${stable_sec}s"
  local start_ts
  start_ts=$(date +%s)
  local last_size=-1
  local stable_start=0
  while true; do
    local now size
    now=$(date +%s)
    size=$(stat -c %s "$server_log" 2>/dev/null || echo 0)
    if [ "$size" = "$last_size" ]; then
      if [ "$stable_start" -eq 0 ]; then
        stable_start=$now
      fi
      if [ $((now - stable_start)) -ge "$stable_sec" ]; then
        echo "DEEPGEMM_QUIET stable_for=$((now - stable_start))s elapsed=$((now - start_ts))s"
        return 0
      fi
    else
      last_size=$size
      stable_start=0
    fi

    if [ $((now - start_ts)) -ge "$max_wait" ]; then
      echo "DEEPGEMM_QUIET_TIMEOUT elapsed=$((now - start_ts))s last_size=${last_size}"
      tail -n 80 "$server_log" || true
      return 0
    fi
    sleep 5
  done
}

run_tput_client() {
  local tag=$1
  local port=$2
  local case_dir=$3
  local log_file="$case_dir/tput.log"
  timeout --kill-after=60s "${TPUT_CLIENT_TIMEOUT_SEC}s" python - <<PY 2>&1 | tee "$log_file"
import asyncio
import aiohttp
import json
import os
import random
import statistics
import time
from pathlib import Path

tag = "${tag}"
mmlu_file = Path("${MMLU_FILE}")
server = "http://127.0.0.1:${port}"
warmup = int("${TPUT_WARMUP_ROUNDS}")
rounds = int("${TPUT_MEASURE_ROUNDS}")
sample_size = int("${TPUT_SAMPLE_SIZE}")
concurrency = int("${TPUT_CONCURRENCY}")
max_tokens = int("${TPUT_MAX_TOKENS}")
timeout_sec = int("${TPUT_TIMEOUT_SEC}")
seed = "${TPUT_SEED}"

async def post_json(session, url, body):
    async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as r:
        return await r.json()

async def send_one(session, prompt, sem):
    async with sem:
        body = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        try:
            d = await post_json(session, f"{server}/v1/chat/completions", body)
            usage = d.get("usage", {})
            return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        except Exception as exc:
            print(f"  ERR: {exc}", flush=True)
            return -1

async def one_round(prompts, name, round_index):
    rng = random if not seed else random.Random(int(seed) + round_index)
    batch = rng.sample(prompts, min(sample_size, len(prompts)))
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=max(concurrency + 44, 300))
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.time()
        results = await asyncio.gather(*[send_one(session, p, sem) for p in batch])
        dt = time.time() - t0
    errs = sum(1 for x in results if x < 0)
    toks = sum(x for x in results if x >= 0)
    tput = toks / dt
    print(f"{name}: {tput:.0f} tok/s ({toks} toks in {dt:.1f}s, {errs} errs)", flush=True)
    return tput if errs == 0 else None

async def maybe_profile(session, action, output_dir=None):
    if int("${PROFILE_ENABLE}") != 1:
        return
    if action == "start":
        body = {
            "activities": "${PROFILE_ACTIVITIES}".split(),
            "num_steps": int("${PROFILE_NUM_STEPS}"),
            "profile_by_stage": bool(int("${PROFILE_BY_STAGE}")),
            "profile_stages": "${PROFILE_STAGES}".split(",") if "${PROFILE_STAGES}" else None,
            "output_dir": output_dir,
            "profile_prefix": tag,
        }
        async with session.post(f"{server}/start_profile", json=body, timeout=aiohttp.ClientTimeout(total=120)) as r:
            print("START_PROFILE", r.status, await r.text(), flush=True)
    else:
        async with session.post(f"{server}/stop_profile", json={}, timeout=aiohttp.ClientTimeout(total=300)) as r:
            print("STOP_PROFILE", r.status, await r.text(), flush=True)

async def main():
    prompts = [json.loads(line)["prompt"] for line in mmlu_file.open()]
    print(f"[{tag}] Loaded {len(prompts)} prompts, warmup={warmup}, rounds={rounds}, sample={sample_size}, concurrency={concurrency}", flush=True)
    for i in range(warmup):
        await one_round(prompts, f"warmup-{i+1}", i)
    profile_dir = "${case_dir}/profile"
    async with aiohttp.ClientSession() as session:
        await maybe_profile(session, "start", profile_dir)
    tputs = []
    for i in range(rounds):
        t = await one_round(prompts, f"round-{i+1}", warmup + i)
        if t is not None:
            tputs.append(t)
    if int("${PROFILE_ENABLE}") == 1:
        async with aiohttp.ClientSession() as session:
            await maybe_profile(session, "stop")
    if len(tputs) >= 4:
        s = sorted(tputs)
        trimmed = s[1:-1]
        print(f"\\n=== RESULT [{tag}] ===", flush=True)
        print(f"All: {[f'{t:.0f}' for t in tputs]}", flush=True)
        print(f"Trimmed mean: {sum(trimmed)/len(trimmed):.0f} tok/s", flush=True)
        print(f"Mean: {statistics.mean(tputs):.0f} tok/s", flush=True)
        print(f"Min={min(tputs):.0f} Max={max(tputs):.0f}", flush=True)

asyncio.run(main())
PY
}

summarize_case() {
  local name=$1
  local case_dir=$2
  local force_local_shared=${3:-0}
  local static_allow_all_ranks=${4:-$SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS}
  local local_pref_numer=${5:-$SGLANG_WATERFILL_LOCAL_PREF_NUMER}
  local local_pref_denom=${6:-$SGLANG_WATERFILL_LOCAL_PREF_DENOM}
  local shared_replicas_per_rank=${7:-$SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK}
  local disable_static_waterfill=${8:-$SGLANG_DISABLE_STATIC_WATERFILL}
  local disable_static_rank_load=${9:-$SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD}
  local remote_cost_tokens=${10:-$SGLANG_WATERFILL_REMOTE_COST_TOKENS}
  local one_way_remote_shared=${11:-$SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED}
  local source_aware_static_load=${12:-$SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD}
  local min_batch_for_balance=${13:-$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE}
  local fuse_megamoe_predispatch=${14:-${SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH:-}}
  local reuse_topk_buffer=${15:-$SGLANG_WATERFILL_REUSE_TOPK_BUFFER}
  local static_block_load_m=${16:-$SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M}
  python - <<PY
import ast
import json
import math
import os
import pathlib
import re
import statistics

case_dir = pathlib.Path("${case_dir}")
name = "${name}"
server_log = (case_dir / "server.log").read_text(errors="replace") if (case_dir / "server.log").exists() else ""
tput_log = (case_dir / "tput.log").read_text(errors="replace") if (case_dir / "tput.log").exists() else ""

rounds = [float(x) for x in re.findall(r"^round-\\d+:\\s+([0-9.]+) tok/s", tput_log, re.M)]
trimmed = None
if len(rounds) >= 4:
    s = sorted(rounds)
    trimmed = sum(s[1:-1]) / len(s[1:-1])

timings = []
for m in re.finditer(r"MEGA_MOE_TIMING .*?tokens=(\\d+).*?counts=(\\[[^\\]]*\\]) ratio=([0-9.]+|inf)(?: shape=(\\{.*?\\}))? timing=(\\{[^\\n]*?\\})", server_log):
    try:
        shape = ast.literal_eval(m.group(4)) if m.group(4) else {}
        timing = ast.literal_eval(m.group(5))
        counts = ast.literal_eval(m.group(2))
    except Exception:
        continue
    timings.append({
        "tokens": int(m.group(1)),
        "counts": counts,
        "ratio": float(m.group(3)) if m.group(3) != "inf" else math.inf,
        "shape": shape,
        "timing": timing,
    })

wf_stats = []
for m in re.finditer(r"WATERFILL_STATS .*?tokens=(\\d+).*?before=(\\[[^\\]]*\\]) shared=(\\[[^\\]]*\\]) after=(\\[[^\\]]*\\]).*?before_max_min=([0-9]+)/([0-9]+) after_max_min=([0-9]+)/([0-9]+)(?: .*?shared_remote=([0-9]+) shared_remote_new_rank=([0-9]+))?", server_log):
    try:
        before = ast.literal_eval(m.group(2))
        shared = ast.literal_eval(m.group(3))
        after = ast.literal_eval(m.group(4))
    except Exception:
        continue
    def ratio(xs):
        nz = [x for x in xs if x > 0]
        return (max(xs) / min(nz)) if nz else None
    def cv(xs):
        if not xs:
            return None
        mean = statistics.mean(xs)
        return statistics.pstdev(xs) / mean if mean else None
    wf_stats.append({
        "tokens": int(m.group(1)),
        "before": before,
        "shared": shared,
        "after": after,
        "before_ratio": ratio(before),
        "after_ratio": ratio(after),
        "before_cv": cv(before),
        "after_cv": cv(after),
        "shared_remote": int(m.group(9)) if m.group(9) is not None else None,
        "shared_remote_new_rank": int(m.group(10)) if m.group(10) is not None else None,
    })

def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isinf(x))]
    return statistics.mean(xs) if xs else None

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isinf(x)))
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))]

def optional_bool(value):
    return None if value == "" else bool(int(value))

def optional_int(value):
    return None if value == "" else int(value)

pre_ms = [x["timing"].get("pre_dispatch_to_fp8_fp4_ms") for x in timings]
shape_keys = (
    "active_experts",
    "active_local_experts",
    "max_expert_tokens",
    "max_local_expert_tokens",
    "local_expert_tokens_sum",
    "local_expert_blocks_64",
    "local_expert_blocks_128",
    "local_expert_blocks_256",
    "max_local_expert_blocks_64",
    "max_local_expert_blocks_128",
    "max_local_expert_blocks_256",
    "p95_nonzero_expert_tokens",
    "remote_routed_entries",
    "remote_shared_entries",
    "shared_remote_new_rank",
    "tokens_multi_routed_rank",
    "tokens_multi_full_rank",
    "mean_distinct_routed_ranks",
    "mean_distinct_full_ranks",
    "shared_replicas_per_rank",
)
shape_summary = {}
for key in shape_keys:
    vals = [x["shape"].get(key) for x in timings if key in x["shape"]]
    if vals:
        shape_summary[f"mega_moe_shape_{key}_mean"] = mean(vals)
summary = {
    "case": name,
    "tput_rounds": rounds,
    "tput_mean": statistics.mean(rounds) if rounds else None,
    "tput_trimmed_mean": trimmed,
    "mega_moe_timing_samples": len(timings),
    "mega_moe_ratio_mean": mean([x["ratio"] for x in timings]),
    "mega_moe_ratio_median": pct([x["ratio"] for x in timings], 0.5),
    "mega_moe_ratio_p95": pct([x["ratio"] for x in timings], 0.95),
    "mega_moe_pre_dispatch_to_fp8_fp4_ms_mean": mean(pre_ms),
    "mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted": (
        sum((x["timing"].get("pre_dispatch_to_fp8_fp4_ms") or 0) * x["tokens"] for x in timings)
        / sum(x["tokens"] for x in timings if x["timing"].get("pre_dispatch_to_fp8_fp4_ms") is not None)
        if timings and sum(x["tokens"] for x in timings if x["timing"].get("pre_dispatch_to_fp8_fp4_ms") is not None) else None
    ),
    "waterfill_stats_samples": len(wf_stats),
    "waterfill_before_ratio_mean": mean([x["before_ratio"] for x in wf_stats]),
    "waterfill_after_ratio_mean": mean([x["after_ratio"] for x in wf_stats]),
    "waterfill_before_ratio_median": pct([x["before_ratio"] for x in wf_stats], 0.5),
    "waterfill_after_ratio_median": pct([x["after_ratio"] for x in wf_stats], 0.5),
    "waterfill_before_cv_mean": mean([x["before_cv"] for x in wf_stats]),
    "waterfill_after_cv_mean": mean([x["after_cv"] for x in wf_stats]),
    "waterfill_shared_remote_mean": mean([x["shared_remote"] for x in wf_stats]),
    "waterfill_shared_remote_new_rank_mean": mean([x["shared_remote_new_rank"] for x in wf_stats]),
    "ep_num_redundant_experts": int("${EP_NUM_REDUNDANT_EXPERTS}"),
    "ep_dispatch_algorithm": "${EP_DISPATCH_ALGORITHM}",
    "init_expert_location": "${INIT_EXPERT_LOCATION}",
    "enable_eplb": bool(int("${ENABLE_EPLB}")),
    "eplb_algorithm": "${EPLB_ALGORITHM}",
    "eplb_rebalance_num_iterations": "${EPLB_REBALANCE_NUM_ITERATIONS}",
    "eplb_rebalance_layers_per_chunk": "${EPLB_REBALANCE_LAYERS_PER_CHUNK}",
    "eplb_min_rebalancing_utilization_threshold": "${EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD}",
    "expert_distribution_recorder_mode": "${EXPERT_DISTRIBUTION_RECORDER_MODE}",
    "expert_distribution_recorder_buffer_size": "${EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE}",
    "mem_fraction_static": "${MEM_FRACTION_STATIC}",
    "waterfill_force_local_shared": bool(int("${force_local_shared}")),
    "waterfill_disable_static": bool(int("${disable_static_waterfill}")),
    "waterfill_disable_static_rank_load": bool(int("${disable_static_rank_load}")),
    "waterfill_static_allow_all_ranks": bool(int("${static_allow_all_ranks}")),
    "waterfill_local_pref_numer": int("${local_pref_numer}"),
    "waterfill_local_pref_denom": int("${local_pref_denom}"),
    "waterfill_remote_cost_tokens": int("${remote_cost_tokens}"),
    "waterfill_one_way_remote_shared": optional_bool("${one_way_remote_shared}"),
    "waterfill_source_aware_static_load": bool(int("${source_aware_static_load}")),
    "waterfill_static_block_load_m": int("${static_block_load_m}"),
    "waterfill_min_batch_for_balance": int("${min_batch_for_balance}"),
    "waterfill_fuse_megamoe_predispatch": optional_bool("${fuse_megamoe_predispatch}"),
    "waterfill_reuse_topk_buffer": bool(int("${reuse_topk_buffer}")),
    "waterfill_shared_replicas_per_rank": optional_int("${shared_replicas_per_rank}"),
    "mega_moe_cap": int("${MEGA_MOE_CAP}"),
    "mega_moe_cap_buckets_auto": bool(int("${MEGA_MOE_CAP_BUCKETS_AUTO}")),
    "mega_moe_cap_buckets": "${MEGA_MOE_CAP_BUCKETS}",
    "mega_moe_preinit_all_cap_buckets": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_PREINIT_ALL_CAP_BUCKETS}")),
    "mega_moe_cap_bucket_min_free_gb": float("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_CAP_BUCKET_MIN_FREE_GB}"),
    "mega_moe_use_fp4_acts": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS}")),
    "mega_moe_use_mxf4_kind": bool(int("${SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND}")),
    "profile_files": [str(p) for p in sorted(case_dir.glob("profile/**/*.json*"))],
    **shape_summary,
}
(case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_case() {
  local name=$1
  local waterfill=$2
  local force_local_shared=${3:-$SGLANG_WATERFILL_FORCE_LOCAL_SHARED}
  local static_allow_all_ranks=${4:-$SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS}
  local local_pref_numer=${5:-$SGLANG_WATERFILL_LOCAL_PREF_NUMER}
  local local_pref_denom=${6:-$SGLANG_WATERFILL_LOCAL_PREF_DENOM}
  local shared_replicas_per_rank=${7:-$SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK}
  local disable_static_waterfill=${8:-$SGLANG_DISABLE_STATIC_WATERFILL}
  local disable_static_rank_load=${9:-$SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD}
  local remote_cost_tokens=${10:-$SGLANG_WATERFILL_REMOTE_COST_TOKENS}
  local one_way_remote_shared=${11:-$SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED}
  local source_aware_static_load=${12:-$SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD}
  local min_batch_for_balance=${13:-$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE}
  local fuse_megamoe_predispatch=${14:-${SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH:-}}
  local reuse_topk_buffer=${15:-$SGLANG_WATERFILL_REUSE_TOPK_BUFFER}
  local static_block_load_m=${16:-$SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M}
  local port=$PORT
  if [ "$name" = "fused_waterfill" ] || [ "$name" = "waterfill" ]; then
    port=$((PORT + CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_local" ] || [ "$name" = "waterfill_local" ]; then
    port=$((PORT + 2 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_routed" ] || [ "$name" = "waterfill_routed" ]; then
    port=$((PORT + 3 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_pref2" ] || [ "$name" = "waterfill_pref2" ]; then
    port=$((PORT + 4 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_pref4" ] || [ "$name" = "waterfill_pref4" ]; then
    port=$((PORT + 5 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_pref8" ] || [ "$name" = "waterfill_pref8" ]; then
    port=$((PORT + 6 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_pref16" ] || [ "$name" = "waterfill_pref16" ]; then
    port=$((PORT + 11 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_pref32" ] || [ "$name" = "waterfill_pref32" ]; then
    port=$((PORT + 12 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost512" ] || [ "$name" = "waterfill_cost512" ]; then
    port=$((PORT + 13 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost1024" ] || [ "$name" = "waterfill_cost1024" ]; then
    port=$((PORT + 14 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost2048" ] || [ "$name" = "waterfill_cost2048" ]; then
    port=$((PORT + 15 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost4096" ] || [ "$name" = "waterfill_cost4096" ]; then
    port=$((PORT + 16 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost8192" ] || [ "$name" = "waterfill_cost8192" ]; then
    port=$((PORT + 17 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_cost16384" ] || [ "$name" = "waterfill_cost16384" ]; then
    port=$((PORT + 18 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_routed_cost4096" ] || [ "$name" = "waterfill_routed_cost4096" ]; then
    port=$((PORT + 19 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_routed_cost8192" ] || [ "$name" = "waterfill_routed_cost8192" ]; then
    port=$((PORT + 20 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_oneway" ] || [ "$name" = "waterfill_oneway" ]; then
    port=$((PORT + 21 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_source" ] || [ "$name" = "waterfill_source" ]; then
    port=$((PORT + 22 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_source_cost4096" ] || [ "$name" = "waterfill_source_cost4096" ]; then
    port=$((PORT + 23 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_source_cost8192" ] || [ "$name" = "waterfill_source_cost8192" ]; then
    port=$((PORT + 24 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_source_cost16384" ] || [ "$name" = "waterfill_source_cost16384" ]; then
    port=$((PORT + 25 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_min256" ] || [ "$name" = "waterfill_min256" ]; then
    port=$((PORT + 26 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_min512" ] || [ "$name" = "waterfill_min512" ]; then
    port=$((PORT + 27 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_min1024" ] || [ "$name" = "waterfill_min1024" ]; then
    port=$((PORT + 28 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_fusedpredispatch" ] || [ "$name" = "waterfill_fusedpredispatch" ]; then
    port=$((PORT + 29 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_oneway_fusedpredispatch" ]; then
    port=$((PORT + 31 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_source_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_source_oneway_fusedpredispatch" ]; then
    port=$((PORT + 32 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep2_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep2_oneway_fusedpredispatch" ]; then
    port=$((PORT + 33 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_oneway_fusedpredispatch" ]; then
    port=$((PORT + 34 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep5_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep5_oneway_fusedpredispatch" ]; then
    port=$((PORT + 43 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep7_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep7_oneway_fusedpredispatch" ]; then
    port=$((PORT + 44 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep10_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep10_oneway_fusedpredispatch" ]; then
    port=$((PORT + 45 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep13_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep13_oneway_fusedpredispatch" ]; then
    port=$((PORT + 46 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep3_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep3_oneway_fusedpredispatch" ]; then
    port=$((PORT + 35 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep6_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep6_oneway_fusedpredispatch" ]; then
    port=$((PORT + 36 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep8_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep8_oneway_fusedpredispatch" ]; then
    port=$((PORT + 37 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_source_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_source_oneway_fusedpredispatch" ]; then
    port=$((PORT + 38 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_block128_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_block128_oneway_fusedpredispatch" ]; then
    port=$((PORT + 39 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_cost1024_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_cost1024_oneway_fusedpredispatch" ]; then
    port=$((PORT + 40 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_cost2048_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_cost2048_oneway_fusedpredispatch" ]; then
    port=$((PORT + 41 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4_pref12_oneway_fusedpredispatch" ] || [ "$name" = "waterfill_rep4_pref12_oneway_fusedpredispatch" ]; then
    port=$((PORT + 42 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_block128" ] || [ "$name" = "waterfill_block128" ]; then
    port=$((PORT + 30 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_reuse" ] || [ "$name" = "waterfill_reuse" ]; then
    port=$((PORT + 7 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep2" ] || [ "$name" = "waterfill_rep2" ]; then
    port=$((PORT + 7 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_rep4" ] || [ "$name" = "waterfill_rep4" ]; then
    port=$((PORT + 8 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_dynamic" ] || [ "$name" = "waterfill_dynamic" ]; then
    port=$((PORT + 9 * CASE_PORT_STRIDE))
  elif [ "$name" = "fused_waterfill_static_local" ] || [ "$name" = "waterfill_static_local" ]; then
    port=$((PORT + 10 * CASE_PORT_STRIDE))
  fi
  local case_dir="$RESULT_ROOT/$name"
  mkdir -p "$case_dir"
  local server_log="$case_dir/server.log"
  rm -f "$server_log"
  cleanup_port "$port"

  local extra=()
  if [ "$waterfill" = "1" ]; then
    extra+=(--enable-deepep-waterfill)
  fi
  if [ "$SKIP_SERVER_WARMUP" = "1" ]; then
    extra+=(--skip-server-warmup)
  fi
  if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
    extra+=(--disable-cuda-graph)
  fi
  if [ "$DISABLE_RADIX_CACHE" = "1" ]; then
    extra+=(--disable-radix-cache)
  fi
  if [ "$ENFORCE_SHARED_EXPERTS_FUSION" = "1" ]; then
    extra+=(--enforce-shared-experts-fusion)
  fi
  if [ -n "$MEM_FRACTION_STATIC" ]; then
    extra+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
  fi
  if [ "$EP_NUM_REDUNDANT_EXPERTS" != "0" ]; then
    extra+=(--ep-num-redundant-experts "$EP_NUM_REDUNDANT_EXPERTS")
  fi
  if [ -n "$EP_DISPATCH_ALGORITHM" ]; then
    extra+=(--ep-dispatch-algorithm "$EP_DISPATCH_ALGORITHM")
  fi
  if [ -n "$INIT_EXPERT_LOCATION" ]; then
    extra+=(--init-expert-location "$INIT_EXPERT_LOCATION")
  fi
  if [ "$ENABLE_EPLB" = "1" ]; then
    extra+=(--enable-eplb)
  fi
  if [ -n "$EPLB_ALGORITHM" ]; then
    extra+=(--eplb-algorithm "$EPLB_ALGORITHM")
  fi
  if [ -n "$EPLB_REBALANCE_NUM_ITERATIONS" ]; then
    extra+=(--eplb-rebalance-num-iterations "$EPLB_REBALANCE_NUM_ITERATIONS")
  fi
  if [ -n "$EPLB_REBALANCE_LAYERS_PER_CHUNK" ]; then
    extra+=(--eplb-rebalance-layers-per-chunk "$EPLB_REBALANCE_LAYERS_PER_CHUNK")
  fi
  if [ -n "$EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD" ]; then
    extra+=(--eplb-min-rebalancing-utilization-threshold "$EPLB_MIN_REBALANCING_UTILIZATION_THRESHOLD")
  fi
  if [ -n "$EXPERT_DISTRIBUTION_RECORDER_MODE" ]; then
    extra+=(--expert-distribution-recorder-mode "$EXPERT_DISTRIBUTION_RECORDER_MODE")
  fi
  if [ -n "$EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE" ]; then
    extra+=(--expert-distribution-recorder-buffer-size "$EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE")
  fi
  if [ -n "$EXTRA_SERVER_ARGS" ]; then
    read -r -a extra_server_args <<<"$EXTRA_SERVER_ARGS"
    extra+=("${extra_server_args[@]}")
  fi

  local fuse_env_args=(-u SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH)
  if [ -n "$fuse_megamoe_predispatch" ]; then
    fuse_env_args=("SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH=$fuse_megamoe_predispatch")
  fi
  local one_way_env_args=(-u SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED)
  if [ -n "$one_way_remote_shared" ]; then
    one_way_env_args=("SGLANG_WATERFILL_ONE_WAY_REMOTE_SHARED=$one_way_remote_shared")
  fi
  local shared_replica_env_args=(-u SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK)
  if [ -n "$shared_replicas_per_rank" ]; then
    shared_replica_env_args=("SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK=$shared_replicas_per_rank")
  fi

  echo "=== START_CASE $name waterfill=$waterfill force_local_shared=$force_local_shared disable_static_waterfill=$disable_static_waterfill disable_static_rank_load=$disable_static_rank_load static_allow_all_ranks=$static_allow_all_ranks local_pref=$local_pref_numer/$local_pref_denom remote_cost=$remote_cost_tokens one_way=$one_way_remote_shared source_aware=$source_aware_static_load static_block_load_m=$static_block_load_m min_batch_for_balance=$min_batch_for_balance fuse_megamoe_predispatch=${fuse_megamoe_predispatch:-auto} reuse_topk_buffer=$reuse_topk_buffer shared_replicas=$shared_replicas_per_rank cap=$MEGA_MOE_CAP cap_buckets_auto=$MEGA_MOE_CAP_BUCKETS_AUTO cap_buckets=$MEGA_MOE_CAP_BUCKETS port=$port $(date) ==="
  SGLANG_WATERFILL_FORCE_LOCAL_SHARED="$force_local_shared" \
  SGLANG_DISABLE_STATIC_WATERFILL="$disable_static_waterfill" \
  SGLANG_WATERFILL_DISABLE_STATIC_RANK_LOAD="$disable_static_rank_load" \
  SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS="$static_allow_all_ranks" \
  SGLANG_WATERFILL_LOCAL_PREF_NUMER="$local_pref_numer" \
  SGLANG_WATERFILL_LOCAL_PREF_DENOM="$local_pref_denom" \
  SGLANG_WATERFILL_REMOTE_COST_TOKENS="$remote_cost_tokens" \
  SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD="$source_aware_static_load" \
  SGLANG_WATERFILL_STATIC_BLOCK_LOAD_M="$static_block_load_m" \
  SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE="$min_batch_for_balance" \
  SGLANG_WATERFILL_REUSE_TOPK_BUFFER="$reuse_topk_buffer" \
  env "${fuse_env_args[@]}" "${one_way_env_args[@]}" "${shared_replica_env_args[@]}" setsid python -m sglang.launch_server \
    --model-path "$MODEL" \
    --trust-remote-code \
    --tp "$TP_SIZE" \
    --dp "$DP_SIZE" \
    --moe-dense-tp-size "$MOE_DENSE_TP_SIZE" \
    --moe-a2a-backend "$MOE_A2A_BACKEND" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --host 127.0.0.1 \
    --port "$port" \
    "${extra[@]}" \
    >"$server_log" 2>&1 &
  local server_pid=$!
  local server_pgid
  server_pgid=$(ps -o pgid= -p "$server_pid" 2>/dev/null | tr -d ' ' || true)
  local cleanup_done=0

  cleanup_case() {
    set +e
    if [ "$cleanup_done" = "1" ]; then
      set -e
      return 0
    fi
    cleanup_done=1
    if [ -n "${server_pgid:-}" ]; then
      kill -TERM "-$server_pgid" >/dev/null 2>&1 || true
    fi
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 8
    if [ -n "${server_pgid:-}" ]; then
      kill -KILL "-$server_pgid" >/dev/null 2>&1 || true
    fi
    kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1
    cleanup_port "$port"
    set -e
    return 0
  }
  trap cleanup_case RETURN

  wait_health "$port" "$server_pid" "$server_log"
  wait_deepgemm_quiet "$server_log" "$DEEPGEMM_QUIET_WAIT_SEC" "$DEEPGEMM_QUIET_STABLE_SEC"
  run_tput_client "$name" "$port" "$case_dir"
  summarize_case "$name" "$case_dir" "$force_local_shared" "$static_allow_all_ranks" "$local_pref_numer" "$local_pref_denom" "$shared_replicas_per_rank" "$disable_static_waterfill" "$disable_static_rank_load" "$remote_cost_tokens" "$one_way_remote_shared" "$source_aware_static_load" "$min_batch_for_balance" "$fuse_megamoe_predispatch" "$reuse_topk_buffer" "$static_block_load_m"
  grep -Ei "MEGA_MOE_TIMING|WATERFILL_STATS|Prepared .*waterfill|enable_deepep_waterfill|Traceback|ERROR|OOM|out of memory" "$server_log" | tail -240 || true

  cleanup_case
  trap - RETURN
  echo "=== END_CASE $name $(date) ==="
}

IFS=',' read -r -a cases <<<"${CASE_ORDER//:/,}"
for case_name in "${cases[@]}"; do
  case "$case_name" in
    fused)
      run_case fused 0
      ;;
    fused_waterfill|waterfill)
      run_case fused_waterfill 1 0
      ;;
    fused_waterfill_dynamic|waterfill_dynamic)
      run_case fused_waterfill_dynamic 1 0 1 11 10 1 1
      ;;
    fused_waterfill_static_local|waterfill_static_local)
      run_case fused_waterfill_static_local 1 0 1 11 10 1 0 1
      ;;
    fused_waterfill_routed|waterfill_routed)
      run_case fused_waterfill_routed 1 0 0
      ;;
    fused_waterfill_pref2|waterfill_pref2)
      run_case fused_waterfill_pref2 1 0 1 2 1
      ;;
    fused_waterfill_pref4|waterfill_pref4)
      run_case fused_waterfill_pref4 1 0 1 4 1
      ;;
    fused_waterfill_pref8|waterfill_pref8)
      run_case fused_waterfill_pref8 1 0 1 8 1
      ;;
    fused_waterfill_pref16|waterfill_pref16)
      run_case fused_waterfill_pref16 1 0 1 16 1
      ;;
    fused_waterfill_pref32|waterfill_pref32)
      run_case fused_waterfill_pref32 1 0 1 32 1
      ;;
    fused_waterfill_cost512|waterfill_cost512)
      run_case fused_waterfill_cost512 1 0 1 11 10 1 0 0 512
      ;;
    fused_waterfill_cost1024|waterfill_cost1024)
      run_case fused_waterfill_cost1024 1 0 1 11 10 1 0 0 1024
      ;;
    fused_waterfill_cost2048|waterfill_cost2048)
      run_case fused_waterfill_cost2048 1 0 1 11 10 1 0 0 2048
      ;;
    fused_waterfill_cost4096|waterfill_cost4096)
      run_case fused_waterfill_cost4096 1 0 1 11 10 1 0 0 4096
      ;;
    fused_waterfill_cost8192|waterfill_cost8192)
      run_case fused_waterfill_cost8192 1 0 1 11 10 1 0 0 8192
      ;;
    fused_waterfill_cost16384|waterfill_cost16384)
      run_case fused_waterfill_cost16384 1 0 1 11 10 1 0 0 16384
      ;;
    fused_waterfill_routed_cost4096|waterfill_routed_cost4096)
      run_case fused_waterfill_routed_cost4096 1 0 0 11 10 1 0 0 4096
      ;;
    fused_waterfill_routed_cost8192|waterfill_routed_cost8192)
      run_case fused_waterfill_routed_cost8192 1 0 0 11 10 1 0 0 8192
      ;;
    fused_waterfill_oneway|waterfill_oneway)
      run_case fused_waterfill_oneway 1 0 1 11 10 1 0 0 0 1
      ;;
    fused_waterfill_source|waterfill_source)
      run_case fused_waterfill_source 1 0 1 11 10 1 0 0 0 0 1
      ;;
    fused_waterfill_source_cost4096|waterfill_source_cost4096)
      run_case fused_waterfill_source_cost4096 1 0 1 11 10 1 0 0 4096 0 1
      ;;
    fused_waterfill_source_cost8192|waterfill_source_cost8192)
      run_case fused_waterfill_source_cost8192 1 0 1 11 10 1 0 0 8192 0 1
      ;;
    fused_waterfill_source_cost16384|waterfill_source_cost16384)
      run_case fused_waterfill_source_cost16384 1 0 1 11 10 1 0 0 16384 0 1
      ;;
    fused_waterfill_min256|waterfill_min256)
      run_case fused_waterfill_min256 1 0 1 11 10 1 0 0 0 0 0 256
      ;;
    fused_waterfill_min512|waterfill_min512)
      run_case fused_waterfill_min512 1 0 1 11 10 1 0 0 0 0 0 512
      ;;
    fused_waterfill_min1024|waterfill_min1024)
      run_case fused_waterfill_min1024 1 0 1 11 10 1 0 0 0 0 0 1024
      ;;
    fused_waterfill_fusedpredispatch|waterfill_fusedpredispatch)
      run_case fused_waterfill_fusedpredispatch 1 0 1 11 10 1 0 0 0 0 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_oneway_fusedpredispatch|waterfill_oneway_fusedpredispatch)
      run_case fused_waterfill_oneway_fusedpredispatch 1 0 1 11 10 1 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_source_oneway_fusedpredispatch|waterfill_source_oneway_fusedpredispatch)
      run_case fused_waterfill_source_oneway_fusedpredispatch 1 0 1 11 10 1 0 0 0 1 1 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep2_oneway_fusedpredispatch|waterfill_rep2_oneway_fusedpredispatch)
      run_case fused_waterfill_rep2_oneway_fusedpredispatch 1 0 1 11 10 2 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep4_oneway_fusedpredispatch|waterfill_rep4_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_oneway_fusedpredispatch 1 0 1 11 10 4 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep5_oneway_fusedpredispatch|waterfill_rep5_oneway_fusedpredispatch)
      run_case fused_waterfill_rep5_oneway_fusedpredispatch 1 0 1 11 10 5 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep7_oneway_fusedpredispatch|waterfill_rep7_oneway_fusedpredispatch)
      run_case fused_waterfill_rep7_oneway_fusedpredispatch 1 0 1 11 10 7 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep10_oneway_fusedpredispatch|waterfill_rep10_oneway_fusedpredispatch)
      run_case fused_waterfill_rep10_oneway_fusedpredispatch 1 0 1 11 10 10 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep13_oneway_fusedpredispatch|waterfill_rep13_oneway_fusedpredispatch)
      run_case fused_waterfill_rep13_oneway_fusedpredispatch 1 0 1 11 10 13 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep3_oneway_fusedpredispatch|waterfill_rep3_oneway_fusedpredispatch)
      run_case fused_waterfill_rep3_oneway_fusedpredispatch 1 0 1 11 10 3 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep6_oneway_fusedpredispatch|waterfill_rep6_oneway_fusedpredispatch)
      run_case fused_waterfill_rep6_oneway_fusedpredispatch 1 0 1 11 10 6 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep8_oneway_fusedpredispatch|waterfill_rep8_oneway_fusedpredispatch)
      run_case fused_waterfill_rep8_oneway_fusedpredispatch 1 0 1 11 10 8 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep4_source_oneway_fusedpredispatch|waterfill_rep4_source_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_source_oneway_fusedpredispatch 1 0 1 11 10 4 0 0 0 1 1 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep4_block128_oneway_fusedpredispatch|waterfill_rep4_block128_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_block128_oneway_fusedpredispatch 1 0 1 11 10 4 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1 0 128
      ;;
    fused_waterfill_rep4_cost1024_oneway_fusedpredispatch|waterfill_rep4_cost1024_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_cost1024_oneway_fusedpredispatch 1 0 1 11 10 4 0 0 1024 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep4_cost2048_oneway_fusedpredispatch|waterfill_rep4_cost2048_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_cost2048_oneway_fusedpredispatch 1 0 1 11 10 4 0 0 2048 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_rep4_pref12_oneway_fusedpredispatch|waterfill_rep4_pref12_oneway_fusedpredispatch)
      run_case fused_waterfill_rep4_pref12_oneway_fusedpredispatch 1 0 1 12 10 4 0 0 0 1 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 1
      ;;
    fused_waterfill_reuse|waterfill_reuse)
      run_case fused_waterfill_reuse 1 0 1 11 10 1 0 0 0 0 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 0 1
      ;;
    fused_waterfill_block128|waterfill_block128)
      run_case fused_waterfill_block128 1 0 1 11 10 1 0 0 0 0 0 "$SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE" 0 0 128
      ;;
    fused_waterfill_rep2|waterfill_rep2)
      run_case fused_waterfill_rep2 1 0 1 11 10 2
      ;;
    fused_waterfill_rep4|waterfill_rep4)
      run_case fused_waterfill_rep4 1 0 1 11 10 4
      ;;
    fused_waterfill_local|waterfill_local)
      run_case fused_waterfill_local 1 1
      ;;
    *)
      echo "UNKNOWN_CASE $case_name"
      exit 2
      ;;
  esac
done

python - <<'PY'
import json
import os
import pathlib

root = pathlib.Path(os.environ["RESULT_ROOT"])
cases = []
for name in os.environ.get("CASE_ORDER", "fused,fused_waterfill").replace(":", ",").split(","):
    key = {
        "waterfill": "fused_waterfill",
        "waterfill_routed": "fused_waterfill_routed",
        "waterfill_local": "fused_waterfill_local",
        "waterfill_pref2": "fused_waterfill_pref2",
        "waterfill_pref4": "fused_waterfill_pref4",
        "waterfill_pref8": "fused_waterfill_pref8",
        "waterfill_pref16": "fused_waterfill_pref16",
        "waterfill_pref32": "fused_waterfill_pref32",
        "waterfill_cost512": "fused_waterfill_cost512",
        "waterfill_cost1024": "fused_waterfill_cost1024",
        "waterfill_cost2048": "fused_waterfill_cost2048",
        "waterfill_cost4096": "fused_waterfill_cost4096",
        "waterfill_cost8192": "fused_waterfill_cost8192",
        "waterfill_cost16384": "fused_waterfill_cost16384",
        "waterfill_routed_cost4096": "fused_waterfill_routed_cost4096",
        "waterfill_routed_cost8192": "fused_waterfill_routed_cost8192",
        "waterfill_oneway": "fused_waterfill_oneway",
        "waterfill_source": "fused_waterfill_source",
        "waterfill_source_cost4096": "fused_waterfill_source_cost4096",
        "waterfill_source_cost8192": "fused_waterfill_source_cost8192",
        "waterfill_source_cost16384": "fused_waterfill_source_cost16384",
        "waterfill_min256": "fused_waterfill_min256",
        "waterfill_min512": "fused_waterfill_min512",
        "waterfill_min1024": "fused_waterfill_min1024",
        "waterfill_fusedpredispatch": "fused_waterfill_fusedpredispatch",
        "waterfill_block128": "fused_waterfill_block128",
        "waterfill_reuse": "fused_waterfill_reuse",
        "waterfill_rep2": "fused_waterfill_rep2",
        "waterfill_rep4": "fused_waterfill_rep4",
        "waterfill_dynamic": "fused_waterfill_dynamic",
        "waterfill_static_local": "fused_waterfill_static_local",
    }.get(name, name)
    p = root / key / "summary.json"
    if p.exists():
        cases.append(json.loads(p.read_text()))
by = {c["case"]: c for c in cases}
summary = {"root": str(root), "cases": cases}

def add_pair(prefix, left, right):
    for key in (
        "tput_trimmed_mean",
        "tput_mean",
        "mega_moe_pre_dispatch_to_fp8_fp4_ms_mean",
        "mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted",
        "mega_moe_ratio_mean",
        "waterfill_after_ratio_mean",
    ):
        a = left.get(key)
        b = right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            summary[f"{prefix}_{key}_left"] = a
            summary[f"{prefix}_{key}_right"] = b
            summary[f"{prefix}_{key}_right_over_left"] = b / a
    if isinstance(left.get("tput_trimmed_mean"), (int, float)) and isinstance(right.get("tput_trimmed_mean"), (int, float)):
        summary[f"{prefix}_tput_trimmed_mean_speedup_pct"] = (right["tput_trimmed_mean"] / left["tput_trimmed_mean"] - 1.0) * 100.0
    if isinstance(left.get("mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted"), (int, float)) and isinstance(right.get("mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted"), (int, float)):
        summary[f"{prefix}_mega_moe_token_weighted_speedup_pct"] = (left["mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted"] / right["mega_moe_pre_dispatch_to_fp8_fp4_ms_token_weighted"] - 1.0) * 100.0

if "fused" in by and "fused_waterfill" in by:
    add_pair("remote_vs_fused", by["fused"], by["fused_waterfill"])
if "fused" in by and "fused_waterfill_local" in by:
    add_pair("local_vs_fused", by["fused"], by["fused_waterfill_local"])
if "fused" in by and "fused_waterfill_routed" in by:
    add_pair("routed_vs_fused", by["fused"], by["fused_waterfill_routed"])
if "fused_waterfill" in by and "fused_waterfill_routed" in by:
    add_pair("routed_vs_remote", by["fused_waterfill"], by["fused_waterfill_routed"])
if "fused_waterfill" in by and "fused_waterfill_local" in by:
    add_pair("local_vs_remote", by["fused_waterfill"], by["fused_waterfill_local"])
if "fused_waterfill_static_local" in by and "fused_waterfill" in by:
    add_pair("static_global_vs_static_local", by["fused_waterfill_static_local"], by["fused_waterfill"])
if "fused_waterfill_static_local" in by and "fused_waterfill_dynamic" in by:
    add_pair("dynamic_vs_static_local", by["fused_waterfill_static_local"], by["fused_waterfill_dynamic"])
for name, case in sorted(by.items()):
    if name.startswith("fused_waterfill_pref") and "fused_waterfill" in by:
        add_pair(f"{name}_vs_remote", by["fused_waterfill"], case)
    if name.startswith("fused_waterfill_cost") and "fused" in by:
        add_pair(f"{name}_vs_fused", by["fused"], case)
    if name.startswith("fused_waterfill_cost") and "fused_waterfill" in by:
        add_pair(f"{name}_vs_remote", by["fused_waterfill"], case)
    if name.startswith("fused_waterfill_routed_cost") and "fused" in by:
        add_pair(f"{name}_vs_fused", by["fused"], case)
    if name.startswith("fused_waterfill_routed_cost") and "fused_waterfill" in by:
        add_pair(f"{name}_vs_remote", by["fused_waterfill"], case)
    if name.startswith("fused_waterfill_source_cost") and "fused" in by:
        add_pair(f"{name}_vs_fused", by["fused"], case)
    if name.startswith("fused_waterfill_min") and "fused" in by:
        add_pair(f"{name}_vs_fused", by["fused"], case)
    if name.startswith("fused_waterfill_min") and "fused_waterfill" in by:
        add_pair(f"{name}_vs_remote", by["fused_waterfill"], case)
    if name == "fused_waterfill_fusedpredispatch" and "fused" in by:
        add_pair("fusedpredispatch_vs_fused", by["fused"], case)
    if name == "fused_waterfill_fusedpredispatch" and "fused_waterfill" in by:
        add_pair("fusedpredispatch_vs_remote", by["fused_waterfill"], case)
    if name == "fused_waterfill_block128" and "fused" in by:
        add_pair("block128_vs_fused", by["fused"], case)
    if name == "fused_waterfill_block128" and "fused_waterfill" in by:
        add_pair("block128_vs_remote", by["fused_waterfill"], case)
    if name == "fused_waterfill_reuse" and "fused_waterfill" in by:
        add_pair("reuse_vs_remote", by["fused_waterfill"], case)
    if name.startswith("fused_waterfill_rep") and "fused_waterfill" in by:
        add_pair(f"{name}_vs_remote", by["fused_waterfill"], case)
    if name == "fused_waterfill_dynamic" and "fused_waterfill" in by:
        add_pair("dynamic_vs_remote", by["fused_waterfill"], case)
(root / "compare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

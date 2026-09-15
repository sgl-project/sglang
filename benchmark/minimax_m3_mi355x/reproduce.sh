#!/bin/bash
# MiniMax-M3 AgentX reproduction on 4x MI350X/MI355X: TP4, MXFP4 quark checkpoint, fp8 KV, EAGLE3, SemiAnalysis AIPerf client.
# Usage: bash reproduce.sh real  [c ...]         recommended config; AIPerf points at c=1 8 24 32 (3600 s each) and a summary
#                                                (first run installs sglang from this checkout, aiter, the client and both models)
#        bash reproduce.sh lossy [c ...]         ATOM-parity performance-only config (forced acceptance; outputs are not the model's)
#        bash reproduce.sh setup | serve real|lossy | stop      setup only / server only / kill the server
# Env:   M3_WORK=/scratch (models, results, logs, aiter, client; default /scratch if writable else ~/m3-agentx)
#        GPUS=0,1,2,3  PORT=30000  DURATION=3600  SKIP_MODELS=1 (setup without downloads)  CHECK_ONLY=1 (preflight only)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SGLANG_DIR=$(cd "$HERE/../.." && pwd)
if [ -z "${M3_WORK:-}" ]; then if [ -d /scratch ] && [ -w /scratch ]; then M3_WORK=/scratch; else M3_WORK=$HOME/m3-agentx; fi; fi
: "${GPUS:=0,1,2,3}" "${PORT:=30000}" "${DURATION:=3600}" "${PYTHON:=python3}"
MODEL=$M3_WORK/models/MiniMax-M3-MXFP4 DRAFT=$M3_WORK/models/MiniMax-M3-EAGLE3-GQA AITER_DIR=$M3_WORK/aiter
AIPERF_DIR=$M3_WORK/aiperf-sa AIPERF=$M3_WORK/aiperf-sa-venv/bin/aiperf RESULTS=$M3_WORK/results LOGS=$M3_WORK/logs
export HF_HOME=${HF_HOME:-$M3_WORK/.cache/huggingface}
mkdir -p "$RESULTS" "$LOGS"

setup() {
  set -e
  "$PYTHON" -c "import torch; assert torch.version.hip, 'need a ROCm torch'" || exit 1
  "$PYTHON" -m pip install -q -e "$SGLANG_DIR/python"
  if [ ! -d "$AITER_DIR/.git" ]; then
    git clone https://github.com/ROCm/aiter "$AITER_DIR" && git -C "$AITER_DIR" checkout 4ad99832 && git -C "$AITER_DIR" submodule update --init --recursive
  fi
  # FlyDSL XCD-swizzle fix: the workgroup remap was not a bijection when the grid is not a multiple of 8
  git -C "$AITER_DIR" apply --check -R "$HERE/aiter_flydsl_xcd_swizzle_fix.patch" 2>/dev/null || git -C "$AITER_DIR" apply "$HERE/aiter_flydsl_xcd_swizzle_fix.patch"
  "$PYTHON" -m pip install -q -e "$AITER_DIR"
  mkdir -p /tmp/aiter_configs && cp "$HERE/tuned_fmoe_m3_gfx950.csv" /tmp/aiter_configs/tuned_fmoe.csv  # aiter's config dir is fixed
  if [ -z "${SKIP_MODELS:-}" ]; then
    [ -f "$MODEL/config.json" ] || "$PYTHON" -c "from huggingface_hub import snapshot_download as d; d('amd/MiniMax-M3-MXFP4', local_dir='$MODEL')"
    [ -f "$DRAFT/config.json" ] || "$PYTHON" -c "from huggingface_hub import snapshot_download as d; d('Inferact/MiniMax-M3-EAGLE3-GQA', local_dir='$DRAFT')"
  fi
  if [ ! -x "$AIPERF" ]; then  # SemiAnalysis fork at InferenceX's pinned commit; stock aiperf rejects --trace-idle-gap-cap-seconds
    "$PYTHON" -m venv "$(dirname "$(dirname "$AIPERF")")"
    [ -d "$AIPERF_DIR/.git" ] || git clone https://github.com/SemiAnalysisAI/aiperf "$AIPERF_DIR"
    git -C "$AIPERF_DIR" checkout 754356e9 && "$(dirname "$AIPERF")/pip" install -q -e "$AIPERF_DIR"
  fi
  echo "setup complete: work dir $M3_WORK"
}

installed() { [ -f "$MODEL/config.json" ] && [ -f "$DRAFT/config.json" ] && [ -x "$AIPERF" ] && "$PYTHON" -c "import sglang, aiter" 2>/dev/null; }

preflight() {
  local fail=0 g used
  installed || { [ -n "${CHECK_ONLY:-}" ] && { echo "setup incomplete under $M3_WORK (run: reproduce.sh setup)"; exit 1; }; echo "setup incomplete under $M3_WORK; running setup first"; (setup) || exit 1; }
  for g in ${GPUS//,/ }; do
    used=$(rocm-smi -d "$g" --showmemuse --csv 2>/dev/null | grep "^card" | cut -d, -f2)
    [ -n "$used" ] || { echo "GPU $g not visible to rocm-smi"; fail=1; continue; }
    [ "$used" -le 5 ] || echo "WARNING: GPU $g has ${used}% VRAM in use by another process; the run will not be clean"
  done
  [ $fail -eq 0 ] || exit 1
  echo "preflight ok: sglang=$SGLANG_DIR gpus=$GPUS port=$PORT results=$RESULTS"
}

serve() {  # $1 = real | lossy
  local mode=$1 aiter_pkg; aiter_pkg=$("$PYTHON" -c "import os, aiter; print(os.path.dirname(aiter.__file__))" 2>/dev/null)
  export HIP_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1 PYTHONPATH=$SGLANG_DIR/python${PYTHONPATH:+:$PYTHONPATH}
  export SGLANG_USE_AITER=1 NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1
  export SGLANG_M3_ALLOW_CUSTOM_AR=1 ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SGLANG_CUSTOM_AR_ONE_STAGE_MAX_BYTES=262144   # all-reduce: custom AR, INT4 quick-reduce for >=64 MB, 1-stage kernel under 256 KB
  export SGLANG_MINIMAX_OPT_USE_GLUON_PREFILL=1 SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4                                    # Gluon sparse prefill on fp8 KV; indexer top-k every 4th layer
  export SGLANG_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1  # long-cached-prefix extends: Triton route, aiter paged batch-prefill
  export SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5 SGLANG_TIMEOUT_KEEP_ALIVE=3600                                     # short extends share the chunk budget; idle client connections stay open
  if [ -n "${PTPC_FP8:-}" ]; then  # optional: quark-excluded dense layers as per-token FP8; measured -9% decode at 24 streams on 2026-09-15, so off by default
    export SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1 SGLANG_QUARK_ONLINE_FP8_SKIP_MODULES=gate,lm_head,index_qkv_proj SGLANG_FUSED_NORM_FP8_QUANT_MAX_M=16384
    export AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=$aiter_pkg/configs/a8w8_bpreshuffle_tuned_gemm.csv:$HERE/tuned_a8w8_bpreshuffle_m3_gfx950.csv
  fi
  local steps=3 draft=4 memfrac=0.85 extra="--max-running-requests 48"
  if [ "$mode" = lossy ]; then  # PERFORMANCE-ONLY: forced acceptance length 2.78 over 3 draft tokens = ATOM's --spec-decode-acceptance-rate 0.5933
    export SGLANG_SIMULATE_ACC_LEN=2.78 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0
    steps=2 draft=3 memfrac=0.9 extra="--schedule-policy lpm --max-running-requests 32"
  fi
  local log=$LOGS/server_$mode.log
  { echo "GIT: $(git -C "$SGLANG_DIR" rev-parse HEAD) mode=$mode"; env | grep -E "^(SGLANG|AITER|ROCM|HIP|NCCL)_"; } > "$log"
  setsid nohup "$PYTHON" -m sglang.launch_server --model-path "$MODEL" --served-model-name MiniMax-M3 --trust-remote-code \
    --tp 4 --host 0.0.0.0 --port "$PORT" --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 8192 --mem-fraction-static $memfrac \
    --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" --speculative-num-steps $steps --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens $draft --speculative-attention-mode decode --triton-attention-num-kv-splits 64 \
    --cuda-graph-backend-prefill breakable --reasoning-parser auto --tool-call-parser auto --enable-metrics --enable-cache-report \
    --watchdog-timeout 3600 $extra >> "$log" 2>&1 &
  echo $! > "$LOGS/server.pid"
  for _ in $(seq 1 180); do curl -sf -m 2 "localhost:$PORT/health" >/dev/null && { echo "server up ($mode): $log"; return 0; }; sleep 10; done
  echo "server did not come up; see $log"; exit 1
}

stop() {  # the server runs in its own session (setsid), so its process group is the whole TP4 server
  local pid; pid=$(cat "$LOGS/server.pid" 2>/dev/null) || return 0
  kill -TERM -- "-$pid" 2>/dev/null && sleep 8; kill -9 -- "-$pid" 2>/dev/null; rm -f "$LOGS/server.pid"; true
}

bench() {  # $1 = mode, $2 = concurrency; ATOM's client environment and flags (recipes/MiniMax-M3-Agentic-InferenceX.md)
  local out=$RESULTS/aiperf_$1_c$2; [ -d "$out" ] && mv "$out" "${out}_prev_$(date -u +%H%M%S)"; mkdir -p "$out"
  AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800 \
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300 AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0 AIPERF_FAILED_REQUEST_THRESHOLD=0.10 \
  AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10 AIPERF_WARMUP_REQUESTS_PER_LANE=10 AIPERF_BENCHMARK_GRACE_PERIOD=30 \
  "$AIPERF" profile --scenario inferencex-agentx-mvp --url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions --endpoint-type chat \
    --streaming --model MiniMax-M3 --tokenizer "$MODEL" --tokenizer-trust-remote-code --apply-chat-template \
    --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393 --concurrency "$2" --benchmark-duration "$DURATION" \
    --random-seed 42 --use-server-token-count --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --trace-idle-gap-cap-seconds 300 --warmup-requests-per-lane 10 --warmup-grace-period 1800 --failed-request-threshold 0.10 \
    --stats-interval 30 --slice-duration 1.0 --ui simple --output-artifact-dir "$out" > "$out/aiperf_stdout.log" 2>&1
  "$PYTHON" - "$out" <<'PY'
import json, sys
d = sys.argv[1]; s = json.load(open(f"{d}/profile_export_aiperf.json"))
recs = [json.loads(l) for l in open(f"{d}/profile_export.jsonl")]
ok = [r for r in recs if r.get("error") is None and r["metadata"]["benchmark_phase"] == "profiling"]
v = lambda r, k: r["metrics"][k]["value"] if isinstance(r["metrics"][k], dict) else r["metrics"][k]
tok = sum(v(r, "input_sequence_length") + v(r, "output_sequence_length") for r in ok)
span = (max(r["metadata"]["request_end_ns"] for r in ok) - min(r["metadata"]["request_start_ns"] for r in ok)) / 1e9
p = lambda k, st: s[k][st]
print(f"{d.split('/')[-1]}: tok/s per GPU reported {p('total_token_throughput','avg')/4:,.0f} | in-window {tok/span/4:,.0f} over {span:.0f} s, {len(ok)} requests"
      f" | TTFT ms p50/p90 {p('time_to_first_token','p50'):.0f}/{p('time_to_first_token','p90'):.0f} | ITL ms p50/p90 {p('inter_token_latency','p50'):.1f}/{p('inter_token_latency','p90'):.1f}"
      f" | interactivity p90 {1000/p('inter_token_latency','p90'):.0f} tok/s/user")
PY
}

case ${1:-} in
  setup) setup ;;
  stop) stop ;;
  serve) preflight; stop; serve "${2:?real|lossy}" ;;
  real|lossy) mode=$1; shift; preflight; [ -n "${CHECK_ONLY:-}" ] && exit 0; stop; serve "$mode"; for c in ${*:-1 8 24 32}; do bench "$mode" "$c"; done; stop ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac

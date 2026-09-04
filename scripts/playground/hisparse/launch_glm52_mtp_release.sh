#!/usr/bin/env bash
set -euo pipefail

variant="${1:?variant: hbm, native, or demand}"
accept_mode="${2:?accept mode: natural or fixed70}"
run_dir="${3:?absolute run directory}"
source_root="${SOURCE_ROOT:-/home/local/workspace/cedar-orbit-src-release}"
runtime_root="${RUNTIME_ROOT:-/home/local/workspace/cedar-orbit-runtime-release}"
model_path="${MODEL_PATH:-/home/models/GLM-5.2-W4AFP8}"
max_total_tokens="${MAX_TOTAL_TOKENS:-300000}"
device_buffer_size=4096
native_hisparse_config="{\"top_k\":2048,\"device_buffer_size\":${device_buffer_size},\"host_to_device_ratio\":1}"
demand_hisparse_config="{\"top_k\":2048,\"device_buffer_size\":${device_buffer_size},\"host_to_device_ratio\":1,\"mtp_demand_buffer\":true}"
cuda_graph_args=()
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  cuda_graph_args+=(--disable-cuda-graph)
fi

case "$variant" in
  hbm)
    hisparse_args=()
    ;;
  native)
    hisparse_args=(
      --enable-hisparse
      --hisparse-config "$native_hisparse_config"
    )
    ;;
  demand)
    hisparse_args=(
      --enable-hisparse
      --hisparse-config "$demand_hisparse_config"
    )
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

case "$accept_mode" in
  natural)
    simulate_env=()
    ;;
  fixed70)
    simulate_env=(
      SGLANG_SIMULATE_ACC_LEN=3.1
      SGLANG_SIMULATE_ACC_METHOD=match-expected
      SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed
      SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=1
    )
    ;;
  *)
    echo "unknown accept mode: $accept_mode" >&2
    exit 2
    ;;
esac

if [[ "$run_dir" != /* ]]; then
  echo "run directory must be absolute: $run_dir" >&2
  exit 2
fi
test -d "$source_root/python/sglang"
test -d "$runtime_root/sgl_kernel"
test -d "$model_path"
if [[ ! "$max_total_tokens" =~ ^[0-9]+$ ]] || (( max_total_tokens < 260000 )); then
  echo "MAX_TOTAL_TOKENS must be an integer >= 260000 for balanced B16" >&2
  exit 2
fi

live_pids=()
while IFS= read -r pid; do
  state="$(ps -p "$pid" -o stat= 2>/dev/null || true)"
  if [[ -n "$state" && "$state" != Z* ]]; then
    live_pids+=("$pid")
  fi
done < <(pgrep -f 'sglang.launch_server|sglang::scheduler|sglang::detokenizer' || true)

if (( ${#live_pids[@]} != 0 )); then
  echo "refusing to launch while a live SGLang process is visible" >&2
  ps -o pid,ppid,stat,args -p "$(IFS=,; echo "${live_pids[*]}")" >&2
  exit 3
fi

test ! -e "$run_dir"
mkdir -p "$run_dir"
cd "$source_root"

setsid env \
  PYTHONPATH="$source_root/python:$runtime_root" \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
  SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  SGLANG_DSA_FUSE_TOPK=1 \
  SGLANG_OPT_USE_TOPK_V2=1 \
  "${simulate_env[@]}" \
  /usr/bin/python3.12 -m sglang.launch_server \
  --model-path "$model_path" \
  --tokenizer-path "$model_path" \
  --host 127.0.0.1 --port 31082 \
  --tp-size 8 --dp-size 8 --enable-dp-attention \
  --mem-fraction-static 0.85 --max-running-requests 16 \
  --max-total-tokens "$max_total_tokens" \
  --chunked-prefill-size 1024 --max-prefill-tokens 8192 \
  --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsa --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv --dsa-topk-backend sgl-kernel \
  "${hisparse_args[@]}" \
  --disable-radix-cache --disable-prefill-cuda-graph --cuda-graph-max-bs 2 \
  "${cuda_graph_args[@]}" \
  --json-model-override-args '{"index_share_for_mtp_iteration":false}' \
  --random-seed 1234 --skip-tokenizer-init --trust-remote-code --skip-server-warmup \
  --decode-log-interval 1 --watchdog-timeout 300 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-accept-threshold-single 1.0 \
  --speculative-accept-threshold-acc 1.0 \
  --speculative-attention-mode prefill \
  >"$run_dir/launcher.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" >"$run_dir/leader.pid"
echo "launched variant=$variant accept_mode=$accept_mode pid=$pid run_dir=$run_dir"

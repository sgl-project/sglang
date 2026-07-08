#!/bin/bash

set -eu
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang@sha256:1d8d7976fe11a8341408b92527200502e93dd69df0a63a81c57b92e70ec6fada}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.6-27B}"
TARGET_REV="${TARGET_REV:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"
DFLASH_MODEL="${DFLASH_MODEL:-z-lab/Qwen3.6-27B-DFlash}"
DFLASH_REV="${DFLASH_REV:-0919688658996800f86b895034249700e9481106}"
WEAVER_REPO="${WEAVER_REPO:-trymirai/weaver}"
WEAVER_REV="${WEAVER_REV:-309ceb4b1a6c44e6a3dfaeab8db1547e904254f8}"
WEAVER_FILE="${WEAVER_FILE:-weaver/qwen36_27b_weaver.pth}"
WEAVER_CKPT="${WEAVER_CKPT:-/artifacts/weaver/weaver/qwen36_27b_weaver.pth}"
WEAVER_SHA256="${WEAVER_SHA256:-71f540b143fb6bab14ba724c20e97a72ce198de103cfd228d31c3ce339227833}"

PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"

usage() {
  cat <<EOF
Usage:
  sh reproduction.sh exec [command...]
      Run a command in the pinned SGLang Docker image.

  sh reproduction.sh serve-ar
  sh reproduction.sh serve-dflash
  sh reproduction.sh serve-tfm
      Launch one serving configuration on the selected PORT (default 30000).

  sh reproduction.sh download
      Download and verify the Weaver checkpoint used by DFlash-TfM.

  sh reproduction.sh bench
      Run the benchmark harness against BASE_URL (default http://127.0.0.1:\$PORT).

Examples:
  CUDA_VISIBLE_DEVICES=3 PORT=30003 sh reproduction.sh serve-tfm
  PORT=30003 sh reproduction.sh bench

EOF
}

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing command: $1" >&2
    exit 1
  }
}

export_common_env() {
  export TARGET_MODEL TARGET_REV
  export DFLASH_MODEL DFLASH_REV
  export WEAVER_REPO WEAVER_REV WEAVER_FILE WEAVER_CKPT WEAVER_SHA256
  export PORT BASE_URL
}

cmd_exec() {
  need docker
  mkdir -p artifacts
  if [ "$#" -eq 0 ]; then
    set -- /bin/zsh
  fi
  docker_tty="-i"
  if [ -t 0 ]; then
    docker_tty="-it"
  fi
  docker pull "$SGLANG_IMAGE"
  docker run $docker_tty --rm --shm-size 32g --gpus all \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$PWD":/sgl-workspace/sglang \
    -v "$PWD/artifacts":/artifacts \
    -w /sgl-workspace/sglang \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e PYTHONPATH=/sgl-workspace/sglang/python \
    -e SGL_REPRO_IN_CONTAINER=1 \
    -e TARGET_MODEL="$TARGET_MODEL" \
    -e TARGET_REV="$TARGET_REV" \
    -e DFLASH_MODEL="$DFLASH_MODEL" \
    -e DFLASH_REV="$DFLASH_REV" \
    -e WEAVER_REPO="$WEAVER_REPO" \
    -e WEAVER_REV="$WEAVER_REV" \
    -e WEAVER_FILE="$WEAVER_FILE" \
    -e WEAVER_CKPT="$WEAVER_CKPT" \
    -e WEAVER_SHA256="$WEAVER_SHA256" \
    -e PORT="$PORT" \
    -e BASE_URL="$BASE_URL" \
    --ipc=host --network=host --privileged \
    "$SGLANG_IMAGE" \
    "$@"
}

run_in_container() {
  subcommand="$1"
  shift
  if [ "${SGL_REPRO_IN_CONTAINER:-0}" = "1" ]; then
    case "$subcommand" in
      serve-ar) cmd_serve_ar "$@" ;;
      serve-dflash) cmd_serve_dflash "$@" ;;
      serve-tfm) cmd_serve_tfm "$@" ;;
      download) cmd_download "$@" ;;
      bench) cmd_bench "$@" ;;
      *)
        echo "error: unknown in-container command: $subcommand" >&2
        exit 1
        ;;
    esac
  else
    cmd_exec sh ./reproduction.sh "$subcommand" "$@"
  fi
}

cmd_serve_ar() {
  export_common_env
  python3 -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --revision "$TARGET_REV" \
    --dtype bfloat16 \
    --tp-size 1 \
    --max-running-requests 1 \
    --cuda-graph-max-bs 32 \
    --mem-fraction-static 0.75 \
    --page-size 64 \
    --disable-radix-cache \
    --decode-attention-backend trtllm_mha \
    --prefill-attention-backend flashinfer \
    --host 127.0.0.1 \
    --port "$PORT"
}

cmd_serve_dflash() {
  export_common_env
  python3 -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --revision "$TARGET_REV" \
    --dtype bfloat16 \
    --tp-size 1 \
    --max-running-requests 1 \
    --cuda-graph-max-bs 32 \
    --mem-fraction-static 0.75 \
    --page-size 64 \
    --disable-radix-cache \
    --attention-backend trtllm_mha \
    --speculative-draft-attention-backend fa4 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DFLASH_MODEL" \
    --speculative-draft-model-revision "$DFLASH_REV" \
    --speculative-dflash-block-size 16 \
    --speculative-num-draft-tokens 16 \
    --host 127.0.0.1 \
    --port "$PORT"
}

ensure_weaver_checkpoint() {
  need hf
  need sha256sum
  mkdir -p /artifacts/weaver
  if [ ! -f "$WEAVER_CKPT" ]; then
    hf download "$WEAVER_REPO" \
      "$WEAVER_FILE" \
      --revision "$WEAVER_REV" \
      --local-dir /artifacts/weaver
  fi
  actual_sha256="$(sha256sum "$WEAVER_CKPT" | awk '{print $1}')"
  if [ "$actual_sha256" != "$WEAVER_SHA256" ]; then
    echo "error: Weaver checkpoint hash mismatch" >&2
    echo "  path: $WEAVER_CKPT" >&2
    echo "  expected: $WEAVER_SHA256" >&2
    echo "  actual:   $actual_sha256" >&2
    exit 1
  fi
}

cmd_download() {
  export_common_env
  ensure_weaver_checkpoint
  echo "Weaver checkpoint is ready: $WEAVER_CKPT"
}

cmd_serve_tfm() {
  export_common_env
  ensure_weaver_checkpoint
  python3 -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --revision "$TARGET_REV" \
    --dtype bfloat16 \
    --tp-size 1 \
    --max-running-requests 1 \
    --cuda-graph-max-bs 32 \
    --mem-fraction-static 0.75 \
    --page-size 64 \
    --disable-radix-cache \
    --decode-attention-backend trtllm_mha \
    --prefill-attention-backend flashinfer \
    --speculative-draft-attention-backend fa4 \
    --speculative-algorithm DFLASH_TFM \
    --speculative-draft-model-path "$DFLASH_MODEL" \
    --speculative-draft-model-revision "$DFLASH_REV" \
    --speculative-dflash-tfm-path "$WEAVER_CKPT" \
    --speculative-dflash-tfm-tree-budget 64 \
    --speculative-gdn-verify-kernel chunk \
    --disable-overlap-schedule \
    --host 127.0.0.1 \
    --port "$PORT"
}

cmd_bench() {
  export_common_env
  python3 -m sglang.bench_dflash_tfm \
    --base-url "$BASE_URL" \
    --model "$TARGET_MODEL" \
    --datasets mtbench sharechat gsm8k math500 aime25 humaneval mbpp livecodebench \
    --temperature 1.0 \
    --reasoning on \
    --max-new-tokens 4096 \
    --concurrency 1 \
    --flush-cache-between-requests
}

command="${1:-}"
if [ "$#" -gt 0 ]; then
  shift
fi

case "$command" in
  exec) cmd_exec "$@" ;;
  serve-ar) run_in_container serve-ar "$@" ;;
  serve-dflash) run_in_container serve-dflash "$@" ;;
  serve-tfm) run_in_container serve-tfm "$@" ;;
  download) run_in_container download "$@" ;;
  bench) run_in_container bench "$@" ;;
  ""|-h|--help|help) usage ;;
  *)
    echo "error: unknown command: $command" >&2
    usage >&2
    exit 1
    ;;
esac

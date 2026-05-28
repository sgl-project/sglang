#!/usr/bin/env bash
set -euo pipefail

# Download/resume the large linear-attention model weights used by the SSM
# dtype accuracy matrix. This intentionally clears HTTP(S)_PROXY for the hf
# process because direct access was more reliable in this workspace.
#
# Default:
#   bash scripts/download_linear_attn_models.sh
#
# Foreground mode:
#   BACKGROUND=0 bash scripts/download_linear_attn_models.sh
#
# Select one model:
#   MODELS="qwen35_fp8" bash scripts/download_linear_attn_models.sh

MODEL_ROOT="${MODEL_ROOT:-/lustre/raplab/client/xutingz/workspace/model}"
MODELS="${MODELS:-qwen35_fp8 kimi_linear}"
BACKGROUND="${BACKGROUND:-1}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

mkdir -p "${MODEL_ROOT}/download_logs" "${MODEL_ROOT}/hub" "${MODEL_ROOT}/xet"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}"
}

repo_for_model() {
  case "$1" in
    qwen35_fp8) printf '%s\n' "Qwen/Qwen3.5-397B-A17B-FP8" ;;
    kimi_linear) printf '%s\n' "moonshotai/Kimi-Linear-48B-A3B-Instruct" ;;
    *) return 1 ;;
  esac
}

dir_for_model() {
  case "$1" in
    qwen35_fp8) printf '%s\n' "${MODEL_ROOT}/Qwen/Qwen3.5-397B-A17B-FP8" ;;
    kimi_linear) printf '%s\n' "${MODEL_ROOT}/Kimi/Kimi-Linear-48B-A3B-Instruct" ;;
    *) return 1 ;;
  esac
}

workers_for_model() {
  case "$1" in
    qwen35_fp8) printf '%s\n' "${HF_MAX_WORKERS_QWEN:-8}" ;;
    kimi_linear) printf '%s\n' "${HF_MAX_WORKERS_KIMI:-4}" ;;
    *) return 1 ;;
  esac
}

download_one() {
  local model_key="$1"
  local repo local_dir workers log_file
  repo="$(repo_for_model "${model_key}")"
  local_dir="$(dir_for_model "${model_key}")"
  workers="$(workers_for_model "${model_key}")"
  log_file="${MODEL_ROOT}/download_logs/${model_key}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "${local_dir}"
  find "${local_dir}" -path '*/.cache/huggingface/*.lock' -type f -delete 2>/dev/null || true

  if ps -eo cmd | grep -F "hf download ${repo} --local-dir ${local_dir}" | grep -v grep >/dev/null; then
    log "Already running: ${repo} -> ${local_dir}"
    return 0
  fi

  log "Download ${repo} -> ${local_dir}; log=${log_file}; workers=${workers}; background=${BACKGROUND}"

  if [[ "${BACKGROUND}" == "1" ]]; then
    setsid bash -c '
      set -euo pipefail
      repo="$1"; local_dir="$2"; workers="$3"; model_root="$4"; log_file="$5"
      {
        printf "[%s] start %s -> %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "${repo}" "${local_dir}"
        env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
          HF_HOME="${model_root}/hub" \
          HF_XET_CACHE="${model_root}/xet" \
          HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}" \
          HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}" \
          hf download "${repo}" --local-dir "${local_dir}" --max-workers "${workers}"
        rc=$?
        printf "[%s] done rc=%s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "${rc}"
        exit "${rc}"
      } >> "${log_file}" 2>&1
    ' _ "${repo}" "${local_dir}" "${workers}" "${MODEL_ROOT}" "${log_file}" </dev/null >/dev/null 2>&1 &
    log "Started pid=$!"
  else
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      HF_HOME="${MODEL_ROOT}/hub" \
      HF_XET_CACHE="${MODEL_ROOT}/xet" \
      HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT}" \
      hf download "${repo}" --local-dir "${local_dir}" --max-workers "${workers}" 2>&1 | tee -a "${log_file}"
  fi
}

for model_key in ${MODELS}; do
  download_one "${model_key}"
done

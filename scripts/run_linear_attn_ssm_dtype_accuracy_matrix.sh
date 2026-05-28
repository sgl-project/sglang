#!/usr/bin/env bash
set -euo pipefail

# Run the SGLang official simple eval path across the linear/hybrid attention
# models we care about, comparing SSM state dtypes. By default this runs full
# MMLU, GPQA, and GSM8K for each model and compares float32 vs bfloat16.
#
# Default:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash scripts/run_linear_attn_ssm_dtype_accuracy_matrix.sh
#
# Common overrides:
#   MODELS="qwen3_next qwen35 kimi_linear" \
#   DTYPES="float32 bfloat16" \
#   EVALS="mmlu gpqa gsm8k" \
#   QWEN35_MODEL_PATH=/lustre/raplab/client/xutingz/workspace/model/Qwen/Qwen3.5-397B-A17B-FP8 \
#   KIMI_LINEAR_MODEL_PATH=/lustre/raplab/client/xutingz/workspace/model/Kimi/Kimi-Linear-48B-A3B-Instruct \
#   bash scripts/run_linear_attn_ssm_dtype_accuracy_matrix.sh
#
# Smoke tests can still use NUM_EXAMPLES or per-eval overrides:
#   NUM_EXAMPLES=100 MODELS="qwen35" bash scripts/run_linear_attn_ssm_dtype_accuracy_matrix.sh
#
# Check resolved model paths and arguments without launching servers:
#   DRY_RUN=1 bash scripts/run_linear_attn_ssm_dtype_accuracy_matrix.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="${RUNNER:-${REPO_ROOT}/scripts/run_qwen3_next_ssm_dtype_accuracy.sh}"
MODEL_ROOT="${MODEL_ROOT:-/lustre/raplab/client/xutingz/workspace/model}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_OUTPUT_ROOT="${MATRIX_OUTPUT_ROOT:-${REPO_ROOT}/eval_results/ssm_dtype_accuracy_matrix/${RUN_ID}}"
MODELS="${MODELS:-qwen3_next qwen35 kimi_linear}"
DTYPES="${DTYPES:-float32 bfloat16}"
EVALS="${EVALS:-mmlu gpqa gsm8k}"
COMMON_EXTRA_SERVER_ARGS="${COMMON_EXTRA_SERVER_ARGS:-${EXTRA_SERVER_ARGS:---attention-backend triton --linear-attn-backend triton --linear-attn-decode-backend triton --linear-attn-prefill-backend triton --grammar-backend none --disable-cuda-graph}}"
COMMON_EXTRA_EVAL_ARGS="${COMMON_EXTRA_EVAL_ARGS:-${EXTRA_EVAL_ARGS:-}}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${MATRIX_OUTPUT_ROOT}"
RUNS_TSV="${MATRIX_OUTPUT_ROOT}/runs.tsv"
printf 'model_key\tmodel_label\tmodel_path\toutput_dir\n' > "${RUNS_TSV}"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${MATRIX_OUTPUT_ROOT}/matrix.log"
}

first_existing_or_default() {
  local fallback="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  printf '%s\n' "${fallback}"
}

env_or_default() {
  local name="$1"
  local default_value="$2"
  if [[ -v "${name}" ]]; then
    printf '%s\n' "${!name}"
  else
    printf '%s\n' "${default_value}"
  fi
}

# Each defaults_* function emits a per-model defaults record into DEF_* vars
# plus ENV_PREFIX. resolve_model_config() then overlays ${PREFIX}_* env vars
# uniformly so we only spell the env-override pattern once instead of per case.

defaults_qwen3_next() {
  ENV_PREFIX=QWEN3_NEXT
  DEF_MODEL_LABEL=qwen3_next_80b_a3b_instruct
  DEF_MODEL_PATH="$(first_existing_or_default \
    Qwen/Qwen3-Next-80B-A3B-Instruct \
    "${MODEL_ROOT}/Qwen3-Next-80B-A3B-Instruct" \
    "${MODEL_ROOT}/Qwen/Qwen3-Next-80B-A3B-Instruct" \
    /lustre/raplab/client/test1/workspace/model_weights/Qwen3-Next-80B-A3B-Instruct)"
  DEF_TP_SIZE=4
  DEF_EXTRA_SERVER_ARGS="${COMMON_EXTRA_SERVER_ARGS}"
  DEF_EXTRA_EVAL_ARGS="${COMMON_EXTRA_EVAL_ARGS}"
  DEF_MAX_TOKENS="${MAX_TOKENS:-2048}"
  DEF_NUM_THREADS="${NUM_THREADS:-512}"
  DEF_NUM_SHOTS="${NUM_SHOTS:-5}"
}

defaults_qwen35() {
  ENV_PREFIX=QWEN35
  DEF_MODEL_LABEL=qwen3_5_397b_a17b_fp8
  DEF_MODEL_PATH="$(first_existing_or_default \
    Qwen/Qwen3.5-397B-A17B-FP8 \
    "${MODEL_ROOT}/Qwen/Qwen3.5-397B-A17B-FP8" \
    "${MODEL_ROOT}/Qwen3.5-397B-A17B-FP8")"
  DEF_TP_SIZE=8
  DEF_EXTRA_SERVER_ARGS="${COMMON_EXTRA_SERVER_ARGS} --trust-remote-code --reasoning-parser=qwen3 --tool-call-parser=qwen3_coder --mem-fraction-static=0.8"
  DEF_EXTRA_EVAL_ARGS="${COMMON_EXTRA_EVAL_ARGS:- --thinking-mode qwen-3}"
  DEF_MAX_TOKENS="${MAX_TOKENS:-8192}"
  DEF_NUM_THREADS="${NUM_THREADS:-512}"
  DEF_NUM_SHOTS="${NUM_SHOTS:-5}"
}

defaults_kimi_linear() {
  ENV_PREFIX=KIMI_LINEAR
  DEF_MODEL_LABEL=kimi_linear_48b_a3b_instruct
  DEF_MODEL_PATH="$(first_existing_or_default \
    moonshotai/Kimi-Linear-48B-A3B-Instruct \
    "${MODEL_ROOT}/Kimi/Kimi-Linear-48B-A3B-Instruct" \
    "${MODEL_ROOT}/Kimi-Linear-48B-A3B-Instruct")"
  DEF_TP_SIZE=2
  DEF_EXTRA_SERVER_ARGS="${COMMON_EXTRA_SERVER_ARGS} --trust-remote-code"
  DEF_EXTRA_EVAL_ARGS="${COMMON_EXTRA_EVAL_ARGS}"
  DEF_MAX_TOKENS="${MAX_TOKENS:-2048}"
  DEF_NUM_THREADS="${NUM_THREADS:-512}"
  DEF_NUM_SHOTS="${NUM_SHOTS:-5}"
}

resolve_model_config() {
  local model_key="$1"
  case "${model_key}" in
    qwen3_next) defaults_qwen3_next ;;
    qwen35) defaults_qwen35 ;;
    kimi_linear) defaults_kimi_linear ;;
    *)
      log "Unknown model key: ${model_key}. Supported: qwen3_next qwen35 kimi_linear"
      return 2
      ;;
  esac
  model_label="$(env_or_default "${ENV_PREFIX}_MODEL_LABEL" "${DEF_MODEL_LABEL}")"
  model_path="$(env_or_default "${ENV_PREFIX}_MODEL_PATH" "${DEF_MODEL_PATH}")"
  tp_size="$(env_or_default "${ENV_PREFIX}_TP_SIZE" "${DEF_TP_SIZE}")"
  extra_server_args="$(env_or_default "${ENV_PREFIX}_EXTRA_SERVER_ARGS" "${DEF_EXTRA_SERVER_ARGS}")"
  extra_eval_args="$(env_or_default "${ENV_PREFIX}_EXTRA_EVAL_ARGS" "${DEF_EXTRA_EVAL_ARGS}")"
  max_tokens="$(env_or_default "${ENV_PREFIX}_MAX_TOKENS" "${DEF_MAX_TOKENS}")"
  num_threads="$(env_or_default "${ENV_PREFIX}_NUM_THREADS" "${DEF_NUM_THREADS}")"
  num_shots="$(env_or_default "${ENV_PREFIX}_NUM_SHOTS" "${DEF_NUM_SHOTS}")"
}

run_model() {
  local model_key="$1"
  local model_label model_path tp_size extra_server_args extra_eval_args max_tokens num_threads num_shots
  local ENV_PREFIX DEF_MODEL_LABEL DEF_MODEL_PATH DEF_TP_SIZE DEF_EXTRA_SERVER_ARGS DEF_EXTRA_EVAL_ARGS DEF_MAX_TOKENS DEF_NUM_THREADS DEF_NUM_SHOTS

  resolve_model_config "${model_key}" || return $?

  local output_dir="${MATRIX_OUTPUT_ROOT}/${model_label}"
  log "Running ${model_key}: label=${model_label}, path=${model_path}, tp=${tp_size}"
  log "  server args: ${extra_server_args}"
  log "  eval args: ${extra_eval_args:-<none>}"
  printf '%s\t%s\t%s\t%s\n' "${model_key}" "${model_label}" "${model_path}" "${output_dir}" >> "${RUNS_TSV}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  MODEL_LABEL="${model_label}" \
  MODEL_PATH="${model_path}" \
  TP_SIZE="${tp_size}" \
  DTYPES="${DTYPES}" \
  EVALS="${EVALS}" \
  OUTPUT_DIR="${output_dir}" \
  EXTRA_SERVER_ARGS="${extra_server_args}" \
  EXTRA_EVAL_ARGS="${extra_eval_args}" \
  MAX_TOKENS="${max_tokens}" \
  NUM_THREADS="${num_threads}" \
  NUM_SHOTS="${num_shots}" \
  bash "${RUNNER}"
}

for model_key in ${MODELS}; do
  run_model "${model_key}"
done

"${PYTHON_BIN:-python3}" "${REPO_ROOT}/scripts/ssm_dtype/summarize_matrix.py" \
  --runs-tsv "${RUNS_TSV}" \
  --matrix-output-root "${MATRIX_OUTPUT_ROOT}"

log "Done. Matrix summary: ${MATRIX_OUTPUT_ROOT}/matrix_summary.md"

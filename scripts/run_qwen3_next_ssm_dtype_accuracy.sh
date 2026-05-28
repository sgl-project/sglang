#!/usr/bin/env bash
set -euo pipefail

# Run SGLang's official simple evals for one hybrid/linear-attention model with
# different Mamba/linear-attention SSM state dtypes and summarize the accuracy
# delta.
#
# Default usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_qwen3_next_ssm_dtype_accuracy.sh
#
# Common overrides:
#   MODEL_LABEL=qwen3_next_80b_a3b_instruct \
#   MODEL_PATH=/path/to/Qwen3-Next-80B-A3B-Instruct \
#   TP_SIZE=4 \
#   DTYPES="float32 bfloat16" \
#   EVALS="mmlu gpqa gsm8k" \
#   bash scripts/run_qwen3_next_ssm_dtype_accuracy.sh
#
# By default this runs full datasets. For smoke tests, set NUM_EXAMPLES or
# per-dataset overrides such as MMLU_NUM_EXAMPLES=64.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  DEFAULT_LOCAL_MODEL_CANDIDATES=(
    "/lustre/raplab/client/xutingz/workspace/model/Qwen3-Next-80B-A3B-Instruct"
    "/lustre/raplab/client/test1/workspace/model_weights/Qwen3-Next-80B-A3B-Instruct"
  )
  MODEL_PATH="Qwen/Qwen3-Next-80B-A3B-Instruct"
  for candidate in "${DEFAULT_LOCAL_MODEL_CANDIDATES[@]}"; do
    if [[ -d "${candidate}" ]]; then
      MODEL_PATH="${candidate}"
      break
    fi
  done
fi

MODEL_LABEL="${MODEL_LABEL:-$(basename "${MODEL_PATH}" | tr -c '[:alnum:]_.-' '_')}"
TP_SIZE="${TP_SIZE:-4}"
DTYPES="${DTYPES:-float32 bfloat16}"
EVALS="${EVALS:-mmlu gpqa gsm8k}"
DEFAULT_PYTHON_BIN="/lustre/raplab/client/xutingz/workspace/miniforge3/envs/dsep_dev/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
  else
    PYTHON_BIN="python3"
  fi
fi
NUM_THREADS="${NUM_THREADS:-512}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
NUM_SHOTS="${NUM_SHOTS:-5}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-128}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-3600}"
ENABLE_DTYPE_PROBE="${ENABLE_DTYPE_PROBE:-1}"
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/eval_results/ssm_dtype_accuracy/${MODEL_LABEL}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"

SSM_DTYPE_LIB_DIR="${REPO_ROOT}/scripts/ssm_dtype"
SSM_DTYPE_PROBE_DIR="${SSM_DTYPE_LIB_DIR}/dtype_probe"

PORT="${PORT:-$("${PYTHON_BIN}" "${SSM_DTYPE_LIB_DIR}/find_free_port.py")}"
BASE_URL="http://127.0.0.1:${PORT}"
PYTHONPATH_BASE="${REPO_ROOT}/python"
if [[ -d "/tmp/flashinfer_py310_064" ]]; then
  PYTHONPATH_BASE="/tmp/flashinfer_py310_064:${PYTHONPATH_BASE}"
fi
if [[ -d "/tmp/sglang_py310_compat" ]]; then
  PYTHONPATH_BASE="${PYTHONPATH_BASE}:/tmp/sglang_py310_compat"
fi
if [[ -n "${PYTHONPATH_EXTRA:-}" ]]; then
  PYTHONPATH_BASE="${PYTHONPATH_EXTRA}:${PYTHONPATH_BASE}"
fi
PYTHONPATH_BASE="${PYTHONPATH_BASE}${PYTHONPATH:+:${PYTHONPATH}}"
LOCAL_NO_PROXY="${LOCAL_NO_PROXY:-127.0.0.1,localhost,0.0.0.0}"
NO_PROXY_VALUE="${LOCAL_NO_PROXY}${NO_PROXY:+,${NO_PROXY}}"
LOCAL_PROXY_ENV=(
  "NO_PROXY=${NO_PROXY_VALUE}"
  "no_proxy=${NO_PROXY_VALUE}"
  "HTTP_PROXY="
  "HTTPS_PROXY="
  "ALL_PROXY="
  "http_proxy="
  "https_proxy="
  "all_proxy="
)

mkdir -p "${OUTPUT_DIR}"

RUNNER_LOG="${OUTPUT_DIR}/runner.log"
SERVER_PID=""

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${RUNNER_LOG}"
}

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "Stopping server process group ${SERVER_PID}"
    kill -INT "-${SERVER_PID}" 2>/dev/null || kill -INT "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        SERVER_PID=""
        return
      fi
      sleep 2
    done
    kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}

trap cleanup_server EXIT

wait_for_server() {
  local server_log="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT))
  while (( SECONDS < deadline )); do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "Server exited during startup. Last 80 log lines:"
      tail -n 80 "${server_log}" | tee -a "${RUNNER_LOG}" || true
      return 1
    fi

    if env "${LOCAL_PROXY_ENV[@]}" curl --noproxy '*' -fsS "${BASE_URL}/health_generate" >/dev/null 2>&1; then
      log "Server is healthy at ${BASE_URL}"
      return 0
    fi

    sleep 10
  done

  log "Timed out waiting for server. Last 80 log lines:"
  tail -n 80 "${server_log}" | tee -a "${RUNNER_LOG}" || true
  return 1
}

start_server() {
  local dtype="$1"
  local server_log="${OUTPUT_DIR}/server_${dtype}.log"
  local server_pythonpath="${SSM_DTYPE_PROBE_DIR}:${PYTHONPATH_BASE}"

  local server_cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host 0.0.0.0
    --port "${PORT}"
    --tp-size "${TP_SIZE}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
    --mamba-track-interval "${MAMBA_TRACK_INTERVAL}"
    --mamba-ssm-dtype "${dtype}"
  )

  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    server_cmd+=(--trust-remote-code)
  fi

  if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_server_args=( ${EXTRA_SERVER_ARGS} )
    server_cmd+=("${extra_server_args[@]}")
  fi

  log "Starting ${MODEL_LABEL} server with mamba SSM dtype=${dtype}"
  log "Server log: ${server_log}"
  printf 'Command: PYTHONPATH=%q SGLANG_SSM_DTYPE_PROBE=%q %q' \
    "${server_pythonpath}" "${ENABLE_DTYPE_PROBE}" "${server_cmd[0]}" \
    > "${server_log}"
  printf ' %q' "${server_cmd[@]:1}" >> "${server_log}"
  printf '\n\n' >> "${server_log}"

  setsid env \
    "${LOCAL_PROXY_ENV[@]}" \
    PYTHONPATH="${server_pythonpath}" \
    SGLANG_SSM_DTYPE_PROBE="${ENABLE_DTYPE_PROBE}" \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK}" \
    "${server_cmd[@]}" >> "${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"
}

num_examples_arg_for_eval() {
  local eval_name="$1"
  local upper
  upper="$(printf '%s' "${eval_name}" | tr '[:lower:]' '[:upper:]')"
  local var_name="${upper}_NUM_EXAMPLES"
  local value="${!var_name:-${NUM_EXAMPLES:-}}"
  if [[ -n "${value}" ]]; then
    printf '%s\n' "${value}"
  fi
}

copy_eval_artifact() {
  local log_file="$1"
  local pattern="$2"
  local dst="$3"
  local path
  path="$(awk -v pat="${pattern}" '$0 ~ pat {print $NF}' "${log_file}" | tail -n 1)"
  if [[ -n "${path}" && -f "${path}" ]]; then
    cp "${path}" "${dst}"
  fi
}

run_one_eval() {
  local dtype="$1"
  local eval_name="$2"
  local eval_log="${OUTPUT_DIR}/eval_${dtype}_${eval_name}.log"
  local metrics_dst="${OUTPUT_DIR}/${dtype}_${eval_name}.metrics.json"
  local report_dst="${OUTPUT_DIR}/${dtype}_${eval_name}.html"

  local eval_cmd=(
    "${PYTHON_BIN}" -m sglang.test.run_eval
    --base-url "${BASE_URL}"
    --eval-name "${eval_name}"
    --num-threads "${NUM_THREADS}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
  )

  if [[ "${eval_name}" == "gsm8k" ]]; then
    eval_cmd+=(--num-shots "${NUM_SHOTS}")
  fi

  local num_examples
  num_examples="$(num_examples_arg_for_eval "${eval_name}")"
  if [[ -n "${num_examples}" ]]; then
    eval_cmd+=(--num-examples "${num_examples}")
  fi

  if [[ -n "${EXTRA_EVAL_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra_eval_args=( ${EXTRA_EVAL_ARGS} )
    eval_cmd+=("${extra_eval_args[@]}")
  fi

  log "Running official SGLang eval=${eval_name} dtype=${dtype}"
  printf 'Command: PYTHONPATH=%q %q' "${PYTHONPATH_BASE}" "${eval_cmd[0]}" > "${eval_log}"
  printf ' %q' "${eval_cmd[@]:1}" >> "${eval_log}"
  printf '\n\n' >> "${eval_log}"

  env "${LOCAL_PROXY_ENV[@]}" PYTHONPATH="${PYTHONPATH_BASE}" "${eval_cmd[@]}" >> "${eval_log}" 2>&1

  copy_eval_artifact "${eval_log}" "Writing results to" "${metrics_dst}"
  copy_eval_artifact "${eval_log}" "Writing report to" "${report_dst}"

  if [[ ! -f "${metrics_dst}" ]]; then
    log "Could not find metrics JSON for dtype=${dtype}, eval=${eval_name}"
    tail -n 80 "${eval_log}" | tee -a "${RUNNER_LOG}" || true
    return 1
  fi

  log "Saved metrics: ${metrics_dst}"
}

write_run_config() {
  env \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    MODEL_LABEL="${MODEL_LABEL}" \
    MODEL_PATH="${MODEL_PATH}" \
    TP_SIZE="${TP_SIZE}" \
    PORT="${PORT}" \
    BASE_URL="${BASE_URL}" \
    DTYPES="${DTYPES}" \
    EVALS="${EVALS}" \
    NUM_THREADS="${NUM_THREADS}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    TEMPERATURE="${TEMPERATURE}" \
    TOP_P="${TOP_P}" \
    NUM_SHOTS="${NUM_SHOTS}" \
    CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE}" \
    MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY}" \
    MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL}" \
    ENABLE_DTYPE_PROBE="${ENABLE_DTYPE_PROBE}" \
    EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}" \
    EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    PYTHONPATH_BASE="${PYTHONPATH_BASE}" \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK}" \
    LOCAL_NO_PROXY="${LOCAL_NO_PROXY}" \
    "${PYTHON_BIN}" "${SSM_DTYPE_LIB_DIR}/write_run_config.py"
}

summarize_results() {
  "${PYTHON_BIN}" "${SSM_DTYPE_LIB_DIR}/summarize_per_model.py" \
    --output-dir "${OUTPUT_DIR}" \
    --model-label "${MODEL_LABEL}" \
    --dtypes "${DTYPES}" \
    --evals "${EVALS}"
}

write_run_config
log "Output dir: ${OUTPUT_DIR}"
log "Model label: ${MODEL_LABEL}"
log "Model path: ${MODEL_PATH}"
log "Python: ${PYTHON_BIN}"
log "Dtypes: ${DTYPES}"
log "Evals: ${EVALS}"
log "Full dataset mode: $(if [[ -z "${NUM_EXAMPLES:-}${MMLU_NUM_EXAMPLES:-}${GPQA_NUM_EXAMPLES:-}${GSM8K_NUM_EXAMPLES:-}" ]]; then echo yes; else echo no; fi)"

for dtype in ${DTYPES}; do
  start_server "${dtype}"
  for eval_name in ${EVALS}; do
    run_one_eval "${dtype}" "${eval_name}"
  done
  cleanup_server
done

summarize_results
log "Done. Summary: ${OUTPUT_DIR}/summary.md"

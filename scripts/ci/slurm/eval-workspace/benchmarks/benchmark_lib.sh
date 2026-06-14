#!/usr/bin/env bash

# Eval helpers for the sglang GB200 nightly pipeline. Sourced by srt-slurm's
# lm-eval runner, which calls `run_eval --framework lm-eval --port <PORT>`
# followed by `append_lm_eval_summary`. Eval artifacts (meta_env.json,
# results*.json, sample*.jsonl) land in the calling script's PWD; srtctl
# copies them to /logs/eval_results/.

# Keep Python bytecode out of the mounted workspace.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/sglang-eval-pycache}"
mkdir -p "$PYTHONPYCACHEPREFIX" 2>/dev/null || true


# ------------------------------
# lm-eval install + runtime patches
# ------------------------------

_install_lm_eval_deps() {
    # torchvision causes circular imports in ATOM; sgl-stack needs it at module level.
    if [[ "${IMAGE:-}" == *atom* ]]; then
        python3 -m pip uninstall -y torchvision 2>/dev/null || true
    fi
    python3 -m pip install -q --no-cache-dir --break-system-packages "lm-eval[api]" || true
    local lm_eval_ref="b315ef3b05176acc9732bb7fdec116abe1ecc476"
    if command -v git >/dev/null 2>&1; then
        if ! python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
            "git+https://github.com/EleutherAI/lm-evaluation-harness.git@${lm_eval_ref}"; then
            python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
                "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
        fi
    else
        python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall --break-system-packages \
            "https://github.com/EleutherAI/lm-evaluation-harness/archive/${lm_eval_ref}.tar.gz" || true
    fi
}

# Patch lm-eval to handle empty content fields and skip injecting "type": "text"
# in chat templates. Done via sitecustomize so changes are picked up automatically.
_patch_lm_eval() {
    local patch_dir
    patch_dir="$(mktemp -d)"
    cat > "$patch_dir/sitecustomize.py" <<'PY'
import json
from lm_eval.models.openai_completions import LocalChatCompletion as _LCC


def _le_parse_generations(outputs, **kwargs):
    res = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in (outputs or []):
        try:
            choices = out.get("choices", [])
            tmp = ["" for _ in choices]
            for choice in choices:
                idx = choice.get("index", 0)
                msg = (choice.get("message") or {})
                content = msg.get("content")
                if content in (None, "", []):
                    content = msg.get("reasoning_content") or ""
                tmp[idx] = content
        except Exception:
            tmp = [""]
        res.extend(tmp)
    return res


_LCC.parse_generations = staticmethod(_le_parse_generations)


try:
    from lm_eval.models import api_models as _api_models
    _TemplateAPI = _api_models.TemplateAPI
    _JsonChatStr = _api_models.JsonChatStr
except Exception:
    _TemplateAPI = None
    _JsonChatStr = None


if _TemplateAPI is not None and _JsonChatStr is not None:
    def _patched_apply_chat_template(self, chat_history, add_generation_prompt: bool = True):
        if self.tokenizer_backend == "huggingface" and self.tokenized_requests:
            return self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        if self.tokenizer_backend == "remote" and self.tokenized_requests:
            return chat_history
        return _JsonChatStr(
            json.dumps([{**item} for item in chat_history], ensure_ascii=False)
        )

    _TemplateAPI.apply_chat_template = _patched_apply_chat_template
PY
    export PYTHONPATH="${patch_dir}:${PYTHONPATH:-}"
}


# ------------------------------
# Eval context-length sizing
# ------------------------------

# Returns the model's native max position embeddings via transformers AutoConfig.
get_native_max_context_length() {
    local model_path="$1"
    if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH}" ]; then
        model_path="${MODEL_PATH}"
    fi
    python3 -c "
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('${model_path}', trust_remote_code=True)
    for attr in ['max_position_embeddings', 'max_sequence_length', 'seq_length', 'n_positions']:
        if hasattr(config, attr):
            print(getattr(config, attr))
            break
    else:
        print(0)
except Exception:
    print(0)
"
}

# Eval context = min(requested, native_max). Falls back to MAX_MODEL_LEN env
# var or 16384 if neither known. Sets EVAL_MAX_MODEL_LEN.
compute_eval_context_length() {
    local model="$1"
    local benchmark_ctx="${2:-0}"
    local native_max
    native_max=$(get_native_max_context_length "$model")
    native_max="${native_max:-0}"

    if [ "$benchmark_ctx" -eq 0 ] 2>/dev/null; then
        benchmark_ctx="${native_max:-0}"
    fi
    local eval_ctx=$(( benchmark_ctx * 1 ))
    if [ "$native_max" -gt 0 ] 2>/dev/null && [ "$eval_ctx" -gt "$native_max" ]; then
        eval_ctx="$native_max"
    fi
    if [ "$eval_ctx" -le 0 ] 2>/dev/null; then
        echo "WARN: compute_eval_context_length could not determine context length for $model" >&2
        eval_ctx="${MAX_MODEL_LEN:-16384}"
    fi
    EVAL_MAX_MODEL_LEN="$eval_ctx"
    echo "$eval_ctx"
}


# ------------------------------
# lm-eval runner
# ------------------------------

run_lm_eval() {
    local port="${PORT:-8888}"
    local tasks_dir="${EVAL_TASKS_DIR:-utils/evals/gsm8k.yaml}"
    local results_dir="${EVAL_RESULT_DIR:-$(mktemp -d /tmp/eval_out-XXXXXX)}"
    local eval_context_len="${EVAL_MAX_MODEL_LEN:-16384}"
    local temperature=0
    local top_p=1
    local concurrent_requests="${EVAL_CONCURRENT_REQUESTS:-64}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)           port="$2"; shift 2 ;;
            --task)           tasks_dir="$2"; shift 2 ;;
            --results-dir)    results_dir="$2"; shift 2 ;;
            --gen-max-tokens) eval_context_len="$2"; shift 2 ;;
            --temperature)    temperature="$2"; shift 2 ;;
            --top-p)          top_p="$2"; shift 2 ;;
            *)                echo "Unknown parameter: $1"; return 1 ;;
        esac
    done

    _install_lm_eval_deps
    _patch_lm_eval

    local openai_server_base="http://0.0.0.0:${port}"
    local openai_chat_base="${openai_server_base}/v1/chat/completions"
    export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
    MODEL_NAME=${MODEL_NAME:-$MODEL}

    # Cap output tokens to fit within context window minus typical input.
    local max_output_tokens=$(( eval_context_len > 4096 ? eval_context_len - 4096 : eval_context_len / 2 ))
    if [ "$max_output_tokens" -gt 16384 ]; then
        max_output_tokens=16384
    fi
    echo "Eval budget: eval_context_len=${eval_context_len}, max_output_tokens=${max_output_tokens}"

    export EVAL_RESULT_DIR="$results_dir"
    set -x
    python3 -m lm_eval --model local-chat-completions --apply_chat_template \
      --tasks "${tasks_dir}" \
      --output_path "${results_dir}" \
      --log_samples \
      --model_args "model=${MODEL_NAME},base_url=${openai_chat_base},api_key=${OPENAI_API_KEY},eos_string=</s>,max_retries=5,num_concurrent=${concurrent_requests},timeout=1800,tokenized_requests=False,max_length=${eval_context_len}" \
      --gen_kwargs "max_tokens=${max_output_tokens},temperature=${temperature},top_p=${top_p}"
    local eval_exit=$?
    set +x
    return $eval_exit
}


# ------------------------------
# Eval artifact collection
# ------------------------------

append_lm_eval_summary() {
    local results_dir="${EVAL_RESULT_DIR}"
    if [ -z "${results_dir}" ]; then
        echo "WARN: EVAL_RESULT_DIR is empty; skipping artifact collection" >&2
        return 1
    fi
    local out_dir="${results_dir}"
    if [ ! -d "${out_dir}" ]; then
        echo "WARN: EVAL_RESULT_DIR='${out_dir}' does not exist; skipping artifact collection" >&2
        return 1
    fi

    # Write meta_env.json so downstream collectors know which sweep dimension
    # the eval was for (framework, precision, topology, conc, model, etc.).
    local meta_json="${out_dir}/meta_env.json"
    local model_name="${MODEL_NAME:-$MODEL}"
    # sglang's GB200 nightly is exclusively multi-node.
    local is_multinode_json="true"
    local prefill_tp="${PREFILL_TP:-${TP:-1}}"
    local prefill_ep="${PREFILL_EP:-${EP_SIZE:-1}}"
    local prefill_num_workers="${PREFILL_NUM_WORKERS:-1}"
    local decode_tp="${DECODE_TP:-${TP:-1}}"
    local decode_ep="${DECODE_EP:-${EP_SIZE:-1}}"
    local decode_num_workers="${DECODE_NUM_WORKERS:-1}"

    local dp_json="false"
    if [ "${DP_ATTENTION:-false}" = "true" ]; then dp_json="true"; fi
    local prefill_dp_json="$dp_json"
    if [ "${PREFILL_DP_ATTENTION:-${DP_ATTENTION:-false}}" = "true" ]; then
        prefill_dp_json="true"
    else
        prefill_dp_json="false"
    fi
    local decode_dp_json="$dp_json"
    if [ "${DECODE_DP_ATTENTION:-${DP_ATTENTION:-false}}" = "true" ]; then
        decode_dp_json="true"
    else
        decode_dp_json="false"
    fi

    cat > "${meta_json}" <<META
{
  "is_multinode": ${is_multinode_json},
  "framework": "${FRAMEWORK:-dynamo-sglang}",
  "precision": "${PRECISION:-unknown}",
  "spec_decoding": "${SPEC_DECODING:-none}",
  "tp": ${TP:-1},
  "conc": ${CONC:-1},
  "ep": ${EP_SIZE:-1},
  "dp_attention": ${dp_json},
  "prefill_tp": ${prefill_tp},
  "prefill_ep": ${prefill_ep},
  "prefill_dp_attention": ${prefill_dp_json},
  "prefill_num_workers": ${prefill_num_workers},
  "decode_tp": ${decode_tp},
  "decode_ep": ${decode_ep},
  "decode_dp_attention": ${decode_dp_json},
  "decode_num_workers": ${decode_num_workers},
  "model": "${model_name:-}",
  "model_prefix": "${MODEL_PREFIX:-unknown}",
  "hw": "${RUNNER_TYPE:-gb200}",
  "isl": "${ISL:-0}",
  "osl": "${OSL:-0}"
}
META

    # Move eval artifacts into PWD so srtctl's bench.sh can copy them to /logs.
    if [ -f "${meta_json}" ]; then
        mv -f "${meta_json}" ./ || echo "WARN: failed to move ${meta_json}" >&2
    fi
    if [ -d "${out_dir}" ]; then
        while IFS= read -r -d '' jf; do
            base=$(basename "$jf")
            if [ "$base" != "meta_env.json" ]; then
                mv -f "$jf" ./ || echo "WARN: failed to move ${jf}" >&2
            fi
        done < <(find "${out_dir}" -type f -name "*.json*" -print0 2>/dev/null)
    fi

    # Best-effort cleanup of the temp output dir.
    if [ -n "${out_dir}" ] && [ -d "${out_dir}" ]; then
        rm -rf --one-file-system "${out_dir}" || rm -rf "${out_dir}" || true
    fi

    echo "Moved eval artifacts to: $(pwd)"
}


# ------------------------------
# Unified eval entrypoint (called from srtctl's lm-eval bench.sh)
# ------------------------------

run_eval() {
    local framework="${EVAL_FRAMEWORK:-lm-eval}"
    local forwarded=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --framework) framework="$2"; shift 2 ;;
            *)           forwarded+=("$1"); shift ;;
        esac
    done

    if [ -z "${EVAL_MAX_MODEL_LEN:-}" ]; then
        compute_eval_context_length "$MODEL" "${MAX_MODEL_LEN:-0}" > /dev/null
    fi

    local eval_rc=0
    case "$framework" in
        lm-eval|lm_eval) run_lm_eval "${forwarded[@]}" || eval_rc=$? ;;
        *)               echo "Unknown framework '${framework}'"; eval_rc=1 ;;
    esac

    if [ "$eval_rc" -ne 0 ]; then
        echo "ERROR: run_eval failed with exit code $eval_rc" >&2
        if [ "${EVAL_ONLY:-false}" = "true" ]; then
            echo "Eval-only mode: failing after artifact collection" >&2
            return "$eval_rc"
        fi
    fi
    return $eval_rc
}

#!/usr/bin/env bash
# Loop 2 Phase 0 preflight: verify the running node matches the assumptions
# baked into serve_double_sparsity.sh BEFORE serving begins.
#
# Exit code 0 = safe to proceed.
# Non-zero  = a Phase 0 invariant failed; do not start the server.
#
# Invariants checked (any mismatch is a non-zero exit):
#   1. --backend must be `flashmla_kv` (NSA decode backend).
#   2. --dtype must be `fp8_e4m3` (KV-cache dtype).
#   3. --page-size must be 64.
#   4. --top-k must be 2048 (DS selector max_top_k).
#   5. --tp-size must equal the value of $TP_SIZE / available GPUs.
#   6. --cuda-arch-major (queried via nvidia-smi or override) must be 9 (H200 / H100 class).
#
# Each invariant is checked via a flag value passed in by the operator OR by
# probing the live environment when --check-only is set.

set -euo pipefail

BACKEND=""
DTYPE=""
PAGE_SIZE=""
TOP_K=""
TP_SIZE=""
CUDA_ARCH_MAJOR=""
CHECK_ONLY=0

usage() {
    cat <<'USAGE'
Usage: preflight.sh [OPTIONS]

Required invariants (each can be passed explicitly OR probed via --check-only):
  --backend <name>            Expected attention backend (must be flashmla_kv).
  --dtype <name>              Expected KV-cache dtype (must be fp8_e4m3).
  --page-size <int>           Expected page size (must be 64).
  --top-k <int>               Expected DS selector top_k (must be 2048).
  --tp-size <int>             Expected TP world size (must equal available GPUs).
  --cuda-arch-major <int>     Expected CUDA arch major (must be 9 / Hopper class).

Modes:
  --check-only                Probe the live environment for any value not passed
                              explicitly. Without this flag, missing values are
                              treated as preflight failures.

Exits 0 on success, non-zero on the first failed invariant.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            BACKEND="$2";          shift 2 ;;
        --dtype)              DTYPE="$2";            shift 2 ;;
        --page-size)          PAGE_SIZE="$2";        shift 2 ;;
        --top-k)              TOP_K="$2";            shift 2 ;;
        --tp-size)            TP_SIZE="$2";          shift 2 ;;
        --cuda-arch-major)    CUDA_ARCH_MAJOR="$2";  shift 2 ;;
        --check-only)         CHECK_ONLY=1;          shift   ;;
        -h|--help)            usage; exit 0 ;;
        *)
            echo "preflight: unknown argument: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

# Helper: fail with a numbered exit code so each invariant is distinguishable.
fail() {
    local code="$1"; shift
    echo "preflight FAIL [$code]: $*" >&2
    exit "$code"
}

probe_cuda_arch_major() {
    # nvidia-smi --query-gpu=compute_cap returns e.g. "9.0" on H200.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local cap
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1)"
        if [[ "$cap" =~ ^([0-9]+)\.[0-9]+$ ]]; then
            echo "${BASH_REMATCH[1]}"
            return
        fi
    fi
    echo ""
}

probe_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --list-gpus 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

# Fill in probed values if --check-only is set. Backend / dtype / page-size /
# top-k cannot be probed from the running system; they come from the launcher's
# emitted environment (DS_PREFLIGHT_*). When --check-only is set and a value
# is missing from BOTH the explicit flag AND the env, we fall back to the
# launcher defaults baked into serve_double_sparsity.sh so the runbook's
# `bash preflight.sh --check-only` invocation can succeed on a matching node.
if [[ "$CHECK_ONLY" -eq 1 ]]; then
    : "${BACKEND:=${DS_PREFLIGHT_BACKEND:-flashmla_kv}}"
    : "${DTYPE:=${DS_PREFLIGHT_DTYPE:-fp8_e4m3}}"
    : "${PAGE_SIZE:=${DS_PREFLIGHT_PAGE_SIZE:-64}}"
    : "${TOP_K:=${DS_PREFLIGHT_TOP_K:-2048}}"
    : "${TP_SIZE:=${DS_PREFLIGHT_TP_SIZE:-$(probe_gpu_count)}}"
    : "${CUDA_ARCH_MAJOR:=${DS_PREFLIGHT_CUDA_ARCH_MAJOR:-$(probe_cuda_arch_major)}}"
fi

# 1. Backend
[[ -z "$BACKEND" ]]             && fail 1 "--backend not provided and --check-only cannot infer it"
[[ "$BACKEND" != "flashmla_kv" ]] && fail 1 "expected backend=flashmla_kv, got '$BACKEND'"

# 2. KV-cache dtype
[[ -z "$DTYPE" ]]               && fail 2 "--dtype not provided and --check-only cannot infer it"
[[ "$DTYPE" != "fp8_e4m3" ]]    && fail 2 "expected dtype=fp8_e4m3, got '$DTYPE'"

# 3. Page size
[[ -z "$PAGE_SIZE" ]]           && fail 3 "--page-size not provided and --check-only cannot infer it"
[[ "$PAGE_SIZE" != "64" ]]      && fail 3 "expected page_size=64, got '$PAGE_SIZE'"

# 4. Top-K
[[ -z "$TOP_K" ]]               && fail 4 "--top-k not provided and --check-only cannot infer it"
[[ "$TOP_K" != "2048" ]]        && fail 4 "expected top_k=2048, got '$TOP_K'"

# 5. TP size
[[ -z "$TP_SIZE" ]]             && fail 5 "--tp-size not provided and --check-only cannot infer it"
[[ "$TP_SIZE" != "8" ]]         && fail 5 "expected tp_size=8, got '$TP_SIZE'"

# 6. CUDA arch major
[[ -z "$CUDA_ARCH_MAJOR" ]]     && fail 6 "--cuda-arch-major not provided and --check-only cannot infer it"
[[ "$CUDA_ARCH_MAJOR" != "9" ]] && fail 6 "expected cuda_arch_major=9 (Hopper / H200 class), got '$CUDA_ARCH_MAJOR'"

echo "preflight OK: backend=$BACKEND dtype=$DTYPE page_size=$PAGE_SIZE top_k=$TOP_K tp_size=$TP_SIZE cuda_arch_major=$CUDA_ARCH_MAJOR"
exit 0

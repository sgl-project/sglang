#!/usr/bin/env bash
# Reference SGLang server invocation for the native_nsa baseline.
#
# Targets DeepSeek-V3.2 (FP8) on a single H200 node, 8-way TP, page=64,
# matching the operating point in DEC-1 (h200-10-220-51-6 or
# h200-10-220-51-8). Pair with development/benchmark_baseline.sh to
# populate the native_nsa column of the two-column comparison.
#
# Double Sparsity is intentionally NOT enabled here — the baseline runs
# unmodified NSA selection so the comparison row pair is honest.
#
# Locked Option B operating point (plan §13 / DEC-1) — same locked flags as
# development/serve_double_sparsity.sh, so the AC-8 / AC-9 / AC-11
# comparison only differs by DS enablement and the AC-10 radix-cache gate:
#   --kv-cache-dtype fp8_e4m3
#   --dsa-prefill-backend flashmla_kv
#   --dsa-decode-backend  flashmla_kv
#   --disable-overlap-schedule
#   --disable-piecewise-cuda-graph
#   --page-size 64
# NOTE: by default this script does NOT pass --disable-radix-cache. Per
# plan §13 the loop4-compatible AC-11 sweep runs the DSA baseline with radix
# cache ON so any DS TPS gap vs DSA reflects the DS configuration alone, not
# the radix gate that DS still has to clear (AC-10).
#
# The TIER-1 smoke is different: DS still launches with --disable-radix-cache
# (the AC-10 gate has not been flipped yet) and the two-column comparator
# (development/benchmark_compare.py) refuses a radix-cache mismatch between
# the columns. Set DISABLE_RADIX_CACHE=1 to launch this baseline radix-off so
# the smoke compares apples-to-apples. Leave it unset (default ON) for the
# radix-on AC-11 sweep once DS has cleared the AC-10 flip.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/cluster-storage/models/deepseek-ai/DeepSeek-V3.2}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-64}"
# DeepSeek-V3.2 FP8 sharded TP=8 is ~84 GB of weights per rank. The stock
# default mem-fraction (0.897) reserves the rest for the KV pool and leaves
# almost no physical headroom for the flashmla attention kernel's workspace
# (~1 GB) at the 4096-ISL benchmark shape, so the baseline OOMs the moment
# bench_serving drives real traffic. 0.85 leaves ~20 GB/rank of runtime
# headroom while keeping a large KV pool. (The DS launcher uses 0.6 because
# it additionally reserves a per-rank TokenLabelTable; the baseline does not.)
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
LOG_DIR="${LOG_DIR:-$(pwd)/development/logs}"

# Radix-off parity knob for the TIER-1 smoke (see header NOTE). Default 0
# keeps radix cache ON for the AC-11 sweep; set DISABLE_RADIX_CACHE=1 to add
# --disable-radix-cache so the smoke matches the DS launcher.
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
RADIX_CACHE_ARG=""
if [[ "${DISABLE_RADIX_CACHE}" == "1" ]]; then
  RADIX_CACHE_ARG="--disable-radix-cache"
fi

mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/server_native_nsa_$(date +%Y%m%d-%H%M%S).log"
echo ">>> Starting native_nsa baseline server"
echo "    model        = ${MODEL_PATH}"
echo "    port         = ${PORT}"
echo "    tp_size      = ${TP_SIZE}"
echo "    kv_cache     = ${KV_CACHE_DTYPE}"
echo "    page_size    = ${PAGE_SIZE}"
echo "    mem_fraction = ${MEM_FRACTION_STATIC}"
echo "    radix_cache  = $([[ -n "${RADIX_CACHE_ARG}" ]] && echo "disabled (smoke parity)" || echo "enabled (AC-11 default)")"
echo "    log          = ${LOG_FILE}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --page-size "${PAGE_SIZE}" \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  ${RADIX_CACHE_ARG} \
  --trust-remote-code \
  2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
# Reference SGLang server invocation for the standalone Double Sparsity path.
#
# Mirrors development/serve_native_nsa.sh but adds --enable-double-sparsity
# and --double-sparsity-config. Targets DeepSeek-V3.2 (FP8) on a single H200
# node, 8-way TP, page=64. Double Sparsity and HiSparse are mutually
# exclusive: enabling both at startup is rejected by the launch validator.
#
# Locked operating point — these dense-prefill / sparse-decode flags MUST
# agree across this script and development/serve_native_nsa.sh so the
# DS-vs-baseline comparison differs only by Double Sparsity enablement and
# the radix-cache gate:
#   --kv-cache-dtype fp8_e4m3
#   --dsa-prefill-backend flashmla_kv
#   --dsa-decode-backend  flashmla_kv
#   --disable-overlap-schedule
#   --disable-piecewise-cuda-graph
#   --page-size 64
#
# DEV-ONLY launcher. The DS startup validator refuses --enable-double-sparsity
# The page-table adapter is now in place; production deployments boot
# without any DS dev-override environment variable. The previous
# adapter-bypass and placeholder-selector overrides were removed;
# production must call bind_runtime_data with a real channel mask +
# page signature table before serving.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/cluster-storage/models/deepseek-ai/DeepSeek-V3.2}"
PORT="${PORT:-30000}"
# Bind address. Default 127.0.0.1 (localhost-only, the sglang default). Set
# HOST=0.0.0.0 to make this DS server reachable from another node (symmetry
# with serve_native_nsa.sh; used when a paired DS-vs-baseline quality run
# spans two nodes). The dense-prefill/sparse-decode flags below are unchanged
# by this knob.
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CHANNEL_MASK_PATH="${CHANNEL_MASK_PATH:-/models/dsv32-fp8-channel-mask.safetensors}"
TOP_K="${TOP_K:-2048}"
DEVICE_BUFFER_SIZE="${DEVICE_BUFFER_SIZE:-4096}"
# Per-slot label storage precision. "fp16" (default) is full precision; "int8"
# selects the compact path (int8 signatures + per-vector fp16 scales, ~0.5625x
# bytes). The compact table only boots when this is set to int8 — running the
# script as-is validates the fp16 table.
SIGNATURE_DTYPE="${SIGNATURE_DTYPE:-fp16}"
# Double Sparsity allocates a per-rank TokenLabelTable (sized from the KV pool,
# tens of GiB on V3.2) AFTER the static weight+KV pool is reserved, and also
# needs headroom for the regular CUDA-graph capture set plus per-request decode
# activations. The stock default (0.897) OOMs at boot; 0.7 boots but OOMs during
# generation on V3.2/H200. 0.6 boots and serves stably (verified): weights are
# ~80 GB/rank, leaving a small KV pool plus ~38 GB of runtime headroom.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}"
# Fixed server seed so a paired DS-vs-baseline SLO comparison is seed-matched
# (the only intended column differences are DS enablement/config + DS mem fraction).
RANDOM_SEED="${RANDOM_SEED:-20260607}"

# Loop-7 measurement op-point. The stock defaults (fp16 / mem 0.6) are the safe
# fp16-table boot point, NOT the Loop-7 baseline/oracle/M1 operating point
# (int8 compact table / mem 0.7). Set LOOP7_MEASUREMENT=1 to PIN the Loop-7
# op-point so a measurement run cannot silently reproduce the old regime.
if [[ "${LOOP7_MEASUREMENT:-0}" == "1" ]]; then
  SIGNATURE_DTYPE="int8"
  MEM_FRACTION_STATIC="0.7"
  echo ">>> LOOP7_MEASUREMENT=1: pinned SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.7"
fi

LOG_DIR="${LOG_DIR:-$(pwd)/development/logs}"
mkdir -p "${LOG_DIR}"

# Flag-gated non-learned selector variants (config-borne so they reach the TP
# workers). Defaults reproduce the production scorer (byte-identical).
SCORER_NORM="${SCORER_NORM:-off}"                          # off | cosine | hybrid
SCORER_NORM_HYBRID_THRESHOLD="${SCORER_NORM_HYBRID_THRESHOLD:-8192}"
HEAD_AGG="${HEAD_AGG:-max}"                                # max | mean
ANCHOR_MODE="${ANCHOR_MODE:-off}"                          # off | recency | global | strided
ANCHOR_BUDGET="${ANCHOR_BUDGET:-0}"
# Fail-closed NIAH recall-oracle diagnostic (config-borne so it reaches TP
# workers). RECALL_ORACLE=1 emits "recall_oracle": true and forces eager
# (--disable-cuda-graph), which the validator requires.
RECALL_ORACLE="${RECALL_ORACLE:-0}"                        # 0 | 1
if [[ "${RECALL_ORACLE}" == "1" ]]; then RECALL_ORACLE_JSON=true; else RECALL_ORACLE_JSON=false; fi
# Opt-in lifted-budget decode (graph-safe production path). LIFTED_BUDGET=1 emits
# "enable_lifted_budget_decode": true + a wider "lifted_budget_top_k" (must be
# > top_k and a multiple of 128). The path runs UNDER CUDA graph (alloc-free
# dequantize_k_cache_paged_out + fixed-shape compact builder + preallocated
# DSGraphState scratch), so it no longer forces --disable-cuda-graph. Off => the
# config emits the default-off pair (lifted_budget_top_k must be 0 when off, else
# the config fails closed).
LIFTED_BUDGET="${LIFTED_BUDGET:-0}"                        # 0 | 1
if [[ "${LIFTED_BUDGET}" == "1" ]]; then
  LIFTED_BUDGET_JSON=true
  LIFTED_BUDGET_TOP_K="${LIFTED_BUDGET_TOP_K:-4096}"
else
  LIFTED_BUDGET_JSON=false
  LIFTED_BUDGET_TOP_K=0
fi
DS_CONFIG=$(printf '{"top_k": %s, "page_size": %s, "channel_mask_path": "%s", "device_buffer_size": %s, "signature_dtype": "%s", "scorer_norm": "%s", "scorer_norm_hybrid_threshold": %s, "head_agg": "%s", "anchor_mode": "%s", "anchor_budget": %s, "recall_oracle": %s, "enable_lifted_budget_decode": %s, "lifted_budget_top_k": %s}' \
  "${TOP_K}" "${PAGE_SIZE}" "${CHANNEL_MASK_PATH}" "${DEVICE_BUFFER_SIZE}" "${SIGNATURE_DTYPE}" "${SCORER_NORM}" "${SCORER_NORM_HYBRID_THRESHOLD}" "${HEAD_AGG}" "${ANCHOR_MODE}" "${ANCHOR_BUDGET}" "${RECALL_ORACLE_JSON}" "${LIFTED_BUDGET_JSON}" "${LIFTED_BUDGET_TOP_K}")
echo ">>> effective double_sparsity_config = ${DS_CONFIG}"

# Eager is required only for the recall-oracle diagnostic (it host-syncs). As of R9
# all non-learned scorer/anchor variants are graph-safe, and the lifted-budget
# decode path is now graph-safe too (fixed-shape compact builder + alloc-free out=
# dequant + preallocated scratch), so it runs UNDER CUDA graph. Auto-add
# --disable-cuda-graph only for RECALL_ORACLE.
CUDA_GRAPH_ARGS=()
if [[ "${RECALL_ORACLE}" == "1" ]]; then
  CUDA_GRAPH_ARGS=(--disable-cuda-graph)
  echo ">>> recall_oracle diagnostic requires eager: adding --disable-cuda-graph"
fi

# Radix cache: DS serves radix-off by default. To serve radix-on, set
# RADIX_FIXTURE_ARTIFACT to a fixtures-passed state file (written by
# validator.write_radix_fixture_state once BOTH the label-capture and FP8
# scale-stability fixtures pass on this exact config). The launcher then passes
# --double-sparsity-radix-fixture-artifact and drops --disable-radix-cache; the
# validator re-verifies the state matches this config before permitting radix-on.
# SGLANG_DS_RADIX_OVERRIDE=1 is a developer-only escape hatch that enables
# radix-on WITHOUT the artifact (used solely to run the radix fixtures, which
# need radix reuse) — it is not the production serving mechanism.
RADIX_FIXTURE_ARTIFACT="${RADIX_FIXTURE_ARTIFACT:-}"
if [[ -n "${RADIX_FIXTURE_ARTIFACT}" ]]; then
  RADIX_ARGS=(--double-sparsity-radix-fixture-artifact "${RADIX_FIXTURE_ARTIFACT}")
elif [[ "${SGLANG_DS_RADIX_OVERRIDE:-}" == "1" ]]; then
  RADIX_ARGS=()  # developer-only radix-on for fixture runs (see note above)
else
  RADIX_ARGS=(--disable-radix-cache)
fi

LOG_FILE="${LOG_DIR}/server_double_sparsity_$(date +%Y%m%d-%H%M%S).log"
echo ">>> Starting Double Sparsity server (standalone)"
echo "    model            = ${MODEL_PATH}"
echo "    host             = ${HOST}"
echo "    port             = ${PORT}"
echo "    tp_size          = ${TP_SIZE}"
echo "    kv_cache         = ${KV_CACHE_DTYPE}"
echo "    page_size        = ${PAGE_SIZE}"
echo "    channel_mask     = ${CHANNEL_MASK_PATH}"
echo "    top_k            = ${TOP_K}"
echo "    device_buffer    = ${DEVICE_BUFFER_SIZE}"
echo "    signature_dtype  = ${SIGNATURE_DTYPE}"
echo "    recall_oracle    = ${RECALL_ORACLE} (eager: ${#CUDA_GRAPH_ARGS[@]} cuda-graph arg(s))"
echo "    radix_cache      = $([[ -n "${RADIX_FIXTURE_ARTIFACT}" ]] && echo "ENABLED (fixture artifact: ${RADIX_FIXTURE_ARTIFACT})" || echo "disabled (no fixture artifact)")"
echo "    log              = ${LOG_FILE}"

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --page-size "${PAGE_SIZE}" \
  --dsa-prefill-backend flashmla_kv \
  --dsa-decode-backend flashmla_kv \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  --enable-double-sparsity \
  --double-sparsity-config "${DS_CONFIG}" \
  --random-seed "${RANDOM_SEED}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  `# Radix cache: --disable-radix-cache (default radix-off) or the` \
  `# fixtures-passed artifact path (radix-on), selected above.` \
  "${RADIX_ARGS[@]}" \
  --trust-remote-code \
  ${EXTRA_SERVER_ARGS:-} \
  2>&1 | tee "${LOG_FILE}"
# The DS validator gates radix cache OFF until the page-stability fixtures have
# been recorded as passing for this configuration. To enable radix-on:
#   1. Run test/manual/test_dsv32_radix_label_capture_fixture.py (boot with
#      SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 so the per-request snapshot attaches)
#      — cold-vs-warm DS label SHA bit-equality via response meta_info.
#   2. Run test/manual/test_dsv32_fp8_scale_stability.py
#      (SGLANG_DS_FP8_SCALE_PROOF=1) — singleton vs packed-page FP8 scale-byte
#      equality via the production fused_store_index_k_cache kernel.
# On BOTH passing, write a fixtures-passed state file with
# validator.write_radix_fixture_state(...) and point RADIX_FIXTURE_ARTIFACT at
# it; the validator re-verifies the state against this config and permits
# radix-on. The continuation-only smoke at
# test/manual/test_dsv32_radix_cache_fixture.py is a pre-flight check, not the
# fixture evidence.

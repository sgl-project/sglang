#!/usr/bin/env bash
# Launch a 2-node 1P1D disaggregation benchmark on the AMD MI355X `amd-sglang`
# Slurm cluster, then emit per-concurrency result JSONs that
# scripts/ci/slurm/process_result.py aggregates.
#
# salloc's (prefill_workers + decode_workers) nodes -- one server per node --
# and runs the Docker harness: prefill server(s) on the first nodes, decode
# server(s) on the rest, a standalone load balancer on the prefill node, then an
# sglang.bench_serving concurrency sweep over MORI. Default recipe is 1P1D (2
# nodes); see the drive.sh note on reserving 2P2D / 1P3D / 3P1D.
#
# Required environment variables (set by the GitHub Actions workflow):
#   MODEL              - HuggingFace model id (table label / served model)
#   MODEL_PREFIX       - short prefix, e.g. dsv4flash
#   PRECISION          - fp8 / fp4
#   ISL, OSL           - input / output sequence lengths for the sweep
#   CONFIG_FILE        - path to the recipe YAML (relative to repo root)
#   RESULT_FILENAME    - prefix for the emitted result JSONs
#   MATRIX_CONFIG_NAME - matrix entry name (used in filenames/tags)
#   GITHUB_WORKSPACE   - set by GitHub Actions; where result JSONs are written
# Optional:
#   MODEL_PATH         - local snapshot dir (preferred over downloading MODEL)
#   SLURM_PARTITION    - default: amd-sglang
#   SLURM_NODELIST     - optional explicit node pin (else scheduler chooses)
#   SLURM_EXCLUDE      - optional comma-separated nodes to keep the scheduler
#                        off (e.g. hosts with a broken RDMA driver)
#   SGLANG_USE_CHECKOUT_RUNTIME
#                      - default 1. Reinstall this workflow checkout's Python
#                        sglang package inside each runtime container, and the
#                        checkout sglang-router package inside the bench
#                        container, before launching servers/bench. Set 0 to
#                        use the image's baked-in packages.
#   RUNNER_NAME        - GitHub runner name (a built-in default env var)
#   GITHUB_RUN_ID      - GitHub Actions run id (a built-in default env var)
#                        The allocation is named
#                          mi355x-ci-<RUNNER_NAME>-<GITHUB_RUN_ID>-<config>
#                        so workflow cleanup can scancel exactly this leg's job
#                        (full name) or this runner's stale jobs (RUNNER_NAME
#                        prefix) -- never a blanket `squeue --me`. The run id +
#                        config make the name unique per matrix leg even if two
#                        runners happen to share a name.
#   SLURM_EXCLUSIVE    - request whole nodes (default 1); set 0 to disable
#   TIME_LIMIT         - salloc time limit, default 02:30:00 (covers server
#                        load + perf sweep + full GSM8K, under the 180m step cap)

set -euo pipefail
set -x

: "${MODEL_PREFIX:?}"
: "${PRECISION:?}"
: "${ISL:?}"
: "${OSL:?}"
: "${CONFIG_FILE:?}"
: "${RESULT_FILENAME:?}"
: "${MATRIX_CONFIG_NAME:?}"
: "${GITHUB_WORKSPACE:?}"

SLURM_PARTITION="${SLURM_PARTITION:-amd-sglang}"
TIME_LIMIT="${TIME_LIMIT:-02:30:00}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"
SGLANG_USE_CHECKOUT_RUNTIME="${SGLANG_USE_CHECKOUT_RUNTIME:-1}"
case "${SGLANG_USE_CHECKOUT_RUNTIME,,}" in
    0|false|no|off) SGLANG_USE_CHECKOUT_RUNTIME=0 ;;
    *) SGLANG_USE_CHECKOUT_RUNTIME=1 ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: set MODEL_PATH (local snapshot) or MODEL" >&2
    exit 1
fi

# Resolve a HuggingFace cache dir (models--org--name) to its live snapshot dir.
# Lets nightly-configs / recipes point at the shared cache without hardcoding a
# snapshot hash; a concrete snapshot dir (or plain dir) is returned unchanged.
# Used for both MODEL_PATH and an optional speculative draft model path.
resolve_snapshot() {
    local p="$1"
    if [[ -f "$p/refs/main" && -d "$p/snapshots" ]]; then
        local hash resolved
        hash="$(cat "$p/refs/main")"
        resolved="$p/snapshots/$hash"
        if [[ -n "$hash" && -d "$resolved" ]]; then
            echo "resolved snapshot: $p -> $resolved" >&2
            echo "$resolved"
            return 0
        fi
        echo "ERROR: refs/main=$hash but $resolved missing" >&2
        return 1
    fi
    echo "$p"
}
MODEL_PATH="$(resolve_snapshot "$MODEL_PATH")" || exit 1

# ---------------------------------------------------------------------------
# Parse the recipe (runtime + bench + topology) into shell vars.
# ---------------------------------------------------------------------------
# Ensure PyYAML is available to the host python used for parsing.
python3 -c 'import yaml' 2>/dev/null || pip install pyyaml -q 2>/dev/null \
    || pip install --user pyyaml -q 2>/dev/null || true

# Emit KEY=value lines and eval them (robust single-level command substitution;
# avoids a nested read<<EOF/$(<<PY) heredoc that misparses on some shells).
RECIPE_VARS="$(python3 - "$CONFIG_FILE" <<'PY'
import sys, yaml, shlex
r = yaml.safe_load(open(sys.argv[1]))
rt = r["runtime"]; b = r["backend"]["sglang_config"]; bn = r["bench"]
res = r.get("resources", {})
def emit(k, v): print(f"{k}={v}")
# Quoted emit for values that may contain spaces (per-role wide-EP flag/env
# strings). eval treats the RHS as a single shell word. Existing emits stay
# space-free so the EP<=8 recipes are byte-identical.
def emitq(k, v): print(f"{k}={shlex.quote(str(v))}")
emit("IMAGE", rt["image"])
# Attention backend: single (`attention_backend`) or split
# (`prefill_attention_backend`/`decode_attention_backend`). Empty when absent so
# the flag is dropped for a model that omits it.
emit("ATTN", rt.get("attention_backend", ""))
emit("PATTN", rt.get("prefill_attention_backend", ""))
emit("DATTN", rt.get("decode_attention_backend", ""))
emit("IB", rt["ib_devices"])
# Wide-EP (EP > GPUs/node) options; empty for the single-node EP<=8 recipes so
# the flags/env below are dropped and their argv stays byte-identical.
#   moe_a2a_backend    -> --moe-a2a-backend <x> --deepep-mode normal (MoE all-to-all)
#   dist_socket_ifname -> NCCL_/GLOO_SOCKET_IFNAME for cross-node torch-dist init
emit("A2A", rt.get("moe_a2a_backend", ""))
emit("DIST_SOCK", rt.get("dist_socket_ifname", ""))
# KV transfer backend for the P->D handoff. Defaults to mori (the pre-wide
# hardcoded value) so EP<=8 recipes stay byte-identical; wide-EP spur recipes
# set mooncake, which is the validated cross-node KV path there.
emit("XFER", rt.get("kv_transfer_backend", "mori"))
# SGLANG_USE_ROCM700A toggles the ROCm-7.0.0-alpha codepath. Default 1 (the
# pre-wide hardcoded value) keeps EP<=8 recipes byte-identical; the validated
# wide-EP run on the rocm720 0715 image needs 0, set via runtime.rocm700a.
emit("ROCM700A", rt.get("rocm700a", 1))
emit("PPORT", rt["prefill_port"])
emit("DPORT", rt["decode_port"])
emit("PBOOT", rt["prefill_bootstrap_port"])
emit("DBOOT", rt["decode_bootstrap_port"])
emit("LBPORT", rt["lb_port"])
emit("MEMFRAC", rt["mem_fraction_static"])
emit("PAGE", rt["page_size"])
emit("MAXREQ", rt["max_running_requests"])
emit("MAXTOK", rt.get("max_total_tokens", ""))
emit("CHUNK", rt["chunked_prefill_size"])
# swa is DSV4-specific; emit empty when a model omits it so the flag is dropped.
emit("SWA", rt.get("swa_full_tokens_ratio", ""))
# 1 when the recipe carries a `model:` block (env + server_args written to
# model_flags.sh); 0 for the DSV4 recipes, which keep the hardcoded DSV4 path.
emit("HAS_MODEL", 1 if r.get("model") else 0)
emit("PTP", b["prefill"]["tensor-parallel-size"])
emit("DTP", b["decode"]["tensor-parallel-size"])
emit("PEP", b["prefill"].get("expert-parallel-size", 1))
emit("PDP", b["prefill"].get("data-parallel-size", 1))
# Decode-side EP/DP: default to the prefill values so a recipe that omits them
# (every EP<=8 recipe today, where both roles are identical) is byte-identical.
# Oren's wide-EP recipes set decode EP/DP=16 while prefill stays EP8.
emit("DEP", b["decode"].get("expert-parallel-size", b["prefill"].get("expert-parallel-size", 1)))
emit("DDP", b["decode"].get("data-parallel-size", b["prefill"].get("data-parallel-size", 1)))
# Wide-EP per-role overrides (optional `runtime.wide_ep` block). Absent for
# EP<=8 recipes -> all defaults collapse to the existing single-value knobs, so
# the generated prefill/decode argv is unchanged. Present only in the asymmetric
# narrow-prefill/wide-decode (Oren) EP16 recipes.
we = rt.get("wide_ep", {}) or {}
emit("KVDTYPE", we.get("kv_cache_dtype", ""))
emit("PMEMFRAC", we.get("prefill_mem_fraction_static", rt["mem_fraction_static"]))
emit("DMEMFRAC", we.get("decode_mem_fraction_static", rt["mem_fraction_static"]))
emit("PCHUNK", we.get("prefill_chunked_prefill_size", rt["chunked_prefill_size"]))
emit("PMAXREQ", we.get("prefill_max_running_requests", rt["max_running_requests"]))
emit("DMAXREQ", we.get("decode_max_running_requests", rt["max_running_requests"]))
emitq("PEXTRA", we.get("prefill_extra_flags", ""))
emitq("DEXTRA", we.get("decode_extra_flags", ""))
emitq("WECOMMON", we.get("common_extra_flags", ""))
def render_env(d):
    return " ".join(f"-e {k}={v}" for k, v in (d or {}).items())
emitq("PENV", render_env(we.get("prefill_extra_env")))
emitq("DENV", render_env(we.get("decode_extra_env")))
m = r.get("mtp", {}) or {}
emit("MTP_ENABLED", 1 if m.get("enabled") else 0)
emit("MTP_ALGO", m.get("algorithm", "EAGLE"))
emit("MTP_STEPS", m.get("num_steps", 3))
emit("MTP_TOPK", m.get("eagle_topk", 1))
emit("MTP_DRAFT", m.get("num_draft_tokens", 4))
# External draft checkpoint (EAGLE3 etc.); empty for DSV4's built-in EAGLE head.
emit("MTP_DRAFT_PATH", m.get("draft_model_path", ""))
# Worker counts double as node counts here: one server per node (TP == GPUs/node).
# 1P1D today; bumping these reserves 2P2D / 1P3D / 3P1D. Multi-node-per-worker
# (TP > GPUs/node, needs --dist-init-addr/--nnodes/--node-rank) is out of scope.
emit("PW", res.get("prefill_workers", 1))
emit("DW", res.get("decode_workers", 1))
emit("CONCS", ",".join(str(c) for c in bn["concurrencies"]))
emit("NPF", bn["num_prompts_factor"])
emit("RRR", bn["random_range_ratio"])
acc = bn.get("accuracy", {}) or {}
emit("ACC_ENABLED", 1 if acc.get("enabled") else 0)
emit("ACC_SHOTS", acc.get("num_shots", 8))
emit("ACC_NQ", acc.get("num_questions", 1319))
emit("ACC_THR", acc.get("threshold", 0.91))
PY
)"
if [[ -z "$RECIPE_VARS" ]]; then
    echo "ERROR: failed to parse recipe $CONFIG_FILE (empty output from python3/yaml)" >&2
    exit 1
fi
eval "$RECIPE_VARS"
# Optional image override from workflow_dispatch input.
if [[ -n "${IMAGE_OVERRIDE:-}" ]]; then
    IMAGE="$IMAGE_OVERRIDE"
fi

# Nodes per engine: an engine whose TP exceeds one node's GPU count spans
# ceil(TP/GPUS_PER_NODE) nodes and needs torch-dist multi-node init. EP<=8
# recipes give 1 (single node) so all downstream multi-node logic no-ops.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PN_PER=$(( (PTP + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
DN_PER=$(( (DTP + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
# torch-dist rendezvous port for a multi-node engine. Default 29500 (torch's
# conventional MASTER_PORT); avoids :5000, which a node-local daemon holds on
# some clusters (spur). Overridable per environment.
DIST_PORT="${DIST_PORT:-29500}"
# A role may have multiple single-node engines (PW>1, each PN_PER=1: the router
# fans out over them) OR one multi-node wide engine (PN_PER>1, workers==1). What
# is NOT wired is *multiple copies of a multi-node engine* (a wide engine with
# workers>1), because the drive split assumes one contiguous node block per wide
# engine and the router only knows one endpoint per wide engine.
if (( (PN_PER > 1 && PW > 1) || (DN_PER > 1 && DW > 1) )); then
    echo "ERROR: a wide engine (nodes/engine>1) cannot have >1 worker of that role (got PW=$PW PN_PER=$PN_PER DW=$DW DN_PER=$DN_PER)" >&2
    exit 1
fi
echo "recipe: image=$IMAGE attn=${ATTN:-$PATTN/$DATTN} ib=$IB ptp=$PTP dtp=$DTP pn_per=$PN_PER dn_per=$DN_PER a2a=${A2A:-<none>} concs=$CONCS isl=$ISL osl=$OSL"

# ---------------------------------------------------------------------------
# Shared NFS scratch (visible to login node + compute nodes). Raw bench output
# lands here; the launcher normalizes it into GITHUB_WORKSPACE afterwards.
# ---------------------------------------------------------------------------
WORKDIR="$HOME/.mi355x_ci/${MATRIX_CONFIG_NAME}"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage the workflow checkout on shared NFS so Slurm compute-node containers can
# reinstall the same code SHA the workflow checked out. The container gets a
# read-only mount and copies it to /tmp before mutating pyproject.toml.
CHECKOUT_DOCKER_ARGS="-e SGLANG_USE_CHECKOUT_RUNTIME=$SGLANG_USE_CHECKOUT_RUNTIME"
if [[ "$SGLANG_USE_CHECKOUT_RUNTIME" == "1" ]]; then
    CHECKOUT_STAGE="$WORKDIR/checkout"
    CHECKOUT_SHA="$(git -C "$GITHUB_WORKSPACE" rev-parse HEAD)"
    echo "Staging checkout runtime: sha=$CHECKOUT_SHA -> $CHECKOUT_STAGE"
    rm -rf "$CHECKOUT_STAGE"
    mkdir -p "$CHECKOUT_STAGE"
    tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.git/config' \
        -C "$GITHUB_WORKSPACE" -cf - . | tar -C "$CHECKOUT_STAGE" -xf -
    CHECKOUT_DOCKER_ARGS="$CHECKOUT_DOCKER_ARGS -e SGLANG_CHECKOUT_SHA=$CHECKOUT_SHA -v $CHECKOUT_STAGE:/sglang-checkout:ro"
else
    echo "SGLANG_USE_CHECKOUT_RUNTIME=0; using sglang package baked into image."
fi

# Accuracy-gate helpers (written when enabled). Pre-stage the GSM8K test set on
# shared NFS from the login node (which has internet) so the in-container eval
# doesn't depend on compute-node connectivity; fall back to in-container
# download if the pre-fetch fails.
if [[ "$ACC_ENABLED" == "1" ]]; then
    GSM8K_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    curl -fsSL "$GSM8K_URL" -o "$WORKDIR/gsm8k_test.jsonl" 2>/dev/null \
        && echo "gsm8k dataset staged at $WORKDIR/gsm8k_test.jsonl" \
        || echo "WARN: gsm8k pre-stage failed; in-container download will be attempted"
    cat > "$WORKDIR/check_acc.py" <<'PY'
import sys
acc, thr = float(sys.argv[1]), float(sys.argv[2])
print(f"[gsm8k] accuracy={acc:.3f} threshold={thr}")
sys.exit(0 if acc > thr else 1)
PY
fi

# DSV4 load-bearing env (see test/registered/amd/test_deepseek_v4_flash_fp8.py).
# SGLANG_DSV4_FP4_EXPERTS is precision-driven: true for fp4 weights, false for fp8.
if [[ "$PRECISION" == "fp4" ]]; then
    FP4_EXPERTS=true
else
    FP4_EXPERTS=false
fi
DSV4_ENV=(
  -e SGLANG_DEFAULT_THINKING=1 -e SGLANG_DSV4_REASONING_EFFORT=max
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=false -e SGLANG_USE_AITER=1
  -e SGLANG_USE_ROCM700A=$ROCM700A -e SGLANG_OPT_USE_FUSED_COMPRESS=true
  -e SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
  -e SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  -e SGLANG_OPT_FP8_WO_A_GEMM=false -e SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
  -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_OPT_USE_AITER_INDEXER=true
  -e SGLANG_OPT_USE_TILELANG_INDEXER=false -e SGLANG_OPT_USE_TILELANG_MHC_PRE=false
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=false -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  -e SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false -e SGLANG_ROCM_USE_MULTI_STREAM=false
  -e AITER_BF16_FP8_MOE_BOUND=0 -e SGLANG_DSV4_FP4_EXPERTS=$FP4_EXPERTS
)
DSV4_ENV_STR="${DSV4_ENV[*]}"
# A recipe carrying a `model:` block supplies its OWN docker env (below), so the
# DSV4 env must not leak into it; the DSV4 recipes keep the string above.
[[ "$HAS_MODEL" == "1" ]] && DSV4_ENV_STR=""
MORI_ENV="-e MORI_DISABLE_AUTO_XGMI=1 -e NCCL_IB_HCA=ionic -e NCCL_IB_GID_INDEX=1 -e NCCL_CROSS_NIC=1"
# Wide-EP (engine spans >1 node) adds mori all-to-all MoE tuning + the cross-node
# torch-dist socket NIC. Gated on nodes-per-engine>1 so EP<=8 recipes are untouched.
if (( PN_PER > 1 || DN_PER > 1 )); then
    # TC=104 (SL=3) = the ionic lossless RoCE queue (DSCP26/pri3); TC=96 is the
    # lossy pri0 queue (~1% BW) and wedges cross-node a2a under load. bf16
    # dispatch/combine matches the validated wide-EP run (job 13196).
    # The base MORI_ENV sets NCCL_IB_HCA=ionic (a spur-ism); on this fabric the
    # IB device names are the recipe's $IB (rdma0..7), and a wide engine's
    # cross-node TP/attention collectives ride NCCL, so point NCCL at the real
    # HCAs. Docker last-wins => this overrides the base value for wide recipes.
    MORI_ENV="$MORI_ENV \
-e NCCL_IB_HCA=$IB \
-e MORI_IB_GID_INDEX=1 \
-e SGLANG_MORI_DISPATCH_DTYPE=bf16 -e SGLANG_MORI_COMBINE_DTYPE=bf16 \
-e SGLANG_MORI_QP_PER_TRANSFER=4 -e SGLANG_MORI_NUM_WORKERS=4 \
-e MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000 -e MORI_IO_QP_MAX_SEND_WR=16384 \
-e MORI_IO_QP_MAX_CQE=32768 -e MORI_IO_QP_MAX_SGE=4 \
-e MORI_SHMEM_MODE=ISOLATION -e MORI_EP_LAUNCH_CONFIG_MODE=AUTO -e MORI_APP_LOG_LEVEL=INFO \
-e MORI_RDMA_SL=3 -e MORI_RDMA_TC=104 -e MORI_IO_SL=3 -e MORI_IO_TC=104 -e MORI_IO_TC_DISABLE=0 \
-e SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096 \
-e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 \
-e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 \
-e SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32 -e SGLANG_EAGER_INPUT_NO_COPY=true \
-e MORI_BOOTSTRAP_TIMEOUT=300"
    if [[ -n "$DIST_SOCK" ]]; then
        MORI_ENV="$MORI_ENV -e GLOO_SOCKET_IFNAME=$DIST_SOCK -e NCCL_SOCKET_IFNAME=$DIST_SOCK -e MORI_SOCKET_IFNAME=$DIST_SOCK"
    fi
fi

# Model-specific docker `-e` env + sglang server args from the recipe's optional
# `model:` block, written as bash arrays to model_flags.sh (sourced by
# prefill.sh/decode.sh). DSV4 recipes have no `model:` block -> empty arrays, so
# their generated docker argv is unchanged. Each server arg + its value MUST be a
# separate YAML list item so shlex.quote keeps "--foo" and "bar" as two tokens.
python3 - "$CONFIG_FILE" "$WORKDIR/model_flags.sh" <<'PY'
import shlex, sys, yaml
r = yaml.safe_load(open(sys.argv[1]))
model = r.get("model", {}) or {}
env = model.get("env", {}) or {}
server_args = model.get("server_args", []) or []
# YAML true/false parse to Python bool; render lowercase so env values stay
# byte-identical to shell (`=false`, not `=False`) -- SGLang parsing is
# case-sensitive for some of these.
def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
env_args = []
for k, v in env.items():
    env_args += ["-e", f"{k}={fmt(v)}"]
def q(items):
    return " ".join(shlex.quote(fmt(x)) for x in items)
with open(sys.argv[2], "w") as f:
    f.write(f"MODEL_ENV_ARGS=({q(env_args)})\n")
    f.write(f"MODEL_SERVER_ARGS=({q(server_args)})\n")
PY

# Optional topology / speculative-decode flags driven by the recipe. Base recipes
# (EP1/DP1, no mtp) leave the extra strings empty, preserving prior behavior.
#
# EP/DP are now PER ROLE: the DP-attention + ep-size flags come from PDP/PEP for
# prefill and DDP/DEP for decode. For every EP<=8 recipe DDP==PDP and DEP==PEP
# (decode inherits prefill), so both role strings equal the old single EXTRA_FLAGS
# and the generated argv is byte-identical. Oren's wide-EP recipes set decode
# EP/DP=16 while prefill stays EP8.
PREFILL_DPEP=""
(( PDP > 1 )) && PREFILL_DPEP="$PREFILL_DPEP --enable-dp-attention --dp-size $PDP"
(( PEP > 1 )) && PREFILL_DPEP="$PREFILL_DPEP --ep-size $PEP"
DECODE_DPEP=""
(( DDP > 1 )) && DECODE_DPEP="$DECODE_DPEP --enable-dp-attention --dp-size $DDP"
(( DEP > 1 )) && DECODE_DPEP="$DECODE_DPEP --ep-size $DEP"
# Flags shared by both roles (a2a backend, mtp). --max-total-tokens stays here for
# non-wide recipes; wide recipes carry a per-role prefill_max_total via wide_ep.
EXTRA_COMMON=""
[[ -n "$A2A" ]] && EXTRA_COMMON="$EXTRA_COMMON --moe-a2a-backend $A2A --deepep-mode normal"
[[ -n "$MAXTOK" ]] && EXTRA_COMMON="$EXTRA_COMMON --max-total-tokens $MAXTOK"
if [[ "$MTP_ENABLED" == "1" ]]; then
    EXTRA_COMMON="$EXTRA_COMMON --speculative-algorithm $MTP_ALGO \
--speculative-num-steps $MTP_STEPS --speculative-eagle-topk $MTP_TOPK \
--speculative-num-draft-tokens $MTP_DRAFT"
    # EAGLE3 (and other draft-model algos) need an external draft checkpoint;
    # built-in EAGLE (DSV4) omits draft_model_path and this stays unset.
    if [[ -n "$MTP_DRAFT_PATH" ]]; then
        DRAFT_RESOLVED="$(resolve_snapshot "$MTP_DRAFT_PATH")" || exit 1
        EXTRA_COMMON="$EXTRA_COMMON --speculative-draft-model-path $DRAFT_RESOLVED"
    fi
fi
# Prefix a leading space only when the arg is non-empty (keeps EP<=8 argv byte-
# identical: the wide-only strings are empty and contribute nothing).
sp() { [[ -n "$1" ]] && printf ' %s' "$1"; return 0; }
# --kv-cache-dtype is emitted only when the recipe sets wide_ep.kv_cache_dtype;
# the pre-wide DSV4 path had no such flag, so EP<=8 recipes omit it.
KV_FLAG=""
[[ -n "$KVDTYPE" ]] && KV_FLAG=" --kv-cache-dtype $KVDTYPE"
# Assemble the per-role tail: role DP/EP + shared + wide common + role-specific
# wide extras. All wide pieces (WECOMMON/PEXTRA/DEXTRA) are empty for EP<=8.
PREFILL_TAIL="$PREFILL_DPEP$EXTRA_COMMON$(sp "$WECOMMON")$(sp "$PEXTRA")"
DECODE_TAIL="$DECODE_DPEP$EXTRA_COMMON$(sp "$WECOMMON")$(sp "$DEXTRA")"
echo "prefill tail:${PREFILL_TAIL:-<none>} | decode tail:${DECODE_TAIL:-<none>} (pep=$PEP pdp=$PDP dep=$DEP ddp=$DDP mtp=$MTP_ENABLED)"

if [[ "$HAS_MODEL" == "1" ]]; then
    # Generic path (e.g. Kimi): attention + swa from the recipe, model parsers /
    # quirks ride MODEL_SERVER_ARGS. Single `--attention-backend` when the recipe
    # sets `attention_backend`; split `--prefill-/--decode-attention-backend` when
    # it sets the per-role keys. swa dropped when the recipe omits it.
    ATTN_FLAGS=""
    [[ -n "$ATTN" ]]  && ATTN_FLAGS="$ATTN_FLAGS --attention-backend $ATTN"
    [[ -n "$PATTN" ]] && ATTN_FLAGS="$ATTN_FLAGS --prefill-attention-backend $PATTN"
    [[ -n "$DATTN" ]] && ATTN_FLAGS="$ATTN_FLAGS --decode-attention-backend $DATTN"
    SWA_FLAG=""
    [[ -n "$SWA" ]] && SWA_FLAG=" --swa-full-tokens-ratio $SWA"
    PREFILL_COMMON_FLAGS="--trust-remote-code --tp $PTP --disable-radix-cache \
$ATTN_FLAGS --max-running-requests $PMAXREQ --page-size $PAGE \
--mem-fraction-static $PMEMFRAC$SWA_FLAG \
--chunked-prefill-size $PCHUNK \
--disaggregation-transfer-backend $XFER --disaggregation-ib-device $IB$KV_FLAG$PREFILL_TAIL"
    DECODE_COMMON_FLAGS="--trust-remote-code --tp $DTP --disable-radix-cache \
$ATTN_FLAGS --max-running-requests $DMAXREQ --page-size $PAGE \
--mem-fraction-static $DMEMFRAC$SWA_FLAG \
--chunked-prefill-size $CHUNK \
--disaggregation-transfer-backend $XFER --disaggregation-ib-device $IB$KV_FLAG$DECODE_TAIL"
else
    # DSV4 path: for EP<=8 recipes (PTP==DTP, no wide_ep) both role strings equal
    # the pre-Kimi launcher's COMMON_FLAGS exactly.
    PREFILL_COMMON_FLAGS="--trust-remote-code --tp $PTP --disable-radix-cache \
--attention-backend $ATTN --max-running-requests $PMAXREQ --page-size $PAGE \
--mem-fraction-static $PMEMFRAC --swa-full-tokens-ratio $SWA \
--chunked-prefill-size $PCHUNK --disable-shared-experts-fusion \
--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
--disaggregation-transfer-backend $XFER --disaggregation-ib-device $IB$KV_FLAG$PREFILL_TAIL"
    DECODE_COMMON_FLAGS="--trust-remote-code --tp $DTP --disable-radix-cache \
--attention-backend $ATTN --max-running-requests $DMAXREQ --page-size $PAGE \
--mem-fraction-static $DMEMFRAC --swa-full-tokens-ratio $SWA \
--chunked-prefill-size $CHUNK --disable-shared-experts-fusion \
--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
--disaggregation-transfer-backend $XFER --disaggregation-ib-device $IB$KV_FLAG$DECODE_TAIL"
fi

DOCKER_COMMON="--rm --network host --ipc host --shm-size 32g --privileged \
--security-opt seccomp=unconfined \
--device /dev/kfd --device /dev/dri --device /dev/infiniband \
-v /it-share:/it-share:ro -v $HOME:/host_home $CHECKOUT_DOCKER_ARGS"
# Optional extra docker args (e.g. bind-mounting a locally-built lib for
# validation). Empty by default so the docker argv is byte-identical otherwise.
[[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER_COMMON="$DOCKER_COMMON ${EXTRA_DOCKER_ARGS}"

# Per-role wide-EP docker env (MORI dispatch-token tuning etc.). Empty for EP<=8
# recipes; carries its own leading space so an empty value leaves the docker argv
# byte-identical (no stray double space).
PENV_ARG=""; [[ -n "$PENV" ]] && PENV_ARG=" $PENV"
DENV_ARG=""; [[ -n "$DENV" ]] && DENV_ARG=" $DENV"

# ---------------------------------------------------------------------------
# Write per-role scripts that srun dispatches to each compute node.
# ---------------------------------------------------------------------------
# These are UNQUOTED `<<EOF` heredocs, so $MORI_ENV/$DSV4_ENV_STR/$COMMON_FLAGS/
# $IMAGE/$MODEL_PATH expand now (at generation). model_flags.sh is sourced at
# runtime for the model's env/server arrays, so the `${MODEL_ENV_ARGS[@]}` /
# `${MODEL_SERVER_ARGS[@]}` refs are backslash-escaped to survive into the script
# and expand after `source`. For DSV4 those arrays are empty and $DSV4_ENV_STR is
# set, so the resulting docker argv is byte-identical to the pre-Kimi launcher.
cat > "$WORKDIR/install_checkout_sglang.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

case "${SGLANG_USE_CHECKOUT_RUNTIME:-1}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    echo "[checkout-sglang] disabled; using image-baked sglang"
    exit 0
    ;;
esac

CHECKOUT_SRC="${CHECKOUT_SRC:-/sglang-checkout}"
RUNTIME_CHECKOUT="${RUNTIME_CHECKOUT:-/tmp/sglang-checkout-runtime}"

if [[ ! -f "$CHECKOUT_SRC/python/sglang/version.py" ]]; then
  echo "[checkout-sglang] ERROR: invalid checkout mount: $CHECKOUT_SRC" >&2
  exit 1
fi

echo "[checkout-sglang] reinstalling sglang from $CHECKOUT_SRC"
rm -rf "$RUNTIME_CHECKOUT"
mkdir -p "$RUNTIME_CHECKOUT"
tar --exclude='__pycache__' --exclude='*.pyc' \
  -C "$CHECKOUT_SRC" -cf - . | tar -C "$RUNTIME_CHECKOUT" -xf -

git config --global --add safe.directory "$RUNTIME_CHECKOUT" || true

# The ROCm pyproject variant is the one used by AMD CI. Mutate only the private
# /tmp copy so prefill/decode/bench never race on the read-only checkout mount.
rm -f "$RUNTIME_CHECKOUT/python/pyproject.toml"
cp "$RUNTIME_CHECKOUT/python/pyproject_other.toml" "$RUNTIME_CHECKOUT/python/pyproject.toml"
for f in README.md LICENSE; do
  if [[ -f "$RUNTIME_CHECKOUT/$f" && ! -e "$RUNTIME_CHECKOUT/python/$f" ]]; then
    cp "$RUNTIME_CHECKOUT/$f" "$RUNTIME_CHECKOUT/python/$f"
  fi
done

python3 -m pip uninstall -y sglang || true
python3 -m pip install --no-deps --no-build-isolation -e "$RUNTIME_CHECKOUT/python"

export RUNTIME_CHECKOUT
export PYTHONPATH="$RUNTIME_CHECKOUT/python:${PYTHONPATH:-}"
python3 - <<'PY'
import importlib.metadata
import os
import subprocess
import sglang

checkout = os.environ["RUNTIME_CHECKOUT"]
expected = os.path.realpath(os.path.join(checkout, "python", "sglang")) + os.sep
actual = os.path.realpath(os.path.dirname(sglang.__file__)) + os.sep
try:
    sha = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
except Exception:
    sha = os.environ.get("SGLANG_CHECKOUT_SHA", "unknown")

print(f"[checkout-sglang] sha={sha}")
print(f"[checkout-sglang] sglang_file={sglang.__file__}")
print(f"[checkout-sglang] sglang_version={importlib.metadata.version('sglang')}")
if not actual.startswith(expected):
    raise SystemExit(f"sglang did not import from checkout: {sglang.__file__}")
PY
EOF

cat > "$WORKDIR/install_checkout_router.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

case "${SGLANG_USE_CHECKOUT_RUNTIME:-1}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    echo "[checkout-router] disabled; using image-baked sglang-router"
    python3 - <<'PY' || true
import importlib.metadata
import sglang_router

print(f"[checkout-router] sglang_router_file={sglang_router.__file__}")
print(
    "[checkout-router] sglang_router_version="
    f"{importlib.metadata.version('sglang-router')}"
)
try:
    import sglang_router.sglang_router_rs as rs

    print(f"[checkout-router] sglang_router_rs_file={rs.__file__}")
except Exception as exc:
    print(f"[checkout-router] sglang_router_rs_import_error={exc}")
PY
    exit 0
    ;;
esac

RUNTIME_CHECKOUT="${RUNTIME_CHECKOUT:-/tmp/sglang-checkout-runtime}"
ROUTER_SRC="$RUNTIME_CHECKOUT/sgl-model-gateway/bindings/python"
WHEEL_DIR="${SGLANG_ROUTER_WHEEL_DIR:-/tmp/sglang-router-wheels}"

if [[ ! -f "$ROUTER_SRC/pyproject.toml" ]]; then
  echo "[checkout-router] ERROR: invalid router checkout: $ROUTER_SRC" >&2
  exit 1
fi

echo "[checkout-router] building sglang-router from $ROUTER_SRC"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
python3 -m maturin --version >/dev/null 2>&1 \
  || python3 -m pip install --no-cache-dir "maturin<1.14"

# Match the ROCm image build recipe when compiling from the checkout copy.
if [[ -f "$RUNTIME_CHECKOUT/sgl-model-gateway/Cargo.toml" ]]; then
  sed -i -E 's|^(smg-[a-zA-Z-]+)\s*=\s*"~1\.0\.0"|\1 = "=1.0.0"|' \
    "$RUNTIME_CHECKOUT/sgl-model-gateway/Cargo.toml"
fi

rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"
(
  cd "$ROUTER_SRC"
  ulimit -n 65536 || true
  python3 -m maturin build --release --features vendored-openssl --out "$WHEEL_DIR"
)

python3 -m pip uninstall -y sglang-router || true
python3 -m pip install --force-reinstall --no-deps "$WHEEL_DIR"/*.whl

python3 - <<'PY'
import importlib.metadata
import sglang_router
import sglang_router.sglang_router_rs as rs
from sglang_router.sglang_router_rs import Router

print(f"[checkout-router] sglang_router_file={sglang_router.__file__}")
print(
    "[checkout-router] sglang_router_version="
    f"{importlib.metadata.version('sglang-router')}"
)
print(f"[checkout-router] sglang_router_rs_file={rs.__file__}")
print(f"[checkout-router] Router={Router}")
PY
EOF

if (( PN_PER > 1 || DN_PER > 1 )); then
  # Wide-EP path: an engine spans >1 node. Entry scripts add torch-dist
  # rendezvous args from NODE_RANK/NNODES/DIST_ADDR; launch scripts take the
  # per-node rank as $1..$3 and forward it into the container. Only reached when
  # a recipe sets TP>GPUS_PER_NODE, so EP<=8 recipes never take this branch.
  cat > "$WORKDIR/prefill_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_checkout_sglang.sh"
if [[ "\${SGLANG_USE_CHECKOUT_RUNTIME:-1}" != "0" ]]; then
  export PYTHONPATH=/tmp/sglang-checkout-runtime/python:\${PYTHONPATH:-}
fi
DIST_ARGS=""
if [[ "\${NNODES:-1}" != "1" ]]; then
  DIST_ARGS="--nnodes \$NNODES --node-rank \$NODE_RANK --dist-init-addr \$DIST_ADDR:\${DIST_PORT:-29500}"
fi
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \
  $PREFILL_COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \$DIST_ARGS \
  --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
EOF

  cat > "$WORKDIR/decode_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_checkout_sglang.sh"
if [[ "\${SGLANG_USE_CHECKOUT_RUNTIME:-1}" != "0" ]]; then
  export PYTHONPATH=/tmp/sglang-checkout-runtime/python:\${PYTHONPATH:-}
fi
DIST_ARGS=""
if [[ "\${NNODES:-1}" != "1" ]]; then
  DIST_ARGS="--nnodes \$NNODES --node-rank \$NODE_RANK --dist-init-addr \$DIST_ADDR:\${DIST_PORT:-29500}"
fi
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \
  $DECODE_COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \$DIST_ARGS \
  --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
EOF

  # Cross-node MoE a2a over mori needs the HOST ionic RDMA userspace libs: the
  # image's libionic-rdmav34.so lacks ionic_dv_create_cq_ex, which makes the mori
  # QP INIT->RTR transition fail (ionic.cpp ModifyInit2Rtr). Each mount is guarded
  # by an existence check, so nodes without ionic (e.g. mi355x rdma) mount nothing.
  IONIC_LIB_MOUNT='IONIC_MOUNTS=()
host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
[ -n "$host_ionic" ] && [ -e "$host_ionic" ] && IONIC_MOUNTS+=( -v "$host_ionic:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro" )
[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ] && IONIC_MOUNTS+=( -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro )
[ -e /etc/libibverbs.d/ionic.driver ] && IONIC_MOUNTS+=( -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro )'

  cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
NODE_RANK="\${1:-0}"; NNODES="\${2:-1}"; DIST_ADDR="\${3:-}"
$IONIC_LIB_MOUNT
docker rm -f mi355x_prefill 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_prefill \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NODE_RANK="\$NODE_RANK" -e NNODES="\$NNODES" -e DIST_ADDR="\$DIST_ADDR" -e DIST_PORT=${DIST_PORT} \
  "\${IONIC_MOUNTS[@]}" \
  $MORI_ENV$PENV_ARG $DSV4_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/prefill_entry.sh
EOF

  cat > "$WORKDIR/decode.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
NODE_RANK="\${1:-0}"; NNODES="\${2:-1}"; DIST_ADDR="\${3:-}"
$IONIC_LIB_MOUNT
docker rm -f mi355x_decode 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_decode \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NODE_RANK="\$NODE_RANK" -e NNODES="\$NNODES" -e DIST_ADDR="\$DIST_ADDR" -e DIST_PORT=${DIST_PORT} \
  "\${IONIC_MOUNTS[@]}" \
  $MORI_ENV$DENV_ARG $DSV4_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/decode_entry.sh
EOF

else
  # Single-node-per-engine path (all EP<=8 recipes): byte-identical to the
  # pre-wide launcher. No NODE_RANK/NNODES/DIST_* plumbing is emitted.
  cat > "$WORKDIR/prefill_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_checkout_sglang.sh"
if [[ "\${SGLANG_USE_CHECKOUT_RUNTIME:-1}" != "0" ]]; then
  export PYTHONPATH=/tmp/sglang-checkout-runtime/python:\${PYTHONPATH:-}
fi
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \
  $PREFILL_COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \
  --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
EOF

  cat > "$WORKDIR/decode_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_checkout_sglang.sh"
if [[ "\${SGLANG_USE_CHECKOUT_RUNTIME:-1}" != "0" ]]; then
  export PYTHONPATH=/tmp/sglang-checkout-runtime/python:\${PYTHONPATH:-}
fi
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \
  $DECODE_COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \
  --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
EOF

  cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
docker rm -f mi355x_prefill 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_prefill \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV$PENV_ARG $DSV4_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/prefill_entry.sh
EOF

  cat > "$WORKDIR/decode.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
docker rm -f mi355x_decode 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_decode \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV$DENV_ARG $DSV4_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/decode_entry.sh
EOF
fi

# Probe payload + validator (separate files to avoid quoting inside the
# bench.sh `bash -lc '...'` block). One real request exercises the full
# prefill->decode KV handoff before we commit to the whole sweep.
cat > "$WORKDIR/probe.json" <<'JSON'
{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
JSON
cat > "$WORKDIR/assert_nonempty.py" <<'PY'
import sys, json
d = json.load(sys.stdin)
t = d.get("text", "") if isinstance(d, dict) else ""
if not (t and t.strip()):
    print("[probe] empty/invalid output:", str(d)[:200])
    sys.exit(1)
print("[probe] ok:", t[:80].replace("\n", " "))
PY

# Prefill health-wait + router --prefill args. For PW=1 this is a single endpoint
# (byte-identical to the pre-fan-out launcher); for PW>1 it iterates the comma-
# separated engine node0 IPs in PCSV ($3), health-waits each, and passes one
# --prefill per engine so the router fans requests across all prefill engines.
if (( PW > 1 )); then
  PREFILL_WAIT_ROUTER="    IFS=',' read -ra PIPS <<< \"\${PCSV:-\$PIP}\"
    PREFILL_ARGS=\"\"
    for pip in \"\${PIPS[@]}\"; do
      echo \"[wait] prefill \$pip\"; for i in \$(seq 1 600); do curl -sf http://\$pip:$PPORT/health >/dev/null && break; sleep 5; done
      PREFILL_ARGS=\"\$PREFILL_ARGS --prefill http://\$pip:$PPORT $PBOOT\"
    done
    echo \"[wait] decode\";  for i in \$(seq 1 600); do curl -sf http://\$DIP:$DPORT/health >/dev/null && break; sleep 5; done
    python3 -m sglang_router.launch_router \\
      --pd-disaggregation \\
      \$PREFILL_ARGS \\
      --decode http://\$DIP:$DPORT \\
      --host 0.0.0.0 --port $LBPORT \\
      --disable-circuit-breaker &"
else
  PREFILL_WAIT_ROUTER="    echo \"[wait] prefill\"; for i in \$(seq 1 600); do curl -sf http://\$PIP:$PPORT/health >/dev/null && break; sleep 5; done
    echo \"[wait] decode\";  for i in \$(seq 1 600); do curl -sf http://\$DIP:$DPORT/health >/dev/null && break; sleep 5; done
    python3 -m sglang_router.launch_router       --pd-disaggregation       --prefill http://\$PIP:$PPORT $PBOOT       --decode http://\$DIP:$DPORT       --host 0.0.0.0 --port $LBPORT       --disable-circuit-breaker &"
fi

# Bench script runs on the prefill node; \$PIP/\$DIP injected at srun time.
cat > "$WORKDIR/bench.sh" <<EOF
#!/bin/bash
set -e
PIP=\$1; DIP=\$2; PCSV=\${3:-\$PIP}
docker rm -f mi355x_bench 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_bench \
  -e PIP=\$PIP -e DIP=\$DIP -e PCSV=\$PCSV \
  $IMAGE bash -lc '
    CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
    bash \$CIDIR/install_checkout_sglang.sh
    if [ "\${SGLANG_USE_CHECKOUT_RUNTIME:-1}" != "0" ]; then
      export PYTHONPATH=/tmp/sglang-checkout-runtime/python:\${PYTHONPATH:-}
    else
      export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}
    fi
    bash \$CIDIR/install_checkout_router.sh
$PREFILL_WAIT_ROUTER
    for i in \$(seq 1 30); do curl -sf http://127.0.0.1:$LBPORT/health >/dev/null && break; sleep 2; done
    echo "[probe] PD end-to-end check via LB"
    curl -sf -X POST http://127.0.0.1:$LBPORT/generate \
      -H "content-type: application/json" -d @\$CIDIR/probe.json > \$CIDIR/probe_out.json \
      || { echo "[probe] request failed -- PD path not serving; aborting before sweep"; exit 1; }
    python3 \$CIDIR/assert_nonempty.py < \$CIDIR/probe_out.json \
      || { echo "[probe] empty/invalid generation; aborting before sweep"; exit 1; }
    # Correctness gate runs BEFORE the perf sweep: if the model is wrong there
    # is no point spending ~15min measuring how fast it is wrong, so a failure
    # here exits immediately and the sweep never runs.
    if [ "$ACC_ENABLED" = "1" ]; then
      echo "=== GSM8K accuracy gate (num_questions=$ACC_NQ shots=$ACC_SHOTS) ==="
      DP_ARG=""
      [ -s \$CIDIR/gsm8k_test.jsonl ] && DP_ARG="--data-path \$CIDIR/gsm8k_test.jsonl"
      python3 -m sglang.test.few_shot_gsm8k \
        --num-shots $ACC_SHOTS --num-questions $ACC_NQ --parallel $MAXREQ \
        --max-new-tokens 512 --host http://127.0.0.1 --port $LBPORT \
        \$DP_ARG 2>&1 | tee \$CIDIR/gsm8k.log
      ACC=\$(grep -oE "Accuracy: [0-9.]+" \$CIDIR/gsm8k.log | tail -1 | cut -d" " -f2)
      [ -n "\$ACC" ] || { echo "[gsm8k] could not parse accuracy from harness output"; exit 1; }
      python3 \$CIDIR/check_acc.py "\$ACC" "$ACC_THR" || { echo "[gsm8k] accuracy below threshold -- failing before sweep"; exit 1; }
    fi
    for C in ${CONCS//,/ }; do
      echo "=== concurrency=\$C ==="
      OUT=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/raw_conc\${C}.json
      rm -f \$OUT
      python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port $LBPORT --model $MODEL_PATH \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio $RRR --max-concurrency \$C \
        --num-prompts \$((C*$NPF)) --warmup-requests \$C \
        --output-file \$OUT || true
    done
  '
EOF
chmod +x "$WORKDIR"/*.sh

# ---------------------------------------------------------------------------
# Orchestration drive (runs inside the salloc allocation on the login node).
# ---------------------------------------------------------------------------
# drive.sh splits the allocation into the first PW nodes (prefill) and the next
# DW nodes (decode), launches one server per node, then benches. For 1P1D
# (PW=DW=1) this is exactly prefill-on-node-A / decode-on-node-B. Larger PW/DW
# reserve 2P2D / 1P3D / 3P1D: all servers come up, but the load balancer and
# bench still target the first prefill + first decode (multi-P/D fan-out is the
# remaining LB piece), so a >1 topology logs an explicit NOTE rather than
# silently producing partial-coverage numbers.
cat > "$WORKDIR/drive.sh" <<'DRIVE'
#!/bin/bash
set -x
WORKDIR="$1"; PW="${2:-1}"; DW="${3:-1}"; PN_PER="${4:-1}"; DN_PER="${5:-1}"; DIST_NIC="${6:-}"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
# Resolve a node's IP. When a cross-node dist NIC is named (wide engines set
# $DIST_NIC to the recipe's dist_socket_ifname), read the address off that
# interface on the node itself -- the multi-node dist-init-addr must be a
# LOCALLY-BINDABLE IP (torch-dist + tokenizer ZMQ bind to it), and a node's
# forward DNS can be stale/point at a non-local mgmt alias (observed: a decode
# root whose hostname resolved to an unpingable IP, ZMQ bind => "Cannot assign
# requested address"). Fall back to getent if the interface lookup is empty.
# $DIST_NIC empty (every EP<=8 recipe) => getent path only => unchanged behavior.
resolve_ip() {
  local n="$1" ip=""
  if [[ -n "$DIST_NIC" ]]; then
    ip=$(srun --overlap -N1 --nodelist="$n" ip -4 -o addr show "$DIST_NIC" 2>/dev/null \
           | awk '{print $4}' | cut -d/ -f1 | head -1)
  fi
  [[ -z "$ip" ]] && ip=$(getent ahostsv4 "$n" | head -1 | awk '{print $1}')
  echo "$ip"
}
# SLURM canonicalizes (sorts) SLURM_JOB_NODELIST, so the requested --nodelist
# order is already lost here. To keep a "slow-root" node out of any engine's
# rank0 slot -- e.g. mia1-p01-g20, whose MORI EP bootstrap reaches the connect
# stage ~16s late and loses the hardcoded 10s worker-connect race whenever it is
# rank0 -- push every node named in SLURM_DIST_TAIL to the END of the list. It
# then lands in a trailing decode-worker position, where its late ranks connect
# into the decode root's 30s accept window (harmless). Comma-separated; unmatched
# names ignored; empty (every EP<=8 recipe) => no-op, order unchanged.
if [[ -n "${SLURM_DIST_TAIL:-}" ]]; then
  IFS=',' read -ra _TAIL <<< "$SLURM_DIST_TAIL"
  _HEAD=(); _TL=()
  for n in "${NODES[@]}"; do
    _skip=0; for t in "${_TAIL[@]}"; do [[ "$n" == "$t" ]] && { _skip=1; break; }; done
    if (( _skip )); then _TL+=("$n"); else _HEAD+=("$n"); fi
  done
  NODES=("${_HEAD[@]}" "${_TL[@]}")
  echo "[drive] SLURM_DIST_TAIL=$SLURM_DIST_TAIL -> node order: ${NODES[*]}"
fi
# Each engine may span PN_PER/DN_PER nodes (ceil(TP/GPUs-per-node)); PN_PER=1 for
# EP<=8 so this reduces to the original one-node-per-worker split.
PN_TOTAL=$((PW * PN_PER)); DN_TOTAL=$((DW * DN_PER))
PNODES=("${NODES[@]:0:PN_TOTAL}")
DNODES=("${NODES[@]:PN_TOTAL:DN_TOTAL}")
PNODE="${PNODES[0]}"; DNODE="${DNODES[0]}"
PIP=$(resolve_ip "$PNODE")
DIP=$(resolve_ip "$DNODE")
echo "[drive] prefill nodes: ${PNODES[*]} ; decode nodes: ${DNODES[*]}"
echo "[drive] bench targets prefill=$PNODE($PIP) decode=$DNODE($DIP)"
if (( DW > 1 )); then
  echo "[drive] NOTE: router + bench use the first decode engine only;"
  echo "[drive]       multi-decode fan-out is not wired yet (LB work)."
fi
# Each server's srun runs here on the login node and returns exactly when its
# compute-node container exits. Wrap it so the return code lands in a marker
# file on shared NFS. The monitor then watches for markers instead of polling
# PIDs -- unambiguous (no zombie/kill -0 guesswork) and it records which role
# died and with what code. (A hung-but-alive server is NOT caught here; that is
# bounded by bench.sh's health-wait timeout.)
rm -f "$WORKDIR"/server_exit_* "$WORKDIR/bench_exit"
# Launch PW prefill engines; each spans PN_PER nodes as its own torch-dist group
# (engine node0 = dist-init addr; node_rank = position WITHIN the engine, so an
# engine's ranks are 0..PN_PER-1, not a global index). PN_PER=1 => NNODES=1 in
# the entry script => dist args dropped => byte-identical single-node launch.
# Collect each engine's node0 IP into PCSV for the router's prefill fan-out.
PCSV=""
for ((k=0; k<PW; k++)); do
  e0="${PNODES[k*PN_PER]}"
  eip=$(resolve_ip "$e0")
  PCSV="${PCSV:+$PCSV,}$eip"
  for ((j=0; j<PN_PER; j++)); do
    n="${PNODES[k*PN_PER + j]}"
    ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/prefill.sh" "$j" "$PN_PER" "$eip" > "$WORKDIR/prefill_$n.log" 2>&1
      echo "prefill@$n rc=$?" > "$WORKDIR/server_exit_prefill_$n" ) &
  done
done
for ((k=0; k<DW; k++)); do
  e0="${DNODES[k*DN_PER]}"
  eip=$(resolve_ip "$e0")
  for ((j=0; j<DN_PER; j++)); do
    n="${DNODES[k*DN_PER + j]}"
    ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/decode.sh" "$j" "$DN_PER" "$eip" > "$WORKDIR/decode_$n.log" 2>&1
      echo "decode@$n rc=$?" > "$WORKDIR/server_exit_decode_$n" ) &
  done
done
echo "[drive] prefill engine endpoints (fan-out): $PCSV"
sleep 5
# Bench in the background with its own marker, so the wait loop is purely file
# based: finish when bench writes its marker, abort if any server marker shows up
# first (a server died before the sweep completed).
( srun --overlap -N1 --nodelist="$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" "$PCSV" > "$WORKDIR/bench.log" 2>&1
  echo $? > "$WORKDIR/bench_exit" ) &
BENCH_BG=$!
# Stream bench output live and poll the markers with xtrace OFF, so the console
# shows clean benchmark/accuracy output instead of a compgen/sleep trace every
# 10s. (Mirrors NVIDIA's launch_gb200.sh, which set +x around its log stream.)
touch "$WORKDIR/bench.log"
tail -n +1 -F "$WORKDIR/bench.log" 2>/dev/null &
TAIL_PID=$!
set +x
RC=0
while [[ ! -f "$WORKDIR/bench_exit" ]]; do
  if compgen -G "$WORKDIR/server_exit_*" > /dev/null; then
    echo "[drive] ERROR: a server exited early before bench finished:"
    cat "$WORKDIR"/server_exit_* || true
    kill "$BENCH_BG" 2>/dev/null || true
    RC=1
    break
  fi
  sleep 10
done
set -x
kill "$TAIL_PID" 2>/dev/null || true
[[ "$RC" -eq 0 ]] && RC=$(cat "$WORKDIR/bench_exit" 2>/dev/null || echo 1)
echo "[drive] bench finished (rc=$RC), tearing down"
for n in "${PNODES[@]}"; do srun --overlap -N1 --nodelist="$n" docker kill mi355x_prefill >/dev/null 2>&1 || true; done
for n in "${DNODES[@]}"; do srun --overlap -N1 --nodelist="$n" docker kill mi355x_decode  >/dev/null 2>&1 || true; done
exit "$RC"
DRIVE
chmod +x "$WORKDIR/drive.sh"

NODELIST_ARG=()
[[ -n "${SLURM_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="$SLURM_NODELIST")

# Request whole nodes so a co-scheduled job can't share a node and skew the
# benchmark numbers. Toggle off with SLURM_EXCLUSIVE=0 on partitions that
# disallow --exclusive.
EXCLUSIVE_ARG=()
[[ "${SLURM_EXCLUSIVE:-1}" == "1" ]] && EXCLUSIVE_ARG=(--exclusive)

# Keep the scheduler off known-bad nodes (e.g. a host whose ionic RDMA driver
# ABI mismatches the container, where MORI reports "no active RDMA device" and
# the disagg server dies on init). Comma-separated node list.
EXCLUDE_ARG=()
[[ -n "${SLURM_EXCLUDE:-}" ]] && EXCLUDE_ARG=(--exclude="$SLURM_EXCLUDE")

# Nodes = sum over engines of nodes-per-engine. EP<=8 (PN_PER=DN_PER=1) gives the
# original PW+DW (1P1D -> 2 nodes); wide EP16 1P1D gives 2+2 = 4 nodes.
TOTAL_NODES=$(( PW * PN_PER + DW * DN_PER ))

# Name the allocation <RUNNER_NAME>-<GITHUB_RUN_ID>-<config> so the workflow's
# cleanup steps can scancel precisely instead of a blanket `squeue --me` that
# would kill a concurrent matrix leg. RUNNER_NAME alone is not assumed unique;
# GITHUB_RUN_ID + config make the name unique per matrix leg regardless.
JOB_NAME="mi355x-ci-${RUNNER_NAME:-norunner}-${GITHUB_RUN_ID:-0}-${MATRIX_CONFIG_NAME}"

set +e
salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \
    bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW" "$PN_PER" "$DN_PER" "$DIST_SOCK"
SALLOC_RC=$?
set -e

# bench output already streamed live from drive.sh (tail -F). drive.sh exits
# non-zero when a server died or bench failed; on failure dump bench.log + the
# server logs (the actual root cause). We still fall through to normalize
# whatever raw results the completed concurrencies produced -- partial perf data
# is worth uploading -- and propagate the failure via the exit code at the end.
if [[ "$SALLOC_RC" -ne 0 ]]; then
    echo "ERROR: allocation/bench failed (rc=$SALLOC_RC); bench + server logs:" >&2
    echo "--- bench.log (tail) ---"; tail -40 "$WORKDIR/bench.log" 2>/dev/null || true
    for f in "$WORKDIR"/prefill_*.log "$WORKDIR"/decode_*.log; do
        [[ -f "$f" ]] && { echo "--- $f (tail) ---"; tail -30 "$f"; }
    done
fi

# Surface the GSM8K accuracy in the job summary -- it scrolls past in the live
# log, and the perf table (collect-results/summarize.py) doesn't include it.
if [[ "$ACC_ENABLED" == "1" && -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    ACC_LINE=$(grep -aoE "Accuracy: [0-9.]+" "$WORKDIR/bench.log" 2>/dev/null | tail -1 || true)
    {
        echo "### GSM8K accuracy gate — ${MATRIX_CONFIG_NAME}"
        echo '```'
        echo "${ACC_LINE:-Accuracy: <not found in bench.log>}   (threshold > ${ACC_THR})"
        echo '```'
    } >> "$GITHUB_STEP_SUMMARY"
fi

# ---------------------------------------------------------------------------
# Normalize raw bench_serving output -> process_result.py schema.
#
# bench_serving and process_result.py disagree on field names, so we remap the
# last JSON line of each raw file. If bench_serving ever renames an output
# field, the KeyError raised here (rather than a silently wrong table) is the
# signal to update this mapping. Field-by-field:
#
#   bench_serving key          ->  process_result.py key       (purpose)
#   --------------------------     ------------------------     -------------------------
#   max_concurrency            ->  max_concurrency             (sweep point; falls back to $C)
#   total_throughput           ->  total_token_throughput      (in+out tok/s, tput_per_gpu)
#   output_throughput          ->  output_throughput           (out tok/s, output_tput_per_gpu)
#   median_ttft_ms             ->  median_ttft_ms              (TTFT; /1000 -> s)
#   median_tpot_ms             ->  median_tpot_ms              (TPOT; -> interactivity)
#   median_e2e_latency_ms      ->  median_e2el_ms              (E2E latency; /1000 -> s)
#   (none; injected here)      ->  model_id                    (served model, from $MODEL_PATH)
# ---------------------------------------------------------------------------
# GPU counts are per-ROLE totals across all engines: PW prefill engines of PTP
# GPUs each, DW decode engines of DTP each. For EP<=8 (PW=DW=1) this is PTP/DTP
# exactly, so the filename fields are unchanged. process_result.py reads the
# _ctx_/_gen_ fields as PREFILL_GPUS/DECODE_GPUS for its per-GPU throughput math,
# so they must be the role totals (Oren EP16: ctx=2*8=16, gen=1*16=16, gpus=32).
PREFILL_GPUS_TOTAL=$((PW * PTP)); DECODE_GPUS_TOTAL=$((DW * DTP))
TOTAL_GPUS=$((PREFILL_GPUS_TOTAL + DECODE_GPUS_TOTAL))
PROCESSED=0
for C in ${CONCS//,/ }; do
    RAW="$WORKDIR/raw_conc${C}.json"
    [[ -f "$RAW" ]] || { echo "WARN: missing $RAW"; continue; }
    DEST="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${MATRIX_CONFIG_NAME}_conc${C}_gpus_${TOTAL_GPUS}_ctx_${PREFILL_GPUS_TOTAL}_gen_${DECODE_GPUS_TOTAL}.json"
    MODEL_ID="$MODEL_PATH" python3 - "$RAW" "$DEST" "$C" <<'PY'
import json, os, sys
raw_path, dest, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
line = [l for l in open(raw_path).read().splitlines() if l.strip()][-1]
r = json.loads(line)
norm = {
    "max_concurrency": r.get("max_concurrency") or conc,
    "model_id": os.environ["MODEL_ID"],
    "total_token_throughput": r["total_throughput"],
    "output_throughput": r["output_throughput"],
    "median_ttft_ms": r["median_ttft_ms"],
    "median_tpot_ms": r["median_tpot_ms"],
    "median_e2el_ms": r["median_e2e_latency_ms"],
}
json.dump(norm, open(dest, "w"), indent=2)
print("normalized ->", dest)
PY
    PROCESSED=$((PROCESSED + 1))
done

# Propagate a benchmark/allocation failure even though we emitted partial
# results above (the workflow uploads them with `always()`).
if [[ "$SALLOC_RC" -ne 0 ]]; then
    echo "ERROR: benchmark failed (rc=$SALLOC_RC); emitted $PROCESSED partial result file(s)." >&2
    exit "$SALLOC_RC"
fi
if [[ "$PROCESSED" -eq 0 ]]; then
    echo "ERROR: no result files produced" >&2
    exit 1
fi
echo "Done. $PROCESSED result file(s) in $GITHUB_WORKSPACE."

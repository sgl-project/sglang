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

# Scheduler profile. `slurm` (default) is the amd-sglang MI355X cluster and is
# the original code path, unchanged. `spur` is the TW PIT cluster, which runs
# Spur -- a Slurm-compatible scheduler that differs in three ways that matter
# here: salloc takes no trailing command, `scontrol show hostnames` does not
# exist (its nodelist env var is already comma-expanded), and `srun --overlap`
# requires an explicit --jobid instead of inheriting one from the environment.
CLUSTER="${CLUSTER:-slurm}"
case "$CLUSTER" in
    slurm|spur) ;;
    *) echo "ERROR: CLUSTER must be 'slurm' or 'spur', got '$CLUSTER'" >&2; exit 1 ;;
esac

# Spur reads a two-field time limit as HH:MM where Slurm reads MM:SS, so "30:00"
# silently becomes 30 HOURS. Require the unambiguous HH:MM:SS / D-HH:MM:SS form
# there rather than let a typo book a node for a day.
if [[ "$CLUSTER" == "spur" && ! "$TIME_LIMIT" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "ERROR: on spur TIME_LIMIT must be HH:MM:SS or D-HH:MM:SS (got '$TIME_LIMIT');" >&2
    echo "       a two-field value like '30:00' is parsed as 30 hours, not 30 minutes." >&2
    exit 1
fi

# Optional account / QoS. Empty on the mi355x cluster (which gates on partition
# alone); required on spur, where every job must name an account and its QoS.
# Both flags are spelled the same in real Slurm, so passing them is portable.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"

# Relocate the shared HuggingFace cache root. nightly-configs.yaml pins
# model_path under /it-share/model_coverage (the mi355x cluster's NFS); a
# cluster that mirrors the same models elsewhere sets MODEL_ROOT and every
# recipe/config path is rewritten, so no recipe or config file needs editing.
MODEL_ROOT="${MODEL_ROOT:-}"
# Root used to RESOLVE the snapshot hash; defaults to MODEL_ROOT. Set it when
# MODEL_ROOT is node-local and therefore unreadable from the driver node.
MODEL_RESOLVE_ROOT="${MODEL_RESOLVE_ROOT:-$MODEL_ROOT}"
# Optional node-local mirror of the model cache, for clusters where shared
# storage cannot serve every rank at once. Resolution stays on the shared root:
# resolve_snapshot runs on the driver node, which has no mirror. Must be assigned
# after MODEL_RESOLVE_ROOT, which keeps the readable root.
MODEL_LOCAL_ROOT="${MODEL_LOCAL_ROOT:-}"
if [[ -n "$MODEL_LOCAL_ROOT" ]]; then
    if [[ -z "$MODEL_RESOLVE_ROOT" ]]; then
        echo "ERROR: MODEL_LOCAL_ROOT is set but MODEL_ROOT/MODEL_RESOLVE_ROOT is empty;" >&2
        echo "       the snapshot hash has to be resolved against a root the driver can read." >&2
        exit 1
    fi
    MODEL_ROOT="$MODEL_LOCAL_ROOT"
fi
MODEL_ROOT_FROM="${MODEL_ROOT_FROM:-/it-share/model_coverage}"
relocate_model_root() {
    local p="$1"
    if [[ -n "$MODEL_ROOT" && "$p" == "$MODEL_ROOT_FROM"/* ]]; then
        printf '%s\n' "$MODEL_ROOT/${p#"$MODEL_ROOT_FROM"/}"
    else
        printf '%s\n' "$p"
    fi
}
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
# Resolve the snapshot against the shared cache first, then relocate. MODEL_ROOT
# may name a node-local copy that does not exist on the node running this
# script; resolve_snapshot would then find no refs/main and pass the bare cache
# dir to the container. Three steps, since the resolve root and the destination
# root can differ: rewrite onto a readable root, resolve refs/main, then rewrite
# onto the root the container will see.
MODEL_PATH="$(MODEL_ROOT="$MODEL_RESOLVE_ROOT" relocate_model_root "$MODEL_PATH")"
MODEL_PATH="$(resolve_snapshot "$MODEL_PATH")" || exit 1
MODEL_PATH="$(MODEL_ROOT_FROM="$MODEL_RESOLVE_ROOT" relocate_model_root "$MODEL_PATH")"

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
if [[ "$CLUSTER" == "spur" ]]; then
  # `rm -rf "$WORKDIR"` is not safe to gate a leg on over NFS: if anything still
  # holds a descriptor inside (an orphaned `tail -F` on bench.log), the server
  # renames the file to .nfsXXXX rather than unlinking it, so rm sees a
  # non-empty directory and exits non-zero. Clear the contents instead, retry,
  # and fall back to moving the directory aside. Housekeeping must not fail a
  # leg.
  for _try in 1 2 3; do
    rm -rf "$WORKDIR"/* "$WORKDIR"/.[!.]* 2>/dev/null || true
    # Anything left is a silent .nfsXXXX handle; give the server a moment.
    [[ -z "$(ls -A "$WORKDIR" 2>/dev/null)" ]] && break
    sleep 5
  done
  mkdir -p "$WORKDIR"
  if [[ -n "$(ls -A "$WORKDIR" 2>/dev/null)" ]]; then
    _stale="$WORKDIR.stale.$$"
    if mv "$WORKDIR" "$_stale" 2>/dev/null; then
      echo "WARN: $WORKDIR would not clear (open handles); moved to $_stale" >&2
    else
      WORKDIR="$WORKDIR.$$"
      echo "WARN: could not clear or move workdir; using fresh $WORKDIR" >&2
    fi
    mkdir -p "$WORKDIR"
  fi
else
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
fi
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
    # See install_checkout_sglang.sh: tar exit 1 is a warning ("file changed as
    # we read it", raised spuriously by NFS attribute caching), 2+ is an error.
    set +e
    tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.git/config' \
        -C "$GITHUB_WORKSPACE" -cf - . | tar -C "$CHECKOUT_STAGE" -xf -
    STAGE_TAR_RC=("${PIPESTATUS[@]}")
    set -e
    if (( STAGE_TAR_RC[0] > 1 || STAGE_TAR_RC[1] != 0 )); then
        echo "ERROR: staging tar failed (create=${STAGE_TAR_RC[0]} extract=${STAGE_TAR_RC[1]})" >&2
        exit 1
    fi
    # Tolerating create-rc=1 above means a warning cannot abort the copy -- but a
    # TRUNCATED copy must still not reach a container, where it only surfaces
    # ~15 minutes later as "invalid checkout mount". Assert the same file the
    # in-container installer checks, right where the copy was made.
    if [[ ! -f "$CHECKOUT_STAGE/python/sglang/version.py" ]]; then
        echo "ERROR: staged checkout is incomplete (no python/sglang/version.py in $CHECKOUT_STAGE);" >&2
        echo "       tar rc were create=${STAGE_TAR_RC[0]} extract=${STAGE_TAR_RC[1]}" >&2
        exit 1
    fi
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
  -e SGLANG_USE_ROCM700A=$ROCM700A
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
# NCCL_IB_HCA must name real devices. "ionic" is the mi355x cluster's spelling;
# on pit2 the HCAs are rdma0..7 (the recipe's $IB) and "ionic" matches nothing,
# which fails TP group init outright with
#   RuntimeError: NCCL error: remote process exited or there was a network error
# from ncclCommInitRank. The wide-EP block below already overrides this for
# EP>8; EP<=8 recipes need it too.
MORI_NCCL_HCA=ionic
[[ "$CLUSTER" == "spur" ]] && MORI_NCCL_HCA="$IB"
MORI_ENV="-e MORI_DISABLE_AUTO_XGMI=1 -e NCCL_IB_HCA=$MORI_NCCL_HCA -e NCCL_IB_GID_INDEX=1 -e NCCL_CROSS_NIC=1"
# Whether NCCL may use IB depends on the role, not on the recipe. A single-node
# engine does all its NCCL traffic intra-node (cross-node KV goes over mori),
# and on pit2 letting it reach for IB hangs ncclCommInitRank. Gating on the
# whole recipe is wrong for 2P1D-EP16 (PN_PER=1, DN_PER=2), whose prefill
# engines are single-node TP8 but whose recipe-wide test left IB on for both
# roles.
#
# Both cases still need a socket interface: NCCL uses one for its bootstrap
# handshake even on an intra-node data path, and pit2 exposes eight tw-eth* /31
# links that auto-selection picks and hangs on. Wide recipes set it in the block
# below; pin it here for any role that block does not cover.
SPUR_PNCCL=""
SPUR_DNCCL=""
if [[ "$CLUSTER" == "spur" ]]; then
    SPUR_SOCK="${DIST_SOCK:-${ADDR_NIC_OVERRIDE:-eno0}}"
    if (( PN_PER == 1 )); then
        SPUR_PNCCL=" -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=$SPUR_SOCK -e GLOO_SOCKET_IFNAME=$SPUR_SOCK"
    fi
    if (( DN_PER == 1 )); then
        SPUR_DNCCL=" -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=$SPUR_SOCK -e GLOO_SOCKET_IFNAME=$SPUR_SOCK"
    fi
fi
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
-e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600 \
-e SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32 -e SGLANG_EAGER_INPUT_NO_COPY=true \
-e MORI_BOOTSTRAP_TIMEOUT=300"
    # The overlap plan stream is a wide-EP perf knob, but it makes
    # EAGLEWorkerV2.verify call attn_backend.update_verify_buffers_to_fill_after_draft,
    # which the DSV4 MLA backend does not implement (NotImplementedError). Only
    # enable it for non-MTP wide-EP; MTP wide-EP uses the plain verify path (as EP8 does).
    if [[ "$MTP_ENABLED" != "1" ]]; then
        MORI_ENV="$MORI_ENV -e SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1"
    fi
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

# Resolve host ionic userspace mounts on each compute node. This keeps server,
# benchmark, and wide-EP containers compatible with the host RDMA ABI.
cat > "$WORKDIR/ionic_mounts.sh" <<'IONIC_EOF'
IONIC_MOUNTS=()
_ionic_provider="/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so"
_ionic_soname="/usr/lib/x86_64-linux-gnu/libionic.so.1"
if [ ! -e "$_ionic_provider" ]; then
    _ionic_provider=$(find /usr/lib/x86_64-linux-gnu -maxdepth 1 \
        -name "libionic.so.*" -print -quit 2>/dev/null)
fi
if [ -n "$_ionic_provider" ] && [ -e "$_ionic_provider" ]; then
    _ionic_real=$(readlink -f "$_ionic_provider" 2>/dev/null)
    _ionic_soname_real=$(readlink -f "$_ionic_soname" 2>/dev/null)
    [ -f "$_ionic_real" ] && IONIC_MOUNTS+=( -v "$_ionic_real:$_ionic_real:ro" )
    [ -f "$_ionic_soname_real" ] && \
        IONIC_MOUNTS+=( -v "$_ionic_soname_real:$_ionic_soname:ro" )
    [ -d /usr/lib/x86_64-linux-gnu/libibverbs ] && \
        IONIC_MOUNTS+=( -v /usr/lib/x86_64-linux-gnu/libibverbs:/usr/lib/x86_64-linux-gnu/libibverbs:ro )
    [ -d /etc/libibverbs.d ] && \
        IONIC_MOUNTS+=( -v /etc/libibverbs.d:/etc/libibverbs.d:ro )
    for _pattern in "libnl-3.so*" "libnl-route-3.so*" "libmnl.so*"; do
        _lib=$(find /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64 \
            -maxdepth 1 -name "$_pattern" -print -quit 2>/dev/null)
        [ -n "$_lib" ] && IONIC_MOUNTS+=( -v "$_lib:$_lib:ro" )
    done
fi
IONIC_EOF

# drive.sh is a quoted heredoc and expands nothing, so its staging check gets the
# resolved paths through a file, as model_flags.sh does. Empty path = no check.
{
    printf 'STAGE_LOCAL_PATH=%q\n' "${MODEL_LOCAL_ROOT:+$MODEL_PATH}"
    printf 'STAGE_SHARED_ROOT=%q\n' "$MODEL_RESOLVE_ROOT"
} > "$WORKDIR/stage_check.sh"

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
        _d="$(MODEL_ROOT="$MODEL_RESOLVE_ROOT" relocate_model_root "$MTP_DRAFT_PATH")"
        _d="$(resolve_snapshot "$_d")" || exit 1
        DRAFT_RESOLVED="$(MODEL_ROOT_FROM="$MODEL_RESOLVE_ROOT" relocate_model_root "$_d")"
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

# /it-share is the mi355x cluster's model NFS. Mount it only where it exists --
# docker would otherwise silently create a root-owned empty /it-share on every
# node of a cluster that keeps its models elsewhere (see MODEL_ROOT).
IT_SHARE_MOUNT="-v /it-share:/it-share:ro "
[[ -d /it-share ]] || IT_SHARE_MOUNT=""
# A cluster that relocates the model cache must bind that root in too, at the
# same path: --model-path is resolved inside the container, and a missing
# directory does not fail as "no such file" -- transformers falls back to
# treating it as a HuggingFace repo id and dies with a confusing
# "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
if [[ -n "$MODEL_ROOT" && "$MODEL_ROOT" != /it-share/* ]]; then
    IT_SHARE_MOUNT="$IT_SHARE_MOUNT-v $MODEL_ROOT:$MODEL_ROOT:ro "
fi
# Slurm hosts each server as a long-lived job step and reaps the container when
# the step ends, so --rm is right there. Spur has no equivalent step (its
# --overlap path is an interactive exec-into-job), so the container runs
# DETACHED under dockerd and drive.sh polls it -- which means it must NOT be
# --rm, or the exit code is gone before the poller can read it. Teardown removes
# it explicitly instead.
DOCKER_LIFECYCLE="--rm"
[[ "$CLUSTER" == "spur" ]] && DOCKER_LIFECYCLE="-d"
# pit2 runs rootless docker, whose spur-authz plugin refuses --privileged
# outright ("denied [P1]"). Spell out what --privileged was actually buying us:
# the device nodes (already listed below), the groups that own kfd/dri, and two
# capabilities -- IPC_LOCK, which RDMA memory registration needs, and SYS_PTRACE
# for profiling. Verified on g11: ibv_devinfo sees all 8 HCAs PORT_ACTIVE and
# --ipc/--network host and --shm-size are all permitted under this set.
PRIV_ARGS="--privileged"
if [[ "$CLUSTER" == "spur" ]]; then
    PRIV_ARGS="--group-add video --group-add render --cap-add IPC_LOCK --cap-add SYS_PTRACE"
fi
DOCKER_COMMON="$DOCKER_LIFECYCLE --network host --ipc host --shm-size 32g $PRIV_ARGS \
--security-opt seccomp=unconfined \
--device /dev/kfd --device /dev/dri --device /dev/infiniband \
${IT_SHARE_MOUNT}-v $HOME:/host_home $CHECKOUT_DOCKER_ARGS"
# Optional extra docker args (e.g. bind-mounting a locally-built lib for
# validation). Empty by default so the docker argv is byte-identical otherwise.
[[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER_COMMON="$DOCKER_COMMON ${EXTRA_DOCKER_ARGS}"

# Spur compute nodes run prolog/epilog hooks that reap any container not tagged
# with the owning job, so an untagged server is SIGKILLed (rc=137) seconds after
# it starts. Single-quoted so the literal $SPUR_JOB_ID survives into the
# generated per-role script and expands on the compute node, where the job
# environment actually defines it.
if [[ "$CLUSTER" == "spur" ]]; then
    DOCKER_COMMON="$DOCKER_COMMON "'--label spur_job_id=$SPUR_JOB_ID'
fi

# Per-role wide-EP docker env (MORI dispatch-token tuning etc.). Empty for EP<=8
# recipes; carries its own leading space so an empty value leaves the docker argv
# byte-identical (no stray double space).
PENV_ARG=""; [[ -n "$PENV" ]] && PENV_ARG=" $PENV"
DENV_ARG=""; [[ -n "$DENV" ]] && DENV_ARG=" $DENV"

# Pin the torch-dist rendezvous port for single-node engines. Without
# --nccl-port, get_free_port() binds port 0, closes the socket, and rank 0
# re-binds the same number later for the TCPStore; anything handed that port in
# between makes rank 0 die with EADDRINUSE. It reproduces rather than being a
# lottery, since the kernel walks its ephemeral range in order.
#
# Only single-node engines need this -- a multi-node engine gets
# --dist-init-addr, which _resolve_dist_init_method() prefers over the port.
# The value is the recipe's own per-role port +1000, which clears the server
# port and stays under the 32768 ephemeral floor. These recipes place at most
# one single-node engine per node, so two engines cannot collide.
#
# Pinning is safe here because --dp-size is only emitted together with
# --enable-dp-attention, and that path gives every DP rank the same nccl port.
# launch_dp_schedulers(), which needs a distinct port per worker, is never
# reached by these recipes.
PNCCL_ARG=""
DNCCL_ARG=""
if [[ "$CLUSTER" == "spur" ]]; then
    PNCCL_ARG=" --nccl-port $((PPORT + 1000))"
    DNCCL_ARG=" --nccl-port $((DPORT + 1000))"
fi
# Per-role NCCL IB/socket policy (spur only; empty under slurm, so the argv
# there is unchanged). Appended after $PENV/$DENV so docker's last-wins applies.
PENV_ARG="$PENV_ARG$SPUR_PNCCL"
DENV_ARG="$DENV_ARG$SPUR_DNCCL"

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
# tar exits 1 for warnings -- notably "file changed as we read it", which NFS
# attribute caching raises spuriously on a tree nothing is writing (seen on the
# pit2 shared-NFS checkout). Only 2+ is a real error, and the extracting tar
# must still succeed outright. Without this, set -o pipefail aborts the install
# on a warning and the server container exits before it ever starts.
set +e
tar --exclude='__pycache__' --exclude='*.pyc' \
  -C "$CHECKOUT_SRC" -cf - . | tar -C "$RUNTIME_CHECKOUT" -xf -
_tar_rc=("${PIPESTATUS[@]}")
set -e
if (( _tar_rc[0] > 1 || _tar_rc[1] != 0 )); then
  echo "[checkout-sglang] ERROR: staging tar failed (create=${_tar_rc[0]} extract=${_tar_rc[1]})" >&2
  exit 1
fi

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
else
  # Single-node engine: no --dist-init-addr, so the rendezvous port is the one
  # sglang would otherwise pick with get_free_port(). Pin it. Empty on slurm.
  DIST_ARGS="$PNCCL_ARG"
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
else
  # Single-node engine: no --dist-init-addr, so the rendezvous port is the one
  # sglang would otherwise pick with get_free_port(). Pin it. Empty on slurm.
  DIST_ARGS="$DNCCL_ARG"
fi
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \
  $DECODE_COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \$DIST_ARGS \
  --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
EOF

  cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
source "$WORKDIR/ionic_mounts.sh"
NODE_RANK="\${1:-0}"; NNODES="\${2:-1}"; DIST_ADDR="\${3:-}"
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
source "$WORKDIR/ionic_mounts.sh"
NODE_RANK="\${1:-0}"; NNODES="\${2:-1}"; DIST_ADDR="\${3:-}"
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
  --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT$PNCCL_ARG
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
  --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT$DNCCL_ARG
EOF

  cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
source "$WORKDIR/ionic_mounts.sh"
docker rm -f mi355x_prefill 2>/dev/null || true
docker run $DOCKER_COMMON "\${IONIC_MOUNTS[@]}" --name mi355x_prefill \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV$PENV_ARG $DSV4_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/prefill_entry.sh
EOF

  cat > "$WORKDIR/decode.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
source "$WORKDIR/ionic_mounts.sh"
docker rm -f mi355x_decode 2>/dev/null || true
docker run $DOCKER_COMMON "\${IONIC_MOUNTS[@]}" --name mi355x_decode \
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
source "$WORKDIR/ionic_mounts.sh"
docker rm -f mi355x_bench 2>/dev/null || true
docker run $DOCKER_COMMON "\${IONIC_MOUNTS[@]}" --name mi355x_bench \
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
# Interface resolve_ip() falls back to when the recipe names no dist NIC (every
# EP<=8 recipe). Empty on Slurm, where drive.sh runs on the login node and never
# resolves a node it is running on; set on Spur, where it runs on an allocated
# node and that node's own name resolves to 127.0.1.1 via Ubuntu's stock
# /etc/hosts -- useless as an address to advertise to the other engine.
ADDR_NIC="${7:-}"
# Scheduler seam (see the CLUSTER block at the top of launch_mi355x.sh). Slurm
# hands out a bracket-compressed hostlist that only `scontrol show hostnames`
# can expand, and srun inherits the allocation from the environment. Spur's
# nodelist env var is already comma-expanded, has no `show hostnames`, and its
# `srun --overlap` requires an explicit --jobid.
CLUSTER="${CLUSTER:-slurm}"
if [[ "$CLUSTER" == "spur" ]]; then
  SRUN=(srun --jobid="$SPUR_JOB_ID" --overlap)
  sched_hostnames() { tr ',' '\n' <<< "$SPUR_JOB_NODELIST" | sed '/^$/d'; }
else
  SRUN=(srun --overlap)
  sched_hostnames() { scontrol show hostnames "$SLURM_JOB_NODELIST"; }
fi
mapfile -t NODES < <(sched_hostnames)
# Dispatch one long-running, log-producing step to a node.
#
# Spur allocates a pty for every --overlap step. drive.sh runs the servers and
# the bench as BACKGROUND subshells, and a background process group writing
# through that pty is stopped by SIGTTOU -- the step then dies (marker rc=137)
# and everything it printed after the first few bytes is lost, which is why a
# failing leg used to leave a log containing nothing but the srun banner. Having
# srun write the file itself keeps the step's output off the pty entirely.
# Slurm has no pty here, so it keeps the original shell redirect.
srun_log() {  # srun_log <logfile> <node> <cmd...>
  local log="$1" node="$2"; shift 2
  # On spur drive.sh runs on one of the allocated nodes, so a dispatch to that
  # node would be srun exec-ing into its own job -- the one spur case that has
  # repeatedly misbehaved. Run it directly instead; the result is identical and
  # it removes a whole class of scheduler interaction.
  if [[ "$CLUSTER" == "spur" && "$node" == "$(hostname)" ]]; then
    "$@" > "$log" 2>&1
    return
  fi
  "${SRUN[@]}" -N1 --nodelist="$node" "$@" > "$log" 2>&1
}

srun_local_or_step() {  # same rule, for short commands whose output we capture
  local node="$1"; shift
  if [[ "$CLUSTER" == "spur" && "$node" == "$(hostname)" ]]; then
    "$@"
    return
  fi
  "${SRUN[@]}" -N1 --nodelist="$node" "$@"
}

# --- spur: containers are detached, liveness comes from dockerd ---------------
# Spur cannot host a long-lived process in a job step, so on spur the servers
# and the bench run as DETACHED containers and these helpers derive the same
# server_exit_*/bench_exit markers the monitor loop already consumes -- from
# real container state rather than from srun's status. Every srun issued here is
# a sub-second foreground command, which is the part of spur that behaves.
declare -A SPUR_SEEN=()   # container -> 1 once observed running, so a container
                          # that has not started yet is not mistaken for a crash
declare -A SPUR_ABSENT=() # container -> consecutive polls that found it gone
SPUR_SERVERS=()           # "role:node:container"
SPUR_POLLS=0
SPUR_ABSENT_LIMIT=3       # polls (10s apart) before "gone" is believed

container_state() {  # <node> <name> -> "running" | "<exit code>" | absent | unknown
  # The obvious `docker inspect | tail -1` cannot be trusted on spur. drive.sh
  # runs on one allocated node, so every poll of a container on the OTHER node
  # is an `srun --overlap` step, and those intermittently fail to spawn
  # ("step command failed to spawn: Permission denied (os error 13)"). A failed
  # step and a genuinely absent container are the same observation -- nonzero
  # exit, empty stdout -- so one flaky poll was enough to declare a healthy
  # server dead. Observed on dsv4flash-fp8-1k1k-1p1d: prefill marked
  # "rc=missing" while `docker ps` on that node still showed it Up.
  #
  # The __ok__ sentinel is emitted only if the remote command actually ran, so a
  # dispatch failure is reported as "unknown" and the caller ignores it rather
  # than counting it as evidence.
  local out
  out="$(srun_local_or_step "$1" bash -c \
      "docker inspect -f '{{if .State.Running}}running{{else}}{{.State.ExitCode}}{{end}}' $2 2>/dev/null || echo absent; echo __ok__" \
      2>/dev/null | tr -d '\r')"
  [[ "$out" == *__ok__* ]] || { echo unknown; return; }
  grep -vxE '__ok__|' <<< "$out" | tail -1
}

container_log() {  # <node> <name> <logfile>
  # Fetch to a temp file and never let a failed fetch shrink the log: the same
  # flaky step that produced the false "missing" also truncated the 5-line
  # prefill log to 0 bytes, destroying the only evidence of what happened.
  local tmp="$3.fetch"
  srun_local_or_step "$1" docker logs "$2" > "$tmp" 2>/dev/null || true
  # Publish in place (truncate + rewrite the SAME inode), never by renaming over
  # the target. drive.sh keeps a `tail -F` on bench.log, and swapping the inode
  # every 10s made tail reopen and re-emit the whole file each poll -- the
  # driver log reached 177 MB and legs took 2 h instead of 36 min.
  if [[ -s "$tmp" || ! -f "$3" ]]; then cat "$tmp" > "$3"; fi
  rm -f "$tmp"
}

spur_rm() {  # <node> <container...> -- remove, retrying past a flaky dispatch
  local n="$1"; shift
  local i
  for i in 1 2 3; do
    srun_local_or_step "$n" bash -c "docker rm -f $* >/dev/null 2>&1; echo __ok__" \
      2>/dev/null | grep -q __ok__ && return 0
    sleep 3
  done
  return 1
}

# Refresh one container's log and write its marker once it is no longer running.
spur_track() {  # <node> <name> <logfile> <markerfile> <marker prefix>
  local node="$1" name="$2" log="$3" marker="$4" label="$5" st
  container_log "$node" "$name" "$log"
  st="$(container_state "$node" "$name")"
  if [[ "$st" == "running" ]]; then
    SPUR_SEEN["$name"]=1
    SPUR_ABSENT["$name"]=0
    return
  fi
  # A step that never ran says nothing about the container.
  [[ "$st" == "unknown" ]] && return
  if [[ "$st" == "absent" || -z "$st" ]]; then
    # Not created yet, or one poll that came back empty. Only a container that
    # is absent on SPUR_ABSENT_LIMIT polls in a row is treated as a crash, so
    # neither a slow `docker run -d` nor a single flaky poll ends the leg.
    SPUR_ABSENT["$name"]=$(( ${SPUR_ABSENT[$name]:-0} + 1 ))
    if [[ -z "${SPUR_SEEN[$name]:-}" && "$SPUR_POLLS" -lt 6 ]]; then return; fi
    [[ "${SPUR_ABSENT[$name]}" -lt "$SPUR_ABSENT_LIMIT" ]] && return
    st="missing"
  fi
  echo "$label rc=$st" > "$marker"
}

spur_refresh_markers() {
  local entry role node cname st
  SPUR_POLLS=$((SPUR_POLLS + 1))
  for entry in "${SPUR_SERVERS[@]}"; do
    IFS=: read -r role node cname <<< "$entry"
    [[ -f "$WORKDIR/server_exit_${role}_${node}" ]] && continue
    spur_track "$node" "$cname" "$WORKDIR/${role}_${node}.log" \
               "$WORKDIR/server_exit_${role}_${node}" "$role@$node"
  done
  # The bench marker holds a bare exit code, not the "role@node rc=" form.
  container_log "$PNODE" mi355x_bench "$WORKDIR/bench.log"
  st="$(container_state "$PNODE" mi355x_bench)"
  if [[ "$st" == "running" ]]; then
    SPUR_SEEN[mi355x_bench]=1
    SPUR_ABSENT[mi355x_bench]=0
  elif [[ "$st" == "unknown" ]]; then
    :
  elif [[ "$st" == "absent" || -z "$st" ]]; then
    SPUR_ABSENT[mi355x_bench]=$(( ${SPUR_ABSENT[mi355x_bench]:-0} + 1 ))
    if [[ -n "${SPUR_SEEN[mi355x_bench]:-}" \
          && "${SPUR_ABSENT[mi355x_bench]}" -ge "$SPUR_ABSENT_LIMIT" ]]; then
      echo 1 > "$WORKDIR/bench_exit"
    fi
  else
    echo "$st" > "$WORKDIR/bench_exit"
  fi
}
# Resolve a node's IP. When a cross-node dist NIC is named (wide engines set
# $DIST_NIC to the recipe's dist_socket_ifname), read the address off that
# interface on the node itself -- the multi-node dist-init-addr must be a
# LOCALLY-BINDABLE IP (torch-dist + tokenizer ZMQ bind to it), and a node's
# forward DNS can be stale/point at a non-local mgmt alias (observed: a decode
# root whose hostname resolved to an unpingable IP, ZMQ bind => "Cannot assign
# requested address"). Fall back to Slurm's NodeAddr, then DNS.
resolve_ip() {
  local n="$1" ip="" addr="" field nic="${DIST_NIC:-$ADDR_NIC}"
  if [[ -n "$nic" ]]; then
    # Go through srun_local_or_step, not a bare srun: on spur drive.sh runs on an
    # allocated node, and resolving its OWN node through `srun --overlap` is the
    # dispatch that has repeatedly misbehaved. Here it did not fail, it HUNG --
    # a leg sat 30 min on `srun ... ip -4 -o addr show eno0` for the node it was
    # already running on, before any container was started. The helper runs it
    # directly in that case. The timeout bounds the genuinely remote call so a
    # wedged step costs 60s and a DNS fallback instead of the whole leg.
    if [[ "$CLUSTER" == "spur" ]]; then
      if [[ "$n" == "$(hostname)" ]]; then
        ip=$(command ip -4 -o addr show "$nic" 2>/dev/null \
               | awk '{print $4}' | cut -d/ -f1 | head -1)
      else
        ip=$(timeout 60 "${SRUN[@]}" -N1 --nodelist="$n" ip -4 -o addr show "$nic" 2>/dev/null \
               | awk '{print $4}' | cut -d/ -f1 | head -1)
      fi
    else
      ip=$("${SRUN[@]}" -N1 --nodelist="$n" ip -4 -o addr show "$nic" 2>/dev/null \
             | awk '{print $4}' | cut -d/ -f1 | head -1)
    fi
  fi
  if [[ -z "$ip" ]]; then
    # Spur's `scontrol show node` does not emit Slurm's NodeAddr= field, so the
    # loop below would just fall through; skip it there and go straight to DNS.
    if [[ "$CLUSTER" != "spur" ]]; then
      for field in $(scontrol show node "$n" -o); do
        case "$field" in
          NodeAddr=*) addr="${field#NodeAddr=}"; break ;;
        esac
      done
    fi
    [[ -n "$addr" ]] || addr="$n"
    read -r ip _ < <(getent ahostsv4 "$addr")
  fi
  # A loopback or empty answer is never a usable advertised address. Fail here
  # rather than let a server bind 127.x (or nothing) and have the peer engine
  # time out connecting to it with no obvious cause. Callers must use
  # `x=$(resolve_ip n) || exit 1` -- the status is visible at the assignment,
  # but an `exit` inside the substitution would only leave the subshell.
  if [[ -z "$ip" ]]; then
    echo "[drive] ERROR: no address found for $n" >&2
    return 1
  fi
  if [[ "$ip" == 127.* ]]; then
    echo "[drive] ERROR: $n resolved to loopback $ip; set ADDR_NIC to a real interface" >&2
    return 1
  fi
  printf '%s\n' "$ip"
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
PIP=$(resolve_ip "$PNODE") || exit 1
DIP=$(resolve_ip "$DNODE") || exit 1
echo "[drive] prefill nodes: ${PNODES[*]} ; decode nodes: ${DNODES[*]}"
echo "[drive] bench targets prefill=$PNODE($PIP) decode=$DNODE($DIP)"
# The local copy is staged out of band and does not survive a reboot. Without
# this the container falls back to the shared root and the only symptom is a
# health-wait timeout 50 minutes later.
source "$WORKDIR/stage_check.sh"
if [[ -n "$STAGE_LOCAL_PATH" ]]; then
  _stage_bad=0
  for n in "${NODES[@]}"; do
    if ! srun_local_or_step "$n" test -d "$STAGE_LOCAL_PATH" >/dev/null 2>&1; then
      echo "ERROR: $n is missing the node-local snapshot $STAGE_LOCAL_PATH" >&2
      _stage_bad=1
    fi
  done
  if (( _stage_bad )); then
    echo "ERROR: re-stage the model onto every allocated node, or unset MODEL_LOCAL_ROOT" >&2
    echo "       to read from $STAGE_SHARED_ROOT (slow: large checkpoints will time out)." >&2
    exit 1
  fi
  echo "[drive] node-local snapshot present on all ${#NODES[@]} nodes"
fi
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
if [[ "$CLUSTER" == "spur" ]]; then
  # Detached containers are not --rm, and a container whose docker client was
  # killed can survive as an exited husk. Clear all three names on every
  # allocated node first: otherwise `docker run` fails with a name conflict and,
  # worse, the poller reads the STALE container's exit code and reports a crash
  # that never happened.
  for n in "${NODES[@]}"; do
    srun_local_or_step "$n" \
      docker rm -f mi355x_prefill mi355x_decode mi355x_bench >/dev/null 2>&1 || true
  done
fi
# Launch PW prefill engines; each spans PN_PER nodes as its own torch-dist group
# (engine node0 = dist-init addr; node_rank = position WITHIN the engine, so an
# engine's ranks are 0..PN_PER-1, not a global index). PN_PER=1 => NNODES=1 in
# the entry script => dist args dropped => byte-identical single-node launch.
# Collect each engine's node0 IP into PCSV for the router's prefill fan-out.
PCSV=""
for ((k=0; k<PW; k++)); do
  e0="${PNODES[k*PN_PER]}"
  eip=$(resolve_ip "$e0") || exit 1
  PCSV="${PCSV:+$PCSV,}$eip"
  for ((j=0; j<PN_PER; j++)); do
    n="${PNODES[k*PN_PER + j]}"
    if [[ "$CLUSTER" == "spur" ]]; then
      # Returns as soon as the container is detached; liveness is polled later.
      srun_log "$WORKDIR/prefill_start_$n.log" "$n" \
        bash "$WORKDIR/prefill.sh" "$j" "$PN_PER" "$eip" || true
      SPUR_SERVERS+=("prefill:$n:mi355x_prefill")
    else
    ( srun_log "$WORKDIR/prefill_$n.log" "$n" bash "$WORKDIR/prefill.sh" "$j" "$PN_PER" "$eip"
      echo "prefill@$n rc=$?" > "$WORKDIR/server_exit_prefill_$n" ) &
    fi
  done
done
for ((k=0; k<DW; k++)); do
  e0="${DNODES[k*DN_PER]}"
  eip=$(resolve_ip "$e0") || exit 1
  for ((j=0; j<DN_PER; j++)); do
    n="${DNODES[k*DN_PER + j]}"
    if [[ "$CLUSTER" == "spur" ]]; then
      srun_log "$WORKDIR/decode_start_$n.log" "$n" \
        bash "$WORKDIR/decode.sh" "$j" "$DN_PER" "$eip" || true
      SPUR_SERVERS+=("decode:$n:mi355x_decode")
    else
    ( srun_log "$WORKDIR/decode_$n.log" "$n" bash "$WORKDIR/decode.sh" "$j" "$DN_PER" "$eip"
      echo "decode@$n rc=$?" > "$WORKDIR/server_exit_decode_$n" ) &
    fi
  done
done
echo "[drive] prefill engine endpoints (fan-out): $PCSV"
sleep 5
# Bench in the background with its own marker, so the wait loop is purely file
# based: finish when bench writes its marker, abort if any server marker shows up
# first (a server died before the sweep completed).
BENCH_BG=""
if [[ "$CLUSTER" == "spur" ]]; then
  srun_log "$WORKDIR/bench_start.log" "$PNODE" \
    bash "$WORKDIR/bench.sh" "$PIP" "$DIP" "$PCSV" || true
else
( srun_log "$WORKDIR/bench.log" "$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" "$PCSV"
  echo $? > "$WORKDIR/bench_exit" ) &
BENCH_BG=$!
fi
# Stream bench output live and poll the markers with xtrace OFF, so the console
# shows clean benchmark/accuracy output instead of a compgen/sleep trace every
# 10s. (Mirrors NVIDIA's launch_gb200.sh, which set +x around its log stream.)
touch "$WORKDIR/bench.log"
tail -n +1 -F "$WORKDIR/bench.log" 2>/dev/null &
TAIL_PID=$!
set +x
RC=0
while [[ ! -f "$WORKDIR/bench_exit" ]]; do
  # On spur the markers are produced here, from container state, instead of by a
  # backgrounded srun. Everything below is unchanged.
  [[ "$CLUSTER" == "spur" ]] && spur_refresh_markers
  if compgen -G "$WORKDIR/server_exit_*" > /dev/null; then
    echo "[drive] ERROR: a server exited early before bench finished:"
    cat "$WORKDIR"/server_exit_* || true
    [[ -n "$BENCH_BG" ]] && kill "$BENCH_BG" 2>/dev/null || true
    RC=1
    break
  fi
  sleep 10
done
set -x
kill "$TAIL_PID" 2>/dev/null || true
[[ "$RC" -eq 0 ]] && RC=$(cat "$WORKDIR/bench_exit" 2>/dev/null || echo 1)
echo "[drive] bench finished (rc=$RC), tearing down"
if [[ "$CLUSTER" == "spur" ]]; then
  # Detached containers are not --rm, so kill is not enough -- remove them, and
  # the bench container too, or the next leg on this node inherits them.
  # Retry the removals: a step that fails to spawn used to leave the containers
  # running, and the next leg on that node then inherited engines holding all
  # eight GPUs.
  for n in "${PNODES[@]}"; do spur_rm "$n" mi355x_prefill || true; done
  for n in "${DNODES[@]}"; do spur_rm "$n" mi355x_decode  || true; done
  spur_rm "$PNODE" mi355x_bench || true
else
for n in "${PNODES[@]}"; do "${SRUN[@]}" -N1 --nodelist="$n" docker kill mi355x_prefill >/dev/null 2>&1 || true; done
for n in "${DNODES[@]}"; do "${SRUN[@]}" -N1 --nodelist="$n" docker kill mi355x_decode  >/dev/null 2>&1 || true; done
fi
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

# Optional comma-separated nodes to keep the scheduler off.
EXCLUDE_ARG=()
[[ -n "${SLURM_EXCLUDE:-}" ]] && EXCLUDE_ARG=(--exclude="$SLURM_EXCLUDE")

# Optional account / QoS (spur requires both; mi355x sets neither).
ACCT_ARG=()
[[ -n "$SLURM_ACCOUNT" ]] && ACCT_ARG+=(-A "$SLURM_ACCOUNT")
[[ -n "$SLURM_QOS" ]] && ACCT_ARG+=(-q "$SLURM_QOS")

# Nodes = sum over engines of nodes-per-engine. EP<=8 (PN_PER=DN_PER=1) gives the
# original PW+DW (1P1D -> 2 nodes); wide EP16 1P1D gives 2+2 = 4 nodes.
TOTAL_NODES=$(( PW * PN_PER + DW * DN_PER ))

# Keep g20 out of a wide engine's root position, where its slower MORI
# bootstrap can miss the worker-connect timeout.
if [[ "$MATRIX_CONFIG_NAME" == *-2p1d-ep16* ]]; then
    export SLURM_DIST_TAIL="${SLURM_DIST_TAIL:-mia1-p01-g20}"
fi

# Name the allocation <RUNNER_NAME>-<GITHUB_RUN_ID>-<config> so the workflow's
# cleanup steps can scancel precisely instead of a blanket `squeue --me` that
# would kill a concurrent matrix leg. RUNNER_NAME alone is not assumed unique;
# GITHUB_RUN_ID + config make the name unique per matrix leg regardless.
JOB_NAME="mi355x-ci-${RUNNER_NAME:-norunner}-${GITHUB_RUN_ID:-0}-${MATRIX_CONFIG_NAME}"

# Address-resolution fallback NIC for drive.sh (see ADDR_NIC there). Only spur
# needs one, because only there does drive.sh run on an allocated node.
ADDR_NIC=""
[[ "$CLUSTER" == "spur" ]] && ADDR_NIC="${ADDR_NIC_OVERRIDE:-eno0}"

set +e
if [[ "$CLUSTER" == "spur" ]]; then
    # Spur's salloc takes no trailing command, so drive.sh is submitted as a
    # batch job instead. That runs it on the first allocated node rather than
    # the login node, which is fine: $WORKDIR and $GITHUB_WORKSPACE are on
    # shared NFS, and drive.sh only ever reaches the other nodes through srun.
    # We stream the batch output back live and take the exit code from a file
    # drive_batch.sh writes, since sbatch returns as soon as the job is queued.
    cat > "$WORKDIR/drive_batch.sh" <<EOF
#!/bin/bash
# Set explicitly rather than relying on sbatch to propagate the submit
# environment; drive.sh needs it to pick the spur scheduler seam.
export CLUSTER=spur
export PATH="$PATH"
# Slurm runs a batch script once, on the first allocated node. Spur runs it as
# one task PER NODE (scontrol reports NumTasks=NumNodes), so without this guard
# every node gets its own drive.sh: the copies race, and each one's pre-launch
# container cleanup kills the servers the other just started. Only rank 0
# drives; the rest exit immediately and simply hold their node in the
# allocation, which is all drive.sh needs them for.
# Elect the driver with an atomic mkdir on shared NFS rather than by rank:
# spur does not consistently place the batch script on rank 0 (it has been
# observed running only on SPUR_NODEID=1), so a rank test would sometimes elect
# nobody and the leg would never start.
if ! mkdir "$WORKDIR/driver.lock" 2>/dev/null; then
  # Must BLOCK, not exit: spur starts tearing the job down as soon as one of its
  # tasks finishes, and a job in Completing rejects further steps
  # ("CreateJobStep failed: job N is not running") -- which killed the decode
  # launch. Holding here keeps the node in the allocation until rank 0 is done.
  echo "[drive_batch] \$(hostname) standing by; another task drives" >> "$WORKDIR/standby.log"
  while [[ ! -f "$WORKDIR/drive_exit" ]]; do sleep 10; done
  exit 0
fi
echo "[drive_batch] \$(hostname) rank \${SPUR_NODEID:-?} elected driver"
# Send the driver's output to a private per-node file, not to the shared
# --output. All tasks of the job write sbatch.out concurrently, so it loses
# records: a leg that failed in the monitor loop left an sbatch.out that simply
# stopped mid-run, with the teardown lines never visible.
#
# A plain redirect, deliberately NOT `| tee`. With a pipeline, drive_batch waits
# for the whole pipeline, and that does not finish when drive.sh does -- the
# backgrounded `tail -F` on bench.log inherits drive.sh's stdout and holds the
# pipe's write end open. drive.sh exited 0 and drive_exit was still unwritten
# 11 minutes later, so the standby tasks kept the allocation alive and the leg
# looked hung after it had actually passed.
bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW" "$PN_PER" "$DN_PER" "$DIST_SOCK" "$ADDR_NIC" \
  > "$WORKDIR/drive_\$(hostname).log" 2>&1
echo \$? > "$WORKDIR/drive_exit"
EOF
    chmod +x "$WORKDIR/drive_batch.sh"
    rm -f "$WORKDIR/drive_exit"
    SBATCH_OUT="$WORKDIR/sbatch.out"
    : > "$SBATCH_OUT"
    # Node count is what matters here; do NOT pass -n1 to make the task count
    # match. Spur sizes the allocation from the task count, so -n1 collapses a
    # 2-node request to a single node and the decode engine has nowhere to run.
    # Uniqueness of the driver is handled in drive_batch.sh by an atomic lock,
    # not by the task count.
    # Spur's controller is a Raft cluster, so submission can fail for reasons
    # that have nothing to do with the request: during a leader election it
    # answers `The service is currently unavailable ... "not the Raft leader"`.
    # That is transient and clears in seconds, but an unretried submit turns it
    # into a failed leg. Worse, the message is multi-line, so the ${VAR##* }
    # job-id parse below extracted `leader"` from it and the leg was reported as
    # a benchmark failure with an empty bench.log -- indistinguishable, at a
    # glance, from a model bug. Observed on glm52-fp4-1k1k-2p1d-ep16 2026-09-17.
    # Retry only on that signature; a genuinely bad sbatch request must still
    # fail immediately rather than being retried six times.
    SPUR_LEG_JOB_ID=""
    for _attempt in 1 2 3 4 5 6; do
        SBATCH_MSG=$(sbatch -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" \
            "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" "${ACCT_ARG[@]}" \
            --job-name "$JOB_NAME" -t "$TIME_LIMIT" \
            --output "$SBATCH_OUT" --error "$SBATCH_OUT" \
            "$WORKDIR/drive_batch.sh" 2>&1)
        echo "$SBATCH_MSG"
        SPUR_LEG_JOB_ID="${SBATCH_MSG##* }"
        [[ "$SPUR_LEG_JOB_ID" =~ ^[0-9]+$ ]] && break
        if [[ "$SBATCH_MSG" == *"not the Raft leader"* \
           || "$SBATCH_MSG" == *"service is currently unavailable"* ]]; then
            echo "[launch] spur controller unavailable (attempt $_attempt/6); retrying in 20s" >&2
            SPUR_LEG_JOB_ID=""
            sleep 20
            continue
        fi
        break
    done
    if [[ ! "$SPUR_LEG_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse a job id out of: $SBATCH_MSG" >&2
        SALLOC_RC=1
    else
        echo "[launch] spur job $SPUR_LEG_JOB_ID submitted; streaming $SBATCH_OUT"
        tail -F "$SBATCH_OUT" 2>/dev/null &
        SPUR_TAIL_PID=$!
        # drive_batch.sh writes drive_exit last, so that file is the primary
        # completion signal; the squeue check catches a job that died without
        # ever running it (node failure, scheduler kill, time limit).
        while :; do
            [[ -f "$WORKDIR/drive_exit" ]] && break
            if ! squeue -h -o "%i" 2>/dev/null | grep -qx "$SPUR_LEG_JOB_ID"; then
                # The job has left the queue. drive_exit is written on a compute
                # node and read here on the login node, so NFS close-to-open
                # visibility can delay it well past a single short sleep. A five
                # second grace produced three false reds on 2026-09-16, each on a
                # leg that had finished every concurrency point and passed the
                # accuracy gate; one of them was the whole GLM EP16 row.
                for _ in $(seq 1 30); do
                    [[ -f "$WORKDIR/drive_exit" ]] && break
                    sleep 3
                done
                if [[ ! -f "$WORKDIR/drive_exit" ]]; then
                    # Still not visible. Do not invent a failure we did not
                    # observe -- decide from what the run actually produced.
                    # bench_exit is written earlier and is usually visible by now;
                    # when even that is missing (seen on glm52-fp4-1k1k-2p1d-ep16,
                    # which had all seven results at 0.945), a passed accuracy
                    # gate with results on disk is the stronger evidence.
                    #
                    # "Some results plus a passed gate" is not enough. A leg
                    # whose decode ran at ~9 tok/s got through conc=1, 8 and 16
                    # of seven points before the job ended, and this branch
                    # called it green: the gate had passed and raw files
                    # existed. Require every expected concurrency point, so a
                    # sweep that stopped early is red rather than a green row
                    # standing on someone else's numbers.
                    WANT=0; HAVE=0
                    for _c in ${CONCS//,/ }; do
                        WANT=$((WANT + 1))
                        [[ -f "$WORKDIR/raw_conc${_c}.json" ]] && HAVE=$((HAVE + 1))
                    done
                    if [[ "$(cat "$WORKDIR/bench_exit" 2>/dev/null)" == "0" ]]; then
                        echo 0 > "$WORKDIR/drive_exit"
                    elif [[ "$HAVE" -eq "$WANT" ]] \
                         && awk '/\[gsm8k\] accuracy=/{
                                   match($0,/accuracy=([0-9.]+)/,a)
                                   match($0,/threshold=([0-9.]+)/,t)
                                   if (a[1]+0 >= t[1]+0) ok=1
                                 } END{ exit ok?0:1 }' \
                                "$WORKDIR/bench.log" 2>/dev/null; then
                        echo 0 > "$WORKDIR/drive_exit"
                    else
                        echo "[launch] incomplete sweep: $HAVE/$WANT concurrency points" >&2
                        echo 1 > "$WORKDIR/drive_exit"
                    fi
                fi
                break
            fi
            sleep 10
        done
        kill "$SPUR_TAIL_PID" 2>/dev/null || true
        SALLOC_RC=$(cat "$WORKDIR/drive_exit" 2>/dev/null || echo 1)
        # Release the allocation. On slurm the job ends when salloc's command
        # returns; on spur drive.sh is an sbatch job, and the loop above exits as
        # soon as drive_exit appears -- while the batch job, and every container
        # it started, is still running. Nothing else ever cancels it, so the
        # allocation is held until the time limit. The next leg then cannot get
        # nodes and dies with `JobLaunchFailure (dispatch confirmation failed
        # (2/4 confirmed))`, which reads as an unrelated infrastructure fault and
        # cascades through every leg after it. Observed 2026-09-17: jobs
        # 1774/1775 held all four nodes long after their legs had been recorded
        # rc=1. Unconditional, because reaching here means this leg is done with
        # its nodes either way.
        if squeue -h -o "%i" 2>/dev/null | grep -qx "$SPUR_LEG_JOB_ID"; then
            echo "[launch] releasing spur job $SPUR_LEG_JOB_ID"
            scancel "$SPUR_LEG_JOB_ID" 2>/dev/null || true
        fi
    fi
else
    salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \
        "${ACCT_ARG[@]}" \
        --job-name "$JOB_NAME" -t "$TIME_LIMIT" \
        bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW" "$PN_PER" "$DN_PER" "$DIST_SOCK"
    SALLOC_RC=$?
fi
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
# Clear this leg's results from any previous run before writing new ones.
# The filenames are a pure function of the config, so a run that produces
# fewer concurrency points than the last one leaves the missing slots filled
# by the old run's files and the published table silently mixes two runs.
# Seen on kimik26-mxfp4-1k1k-2p1d-ep16-mxfp4: 2 points measured, 7 published,
# 5 of them a day old. Deleting first means a short run looks short.
rm -f "$GITHUB_WORKSPACE/${RESULT_FILENAME}_${MATRIX_CONFIG_NAME}_conc"*"_gpus_"*".json"

PROCESSED=0
for C in ${CONCS//,/ }; do
    RAW="$WORKDIR/raw_conc${C}.json"
    # raw_conc*.json is written by the bench container on a compute node; this
    # loop runs on the driver. NFS close-to-open means the last file written is
    # routinely not visible here yet, and skipping it outright silently dropped
    # one or two concurrency points from four legs of a 34-leg sweep -- the runs
    # were fine, only the published results were short. Wait a bounded 60s.
    for _ in $(seq 1 20); do
        [[ -f "$RAW" ]] && break
        sleep 3
    done
    [[ -f "$RAW" ]] || { echo "WARN: missing $RAW (not visible after 60s)"; continue; }
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

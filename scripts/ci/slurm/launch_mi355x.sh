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
#   MOONCAKE_COMMIT_OVERRIDE
#                      - optional debug override. When set, prefill/decode
#                        containers rebuild and install Mooncake at this commit
#                        before starting SGLang.
#   MOONCAKE_REPO      - optional Mooncake repo for the override; defaults to
#                        docker/rocm.Dockerfile's MOONCAKE_REPO.
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
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))
rt = r["runtime"]; b = r["backend"]["sglang_config"]; bn = r["bench"]
res = r.get("resources", {})
def emit(k, v): print(f"{k}={v}")
emit("IMAGE", rt["image"])
# Attention backend: single (`attention_backend`) or split
# (`prefill_attention_backend`/`decode_attention_backend`). Empty when absent so
# the flag is dropped for a model that omits it.
emit("ATTN", rt.get("attention_backend", ""))
emit("PATTN", rt.get("prefill_attention_backend", ""))
emit("DATTN", rt.get("decode_attention_backend", ""))
emit("IB", rt["ib_devices"])
emit("PPORT", rt["prefill_port"])
emit("DPORT", rt["decode_port"])
emit("PBOOT", rt["prefill_bootstrap_port"])
emit("DBOOT", rt["decode_bootstrap_port"])
emit("LBPORT", rt["lb_port"])
emit("MEMFRAC", rt["mem_fraction_static"])
emit("PAGE", rt["page_size"])
emit("MAXREQ", rt["max_running_requests"])
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
echo "recipe: image=$IMAGE attn=${ATTN:-$PATTN/$DATTN} ib=$IB ptp=$PTP dtp=$DTP concs=$CONCS isl=$ISL osl=$OSL"

MOONCAKE_ENV_STR=""
if [[ -n "${MOONCAKE_COMMIT_OVERRIDE:-}" ]]; then
    MOONCAKE_REPO="${MOONCAKE_REPO:-$(grep -E '^[[:space:]]*ARG[[:space:]]+MOONCAKE_REPO=' docker/rocm.Dockerfile | head -n1 | sed 's/.*MOONCAKE_REPO="\([^"]*\)".*/\1/' || true)}"
    MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
    MOONCAKE_ENV_STR="-e MOONCAKE_COMMIT_OVERRIDE=$MOONCAKE_COMMIT_OVERRIDE -e MOONCAKE_REPO=$MOONCAKE_REPO"
    echo "Mooncake override enabled: repo=$MOONCAKE_REPO commit=$MOONCAKE_COMMIT_OVERRIDE"
fi

# ---------------------------------------------------------------------------
# Shared NFS scratch (visible to login node + compute nodes). Raw bench output
# lands here; the launcher normalizes it into GITHUB_WORKSPACE afterwards.
# ---------------------------------------------------------------------------
WORKDIR="$HOME/.mi355x_ci/${MATRIX_CONFIG_NAME}"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
  -e SGLANG_USE_ROCM700A=1 -e SGLANG_OPT_USE_FUSED_COMPRESS=true
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

cat > "$WORKDIR/install_mooncake.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

if [[ -z "${MOONCAKE_COMMIT_OVERRIDE:-}" ]]; then
  exit 0
fi

MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MOONCAKE_DIR="/sgl-workspace/Mooncake"
MOONCAKE_BUILD_JOBS="${MOONCAKE_BUILD_JOBS:-$(nproc)}"
MOONCAKE_CMAKE_FLAGS="${MOONCAKE_CMAKE_FLAGS:--DUSE_HIP=ON -DUSE_ETCD=ON -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=ON -DBUILD_UNIT_TESTS=OFF}"

echo "[Mooncake] Reinstalling commit ${MOONCAKE_COMMIT_OVERRIDE} from ${MOONCAKE_REPO}"

if [[ ! -d "${MOONCAKE_DIR}/.git" ]]; then
  rm -rf "${MOONCAKE_DIR}"
  git clone "${MOONCAKE_REPO}" "${MOONCAKE_DIR}"
fi

cd "${MOONCAKE_DIR}"
git fetch --all --tags || git fetch "${MOONCAKE_REPO}" "${MOONCAKE_COMMIT_OVERRIDE}" || true
if ! git cat-file -e "${MOONCAKE_COMMIT_OVERRIDE}^{commit}" 2>/dev/null; then
  echo "[Mooncake] ERROR: commit not found after fetch: ${MOONCAKE_COMMIT_OVERRIDE}" >&2
  exit 1
fi

git checkout -f "${MOONCAKE_COMMIT_OVERRIDE}"
git submodule update --init --recursive
rm -rf build
mkdir -p build
cd build
cmake .. ${MOONCAKE_CMAKE_FLAGS}
make -j "${MOONCAKE_BUILD_JOBS}"
make install
ldconfig

python3 - <<'PY'
import mooncake
import os

print("[Mooncake] installed package:", os.path.dirname(mooncake.__file__))
PY
EOF

# Optional topology / speculative-decode flags driven by the recipe. Base recipes
# (EP1/DP1, no mtp) leave EXTRA_FLAGS empty, preserving prior behavior exactly.
EXTRA_FLAGS=""
(( PDP > 1 )) && EXTRA_FLAGS="$EXTRA_FLAGS --enable-dp-attention --dp-size $PDP"
(( PEP > 1 )) && EXTRA_FLAGS="$EXTRA_FLAGS --ep-size $PEP"
if [[ "$MTP_ENABLED" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --speculative-algorithm $MTP_ALGO \
--speculative-num-steps $MTP_STEPS --speculative-eagle-topk $MTP_TOPK \
--speculative-num-draft-tokens $MTP_DRAFT"
    # EAGLE3 (and other draft-model algos) need an external draft checkpoint;
    # built-in EAGLE (DSV4) omits draft_model_path and this stays unset.
    if [[ -n "$MTP_DRAFT_PATH" ]]; then
        DRAFT_RESOLVED="$(resolve_snapshot "$MTP_DRAFT_PATH")" || exit 1
        EXTRA_FLAGS="$EXTRA_FLAGS --speculative-draft-model-path $DRAFT_RESOLVED"
    fi
fi
echo "extra flags: ${EXTRA_FLAGS:-<none>} (pep=$PEP pdp=$PDP mtp=$MTP_ENABLED algo=$MTP_ALGO)"

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
    COMMON_FLAGS="--trust-remote-code --tp $PTP --disable-radix-cache \
$ATTN_FLAGS --max-running-requests $MAXREQ --page-size $PAGE \
--mem-fraction-static $MEMFRAC$SWA_FLAG \
--chunked-prefill-size $CHUNK \
--disaggregation-transfer-backend mori --disaggregation-ib-device $IB$EXTRA_FLAGS"
else
    # DSV4 path: byte-identical to the pre-Kimi launcher.
    COMMON_FLAGS="--trust-remote-code --tp $PTP --disable-radix-cache \
--attention-backend $ATTN --max-running-requests $MAXREQ --page-size $PAGE \
--mem-fraction-static $MEMFRAC --swa-full-tokens-ratio $SWA \
--chunked-prefill-size $CHUNK --disable-shared-experts-fusion \
--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
--disaggregation-transfer-backend mori --disaggregation-ib-device $IB$EXTRA_FLAGS"
fi

DOCKER_COMMON="--rm --network host --ipc host --shm-size 32g --privileged \
--security-opt seccomp=unconfined \
--device /dev/kfd --device /dev/dri --device /dev/infiniband \
-v /it-share:/it-share:ro -v $HOME:/host_home"

# ---------------------------------------------------------------------------
# Write per-role scripts that srun dispatches to each compute node.
# ---------------------------------------------------------------------------
# These are UNQUOTED `<<EOF` heredocs, so $MORI_ENV/$DSV4_ENV_STR/$COMMON_FLAGS/
# $IMAGE/$MODEL_PATH expand now (at generation). model_flags.sh is sourced at
# runtime for the model's env/server arrays, so the `${MODEL_ENV_ARGS[@]}` /
# `${MODEL_SERVER_ARGS[@]}` refs are backslash-escaped to survive into the script
# and expand after `source`. For DSV4 those arrays are empty and $DSV4_ENV_STR is
# set, so the resulting docker argv is byte-identical to the pre-Kimi launcher.
cat > "$WORKDIR/prefill_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_mooncake.sh"
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \
  $COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \
  --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
EOF

cat > "$WORKDIR/decode_entry.sh" <<EOF
#!/bin/bash
set -euo pipefail
CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
source "\$CIDIR/model_flags.sh"
bash "\$CIDIR/install_mooncake.sh"
exec python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \
  $COMMON_FLAGS "\${MODEL_SERVER_ARGS[@]}" \
  --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
EOF

cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
docker rm -f mi355x_prefill 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_prefill \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV $DSV4_ENV_STR $MOONCAKE_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/prefill_entry.sh
EOF

cat > "$WORKDIR/decode.sh" <<EOF
#!/bin/bash
source "$WORKDIR/model_flags.sh"
docker rm -f mi355x_decode 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_decode \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV $DSV4_ENV_STR $MOONCAKE_ENV_STR "\${MODEL_ENV_ARGS[@]}" \
  $IMAGE bash /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/decode_entry.sh
EOF

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

# Bench script runs on the prefill node; \$PIP/\$DIP injected at srun time.
cat > "$WORKDIR/bench.sh" <<EOF
#!/bin/bash
set -e
PIP=\$1; DIP=\$2
docker rm -f mi355x_bench 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_bench \
  -e PIP=\$PIP -e DIP=\$DIP \
  $IMAGE bash -lc '
    export PYTHONPATH=/sgl-workspace/sglang/python:\$PYTHONPATH
    echo "[wait] prefill"; for i in \$(seq 1 600); do curl -sf http://\$PIP:$PPORT/health >/dev/null && break; sleep 5; done
    echo "[wait] decode";  for i in \$(seq 1 600); do curl -sf http://\$DIP:$DPORT/health >/dev/null && break; sleep 5; done
    python3 -m sglang_router.launch_router \
      --pd-disaggregation \
      --prefill http://\$PIP:$PPORT $PBOOT \
      --decode http://\$DIP:$DPORT \
      --host 0.0.0.0 --port $LBPORT \
      --disable-circuit-breaker &
    for i in \$(seq 1 30); do curl -sf http://127.0.0.1:$LBPORT/health >/dev/null && break; sleep 2; done
    CIDIR=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}
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
WORKDIR="$1"; PW="${2:-1}"; DW="${3:-1}"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
PNODES=("${NODES[@]:0:PW}")
DNODES=("${NODES[@]:PW:DW}")
PNODE="${PNODES[0]}"; DNODE="${DNODES[0]}"
PIP=$(getent ahostsv4 "$PNODE" | head -1 | awk '{print $1}')
DIP=$(getent ahostsv4 "$DNODE" | head -1 | awk '{print $1}')
echo "[drive] prefill nodes: ${PNODES[*]} ; decode nodes: ${DNODES[*]}"
echo "[drive] bench targets prefill=$PNODE($PIP) decode=$DNODE($DIP)"
if (( PW > 1 || DW > 1 )); then
  echo "[drive] NOTE: router + bench use the first prefill and first decode only;"
  echo "[drive]       multi-prefill/multi-decode fan-out is not wired yet (LB work)."
fi
# Each server's srun runs here on the login node and returns exactly when its
# compute-node container exits. Wrap it so the return code lands in a marker
# file on shared NFS. The monitor then watches for markers instead of polling
# PIDs -- unambiguous (no zombie/kill -0 guesswork) and it records which role
# died and with what code. (A hung-but-alive server is NOT caught here; that is
# bounded by bench.sh's health-wait timeout.)
rm -f "$WORKDIR"/server_exit_* "$WORKDIR/bench_exit"
for n in "${PNODES[@]}"; do
  ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/prefill.sh" > "$WORKDIR/prefill_$n.log" 2>&1
    echo "prefill@$n rc=$?" > "$WORKDIR/server_exit_prefill_$n" ) &
done
for n in "${DNODES[@]}"; do
  ( srun --overlap -N1 --nodelist="$n" bash "$WORKDIR/decode.sh" > "$WORKDIR/decode_$n.log" 2>&1
    echo "decode@$n rc=$?" > "$WORKDIR/server_exit_decode_$n" ) &
done
sleep 5
# Bench in the background with its own marker, so the wait loop is purely file
# based: finish when bench writes its marker, abort if any server marker shows up
# first (a server died before the sweep completed).
( srun --overlap -N1 --nodelist="$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" > "$WORKDIR/bench.log" 2>&1
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

# One node per prefill/decode worker (TP == GPUs/node). 1P1D -> 2 nodes.
TOTAL_NODES=$((PW + DW))

# Name the allocation <RUNNER_NAME>-<GITHUB_RUN_ID>-<config> so the workflow's
# cleanup steps can scancel precisely instead of a blanket `squeue --me` that
# would kill a concurrent matrix leg. RUNNER_NAME alone is not assumed unique;
# GITHUB_RUN_ID + config make the name unique per matrix leg regardless.
JOB_NAME="mi355x-ci-${RUNNER_NAME:-norunner}-${GITHUB_RUN_ID:-0}-${MATRIX_CONFIG_NAME}"

set +e
salloc -p "$SLURM_PARTITION" -N"$TOTAL_NODES" "${NODELIST_ARG[@]}" "${EXCLUDE_ARG[@]}" "${EXCLUSIVE_ARG[@]}" \
    --job-name "$JOB_NAME" -t "$TIME_LIMIT" \
    bash "$WORKDIR/drive.sh" "$WORKDIR" "$PW" "$DW"
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
TOTAL_GPUS=$((PTP + DTP))
PROCESSED=0
for C in ${CONCS//,/ }; do
    RAW="$WORKDIR/raw_conc${C}.json"
    [[ -f "$RAW" ]] || { echo "WARN: missing $RAW"; continue; }
    DEST="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${MATRIX_CONFIG_NAME}_conc${C}_gpus_${TOTAL_GPUS}_ctx_${PTP}_gen_${DTP}.json"
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

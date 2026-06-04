#!/usr/bin/env bash
# Kimi-K2.5-NVFP4 ONLY — acc + bench + profile, base vs variant, 2-node MNNVL. HARDENED.
#
# Per cell (base, then variant): launch graph-ON -> acc (logprobs) -> bench (bs16/32/64) ->
# profile graph-ON bs16+bs64 -> relaunch graph-OFF -> profile bs16. Downloads incrementally.
#
# Robustness baked in from real failures (see SKILL.md "Hard-won robustness"):
#   * kill_all kills LOCAL orphaned kubectl-exec launchers + VERIFIES GPU=0 on both nodes
#     (the #1 fix: a stale driver's launch racing a new one hangs the cold autotune at 1/20 / exit-7).
#   * wait_ready waits 40min for the cold fp4_gemm autotune, logs progress, DIED only when ALL
#     sglang procs are gone (a narrow `pgrep launch_server` false-DIEDs mid-autotune).
#   * launch retries once on failure (transient rank death happens).
#   * mem-fraction 0.83 (NOT 0.88 — the trtllm-LoRA decomposed path OOMs at 0.88).
#   * NEVER --disable-flashinfer-autotune (it lowers speed / flatters overlap). Pay the cold autotune.
#   * bench uses --result-filename + tee (NEVER grep|tail — you lose the table + prefill/decode split).
set -uo pipefail   # NOT -e: failures are handled explicitly (launch retry); -e would abort on a transient.

# ===== identity / pods =====
ID="${ID:?export ID=<dns-safe-identifier> (names the pods: mnnvl-kimi-<ID>-0/1)}"
HEAD_POD=mnnvl-kimi-${ID}-0
WORKER_POD=mnnvl-kimi-${ID}-1
DIST_INIT=mnnvl-kimi-${ID}-0.mnnvl-kimi-${ID}-head:20000
MODEL_PATH=/root/Kimi-K2.5-NVFP4
LORA_PATH=/root/kimi_k25_lora_alpha
LORA_NAME=alpha
MAX_LORA_RANK=32
PORT=30000
ACC_DATA="${LORA_PATH}/compare_sample_train_data.pt"   # ships inside the LoRA adapter repo

# ===== cells: base (control) vs variant (candidate) — EDIT per A/B =====
# Each cell = REF (injected branch, §3.3) + LORA on/off + EXTRA server flags + ENVS (launch env prefix).
# DEFAULT below: no-LoRA base  vs  trtllm-LoRA + two-stream (the trtllm-lora-opti workflow).
# For an ACC REGRESSION check, make the two cells NUMERICALLY EQUIVALENT (e.g. trtllm-LoRA-2stream vs
# trtllm-LoRA-no-2stream, or trtllm-LoRA vs cutlass-LoRA) — base-vs-LoRA is an *intended* diff, not a regression.
# === MoE TP8 vs EP8, NO-LoRA (this task) ===
# base    = today's no-LoRA baseline: --tp 8, no EP, default backend (auto -> flashinfer_trtllm, a2a=none).
# variant = the SAME, + ONLY `--ep-size 8` (keeps flashinfer_trtllm + a2a=none; just turns EP on so each
#           rank owns 384/8=48 experts). NOT a cutlass/cutedsl swap, NOT deepep. Both cells = same commit.
# Numerically equivalent (EP vs TP is a math-equivalent expert reshuffle, modulo atomic-add noise) =>
# the acc logprob diff is a real regression check; expect it within the ~0.30 noise floor.
# --- WITH-LoRA A/B (same trtllm LoRA stack on both; variant adds only --ep-size 8) ---
# base = TP8 + LoRA ; variant = EP8 + LoRA. (no-LoRA TP8/EP8 already measured in a prior RUN_ROOT.)
LORA_STACK_ENVS="SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_LORA_TWO_STREAM=1"
BASE_REF=__bench_target;     BASE_LORA=on;    BASE_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm --lora-use-virtual-experts";              BASE_ENVS="$LORA_STACK_ENVS"
VARIANT_REF=__bench_target;  VARIANT_LORA=on; VARIANT_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm --lora-use-virtual-experts --ep-size 8"; VARIANT_ENVS="$LORA_STACK_ENVS"
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }
cell_envs(){  [ "$1" = base ] && echo "$BASE_ENVS"  || echo "$VARIANT_ENVS"; }

# ===== workload =====
IN=2048; OUT=2048; PROF_OUT=64; BENCH_BS="16 32 64 128 256"   # larger-bs sweep: EP amortizes at big bs
OUTROOT=/tmp/kimi_reg
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_kimi_reg_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/kimi"; mkdir -p "$LOCAL_OUT"

COMMON="--model-path ${MODEL_PATH} --tp 8 --nnodes 2 --dist-init-addr ${DIST_INIT} \
--host 0.0.0.0 --port ${PORT} --quantization modelopt_fp4 --mem-fraction-static 0.83 \
--cuda-graph-max-bs 256 --trust-remote-code --max-prefill-tokens 40960 --chunked-prefill-size 40960 \
--dist-timeout 7200 --watchdog-timeout 7200"
# ^ EP8 first-launch: each rank cold-JIT-autotunes its OWN 48-expert grouped-GEMM shapes at different
#   speeds (head stuck on a 340s step-1 while worker raced to 8/20) → they desync and the cross-rank
#   collective/watchdog aborts before the cache is written. Big dist+watchdog timeouts let the slow rank
#   finish all 20 steps so the autotune cache gets WRITTEN once (warm). Subsequent EP8 launches load the
#   cache → no per-rank JIT skew → synchronized. (TP8 doesn't hit this: replicated GEMM, lockstep tune.)
NCCL="NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

kh(){ kubectl exec "${HEAD_POD}"   -- bash -lc "$1"; }
kw(){ kubectl exec "${WORKER_POD}" -- bash -lc "$1"; }

# ---- BULLETPROOF cleanup (the #1 robustness fix) ----
kill_all(){
  pkill -9 -f "kubectl exec.*launch_server" 2>/dev/null || true   # LOCAL orphaned launch clients
  sleep 2
  for i in $(seq 1 30); do
    for P in "$WORKER_POD" "$HEAD_POD"; do
      kubectl exec "$P" -- bash -lc 'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; pkill -9 -f "[s]glang" 2>/dev/null; pkill -9 -f "[b]ench_one_batch" 2>/dev/null; true' >/dev/null 2>&1
    done
    g0=$(kh 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l'|tr -d ' ')
    g1=$(kw 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l'|tr -d ' ')
    [ "${g0:-1}" = 0 ] && [ "${g1:-1}" = 0 ] && { echo "  GPU clean (iter $i)"; break; }
    sleep 4
  done
  sleep 6
}

checkout(){
  # source ~/.cargo/env before pip install — the branch editable rebuild needs the Rust compiler,
  # which `kubectl exec bash -lc` does not put on PATH (setup.sh sourced it directly). Without this the
  # rebuild dies with "can't find Rust compiler".
  kw "cd /root/sglang; git checkout -q --detach $1; . \$HOME/.cargo/env 2>/dev/null||true; pip install -e python >/tmp/pip.log 2>&1"
  kh "cd /root/sglang; git checkout -q --detach $1; . \$HOME/.cargo/env 2>/dev/null||true; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"
}

# Pre-warm the HF dynamic-module cache so 4 ranks/node don't race copying trust_remote_code *.py.
prewarm(){ for P in "$WORKER_POD" "$HEAD_POD"; do kubectl exec "$P" -- bash -lc 'python3 -c "from transformers import AutoConfig,AutoTokenizer,AutoProcessor as P; m=\"/root/Kimi-K2.5-NVFP4\"; AutoConfig.from_pretrained(m,trust_remote_code=True); AutoTokenizer.from_pretrained(m,trust_remote_code=True); P.from_pretrained(m,trust_remote_code=True)" 2>/dev/null; echo "prewarmed $P"'; done; }

# ---- observable, patient wait_ready (cold autotune ~17-21min; cache shared across configs) ----
wait_ready(){
  for i in $(seq 1 240); do
    kh "curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1" && { echo "  READY (~$((i*10+12))s)"; return 0; }
    n=$(kh 'pgrep -cf "[s]glang" 2>/dev/null'|tr -d ' ')
    [ "${n:-0}" = 0 ] && { echo "  DIED (0 sglang procs)"; kh 'tr "\r" "\n" </tmp/server.log|grep -aviE "shards: +[0-9]+%|profile/s"|tail -15'; return 1; }
    [ $((i % 9)) = 0 ] && echo "  ...i=$i procs=$n | $(kh 'tr "\r" "\n" </tmp/server.log 2>/dev/null|grep -aiE "autotune|Tuning|capturing|ready"|tail -1')"
    sleep 10
  done
  echo "  TIMEOUT"; return 1
}

start_rank(){ kubectl exec "$1" -- bash -lc "cd /root/sglang && ${NCCL} $3 exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} --node-rank $2 $4 > /tmp/server.log 2>&1" >/dev/null 2>&1 & }

launch(){  # $1=cell  $2=on|off  → retries once on failure
  local lora extra envs lora_flags="" graph_flags="" flags attempt
  lora=$(cell_lora "$1"); extra=$(cell_extra "$1"); envs=$(cell_envs "$1")
  [ "$lora" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ] && graph_flags="--disable-cuda-graph"
  flags="${graph_flags} ${extra} ${lora_flags}"
  for attempt in 1 2; do
    kill_all
    kw "for i in \$(seq 1 60); do getent hosts ${DIST_INIT%%:*} >/dev/null 2>&1 && break; sleep 2; done"
    start_rank "$WORKER_POD" 1 "$envs" "$flags"
    start_rank "$HEAD_POD"   0 "$envs" "$flags"
    sleep 12
    for spec in "${WORKER_POD}:1" "${HEAD_POD}:0"; do
      local pod="${spec%%:*}" rank="${spec##*:}"
      kubectl exec "$pod" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' || { echo "  rank${rank} on ${pod} not up — restart"; start_rank "$pod" "$rank" "$envs" "$flags"; }
    done
    wait_ready && return 0
    echo "  launch attempt ${attempt} failed ($1 graph-$2) — $([ "$attempt" = 1 ] && echo 'retry clean' || echo 'give up')"
  done
  return 1
}

acc(){   local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"; local d="${OUTROOT}/$1/acc"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"; }
bench(){ local la="";   [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/bench"
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${la} --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"; }
prof(){  local la="";   [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/profile_graph_$2/bs$3"
  kh "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size $3 --input-len ${IN} --output-len ${PROF_OUT} ${la} --profile --profile-activities CPU GPU --profile-start-step 4 --profile-steps 12 --profile-prefix kimi_$1_graph_$2_bs$3 --profile-output-dir ${d} --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; find ${d} -name '*.trace.json.gz' -printf '  %p %s\n'|sort"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${HEAD_POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }
# Flattened, ASYMMETRIC trace pull. Traces live per-rank on the node that ran the rank, so:
#   graph-ON  -> ALL 8 TP ranks from BOTH pods (TP0-3 head, TP4-7 worker). ~4.4M/rank, cheap; this is the
#               real-timing trace you actually read, so get it complete. A plain head-only `dl` SILENTLY
#               collects just 4 of 8 ranks.
#   graph-OFF -> ONLY TP0 (==tp0ep0) from head. ~39M/rank; graph-off is for kernel STRUCTURE, 1 rank suffices.
# Layout: ${LOCAL_OUT}/<cell>/traces/graph_{on,off}/bs16_TP<r>.trace.json.gz (+ server_args.json on graph-on)
pull_traces(){  # $1=cell  $2=on|off
  # NOTE: assign cell/g on their OWN `local` first — `local` expands ALL its arg words before binding
  # any, so referencing ${cell}/${g} in a later same-line assignment errors under `set -u` (unbound).
  local cell=$1 g=$2
  local src="${OUTROOT}/${cell}/profile_graph_${g}/bs16" dst="${LOCAL_OUT}/${cell}/traces/graph_${g}" ranks pod r s
  mkdir -p "$dst"
  [ "$g" = on ] && ranks="0 1 2 3 4 5 6 7" || ranks="0"
  for r in $ranks; do
    [ "$r" -le 3 ] && pod="$HEAD_POD" || pod="$WORKER_POD"
    kubectl exec "$pod" -- bash -lc "cat ${src}/*/*-TP-${r}.trace.json.gz 2>/dev/null" > "${dst}/bs16-TP-${r}.trace.json.gz" 2>/dev/null
    s=$(stat -f%z "${dst}/bs16-TP-${r}.trace.json.gz" 2>/dev/null || stat -c%s "${dst}/bs16-TP-${r}.trace.json.gz" 2>/dev/null || echo 0)
    if [ "${s:-0}" -lt 10000 ]; then echo "  MISSING graph_${g} TP${r} (${s}B)"; rm -f "${dst}/bs16-TP-${r}.trace.json.gz"; else echo "  graph_${g}/bs16_TP${r}  $((s/1024/1024))M"; fi
  done
  [ "$g" = on ] && { kubectl exec "$HEAD_POD" -- bash -lc "cat ${src}/*/server_args.json 2>/dev/null" > "${dst}/server_args.json" 2>/dev/null; [ -s "${dst}/server_args.json" ] || rm -f "${dst}/server_args.json"; }
}

# acc_capture.py (per-token logprobs over compare_sample_train_data.pt via /generate return_logprob)
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests, json
ap=argparse.ArgumentParser(); ap.add_argument("--port",required=True); ap.add_argument("--data",required=True); ap.add_argument("--lora",default=""); ap.add_argument("--out",required=True); a=ap.parse_args()
data=torch.load(a.data,weights_only=False); toks=data["tokens"]
if torch.is_tensor(toks): toks=toks.tolist()
seqs=toks if (toks and isinstance(toks[0],list)) else [toks]; lp=[]
for s in seqs:
    p={"input_ids":s,"sampling_params":{"max_new_tokens":0,"temperature":0.0},"return_logprob":True,"logprob_start_len":0}
    if a.lora: p["lora_path"]=a.lora
    r=requests.post(f"http://127.0.0.1:{a.port}/generate",json=p,timeout=1800); r.raise_for_status()
    lp+=[x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]   # [1:] skips BOS (no logprob)
json.dump(lp,open(a.out,"w")); print("wrote",len(lp),"logprobs ->",a.out)
PY
kubectl cp /tmp/acc_capture.py ${HEAD_POD}:/root/acc_capture.py >/dev/null

run_cell(){  # $1=cell (base|variant)  — runs ALL THREE tests
  echo "================= CELL $1 ================="
  checkout "$(cell_ref "$1")"
  launch "$1" on || { echo "[$1] graph-ON launch FAILED after retry — skipping cell"; return 1; }
  acc   "$1"; dl "$1/acc";                    echo "[$(date +%H:%M:%S)] kimi $1 ACC done"   | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench";                  echo "[$(date +%H:%M:%S)] kimi $1 BENCH done" | tee -a "${RUN_ROOT}/progress.log"
  prof  "$1" on 16; pull_traces "$1" on            # graph-on: bs16 only, all 8 TP ranks from both pods
  launch "$1" off && { prof "$1" off 16; pull_traces "$1" off; } || echo "[$1] graph-OFF launch FAILED — graph-off profile skipped"
  echo "[$(date +%H:%M:%S)] kimi $1 PROFILE done"  | tee -a "${RUN_ROOT}/progress.log"
}

prewarm
# CELLS lets you re-run a subset after a partial failure (e.g. CELLS=variant). Default: both.
for __c in ${CELLS:-base variant}; do run_cell "$__c"; done
kill_all
find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} + 2>/dev/null && echo "traces integrity OK"
echo "[$(date +%H:%M:%S)] kimi DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"

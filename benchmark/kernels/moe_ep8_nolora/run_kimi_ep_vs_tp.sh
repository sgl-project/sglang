#!/usr/bin/env bash
# MoE EP8 vs TP8 — no-LoRA baseline bench & profile for Kimi-K2.5-NVFP4 (2-node MNNVL/GB200).
#
# Purpose: measure how much switching the MoE from TP8 (every rank holds the full per-expert weight,
# GEMM rhs not sliced) to EP8 (each rank owns 384/8 = 48 experts, everything 1/8-sliced) costs the
# *no-LoRA* baseline — especially at LARGER batch sizes. This is the foundation for moving the LoRA
# MoE kernels off the memory-bound TP layout. See JOURNAL.md in this dir.
#
# This is a focused experiment derived from sglang-lora-base-perf-benchmark.md §3 (Kimi). It drops the
# LoRA cell entirely and instead runs the no-LoRA baseline under two parallelism variants.
#
# Run from the HEAD pod's driver shell (your local machine, with kubectl ctx `leira`).
# Required: ID (k8s identifier, set once). Optional knobs are env vars (see below) so EP backend
# choices can be iterated on-pod WITHOUT re-pushing the bundle.
set -euo pipefail

# ===== identity / topology =====
ID="${ID:?export ID=yushengsu-<date>-<time> first}"
HEAD_POD=mnnvl-kimi-${ID}-0
WORKER_POD=mnnvl-kimi-${ID}-1
DIST_INIT=mnnvl-kimi-${ID}-0.mnnvl-kimi-${ID}-head:20000
MODEL_PATH=/root/Kimi-K2.5-NVFP4
PORT=30000
HEAD_REF=__bench_target          # the commit under test (this branch HEAD)

# ===== variant selection =====
# VARIANT=tp8  -> control: --tp 8, no EP, default MoE backend (matches the skill's no-LoRA baseline)
# VARIANT=ep8  -> --tp 8 --ep-size 8 with the fp4 EP backends below
VARIANT="${VARIANT:?set VARIANT=tp8 or VARIANT=ep8}"

# EP8 policy: KEEP the exact same MoE backend + a2a as today's no-LoRA baseline — i.e. do NOT switch
# the runner to cutlass/cutedsl (neither is fast on this NVFP4/Blackwell path) and do NOT use a deepep
# a2a (deepep low_latency = NVSHMEM/IBGDA, built for inter-node IB; this is an MNNVL fabric where all
# 8 GPUs share one NVLink domain, so a deepep dispatch is the slow path). The baseline auto-resolves to
# moe_runner_backend=flashinfer_trtllm (the "trtllm-gen" MoE) + moe_a2a_backend=none, and that auto
# selection only checks (quant in {fp8,fp4} & a2a==none & runner==auto) — it does NOT depend on ep_size
# (server_args.py ~L2404). So adding ONLY `--ep-size 8` keeps flashinfer_trtllm + a2a=none and just
# turns on EP (each rank owns 384/8=48 experts), with combine over the existing NVLink TP communicator.
# EXPERT_LOC is the only EP knob, for the balancedness fallback (--init-expert-location).
EXPERT_LOC="${EXPERT_LOC:-trivial}"

# ===== workload =====
# Larger-bs sweep is the whole point of this experiment. cuda-graph-max-bs bumped to cover it.
IN=2048; OUT=2048
PROF_OUT=64
BENCH_BS="${BENCH_BS:-16 32 64 128 256}"
PROF_BS_ON="${PROF_BS_ON:-16 64 128 256}"
MAX_BS="${MAX_BS:-256}"

OUTROOT=/tmp/moe_ep_vs_tp
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/moe_ep8_nolora_${ID}_VARIANT_${VARIANT}}"
LOCAL_OUT="${RUN_ROOT}"

# ===== server args =====
COMMON="--model-path ${MODEL_PATH} --tp 8 --nnodes 2 --dist-init-addr ${DIST_INIT} \
--host 0.0.0.0 --port ${PORT} --quantization modelopt_fp4 --mem-fraction-static 0.88 \
--cuda-graph-max-bs ${MAX_BS} --trust-remote-code \
--max-prefill-tokens 40960 --chunked-prefill-size 40960"

if [ "${VARIANT}" = ep8 ]; then
  # Only EP on — same flashinfer_trtllm runner + a2a=none as the tp8 control (see policy note above).
  EP_FLAGS="--ep-size 8"
  [ "${EXPERT_LOC}" != trivial ] && EP_FLAGS="${EP_FLAGS} --init-expert-location ${EXPERT_LOC}"
  COMMON="${COMMON} ${EP_FLAGS}"
fi

ENVS="NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false"

kh(){ kubectl exec "${HEAD_POD}"   -- bash -lc "$1"; }
kw(){ kubectl exec "${WORKER_POD}" -- bash -lc "$1"; }
both(){ kw "$1"; kh "$1"; }
kill_all(){ both 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 8'; }
# NOTE: source ~/.cargo/env before pip install — the editable build needs the Rust compiler, which
# the in-pod setup.sh sourced directly but `kubectl exec bash -lc` does not put on PATH (=> the
# branch rebuild fails with "can't find Rust compiler" even though the base/main install succeeded).
checkout(){ both "cd /root/sglang; git cat-file -e $1^{commit} 2>/dev/null || git fetch origin main; git checkout -q --detach $1; . \$HOME/.cargo/env 2>/dev/null || true; pip install -e python >/tmp/pip.log 2>&1"; kh "cd /root/sglang && git --no-pager log -1 --oneline"; }
prewarm(){ for P in "${WORKER_POD}" "${HEAD_POD}"; do kubectl exec "${P}" -- bash -lc 'python3 -c "from transformers import AutoConfig, AutoTokenizer, AutoProcessor; m=\"/root/Kimi-K2.5-NVFP4\"; AutoConfig.from_pretrained(m, trust_remote_code=True); AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoProcessor.from_pretrained(m, trust_remote_code=True); print(\"prewarmed\")"'; done; }

start_rank(){  # $1=pod $2=node-rank $3=extra-flags
  kubectl exec "$1" -- bash -lc "cd /root/sglang && ${ENVS} exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} --node-rank $2 $3 > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  echo "started-rank$2 on $1 (VARIANT=${VARIANT})"
}
launch(){  # $1=on|off  (no-LoRA baseline only)
  local graph_flags=""; [ "$1" = off ] && graph_flags="--disable-cuda-graph"
  kw "for i in \$(seq 1 60); do getent hosts ${DIST_INIT%%:*} >/dev/null 2>&1 && break; sleep 2; done"
  start_rank "$WORKER_POD" 1 "$graph_flags"
  start_rank "$HEAD_POD"   0 "$graph_flags"
  sleep 12
  for spec in "${WORKER_POD}:1" "${HEAD_POD}:0"; do
    local pod="${spec%%:*}" rank="${spec##*:}"
    kubectl exec "$pod" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' \
      || { echo "rank${rank} on ${pod} not running — retry"; start_rank "$pod" "$rank" "$graph_flags"; }
  done
  kh "for i in \$(seq 1 600); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 100 /tmp/server.log; exit 1"
}
bench(){
  local d="${OUTROOT}/${VARIANT}/bench"
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}
profile(){  # $1=on|off $2=bs
  local d="${OUTROOT}/${VARIANT}/profile_graph_$1/bs$2"
  kh "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; \
     python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
       --batch-size $2 --input-len ${IN} --output-len ${PROF_OUT} \
       --profile --profile-activities CPU GPU --profile-start-step 4 --profile-steps 12 \
       --profile-prefix kimi_base_${VARIANT}_graph_$1_bs$2 --profile-output-dir ${d} \
       --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; \
     find ${d} -name '*.trace.json.gz' -printf '%p %s\n' | sort"
}
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${HEAD_POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

# ===== run =====
prewarm
kill_all; checkout "${HEAD_REF}"
kill_all; launch on
if [ "${RUN_BENCH:-true}" = true ]; then bench; dl "${VARIANT}/bench"; \
  echo "[$(date +%H:%M:%S)] ${VARIANT} BENCH done -> ${LOCAL_OUT}/${VARIANT}/bench" | tee -a "${RUN_ROOT}/progress.log"; fi
for bs in ${PROF_BS_ON}; do profile on "${bs}"; dl "${VARIANT}/profile_graph_on/bs${bs}"; done
kill_all; launch off
profile off 16; dl "${VARIANT}/profile_graph_off/bs16"
kill_all
dl "."
echo "DONE VARIANT=${VARIANT} -> ${LOCAL_OUT}"

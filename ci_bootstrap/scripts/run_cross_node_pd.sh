#!/usr/bin/env bash
# Orchestrate a cross-node 1P1D PD benchmark on two AMD MI35x nodes
# (one prefill, one decode) over the Pensando RoCE fabric, and run a benchmark
# client against the router.
#
# Phase 1 (manual): run this from your jumpbox/laptop. SSH keys must be
#                   installed on PREFILL_NODE and DECODE_NODE.
# Phase 2 (GH runner): run this inside an ARC runner pod that has SSH access
#                      to the two converted Conductor nodes. No script changes
#                      between phases — only where the script runs.
#
# NOTE: Slurm-gated nodes (pam_slurm_adopt) need an active allocation before SSH
# works. Allocate first, e.g. from a node with the Slurm client:
#   salloc -A <account> -p <partition> -w <node> -t 02:00:00 --no-shell
#
# Env vars (required):
#   PREFILL_NODE    SSH target (user@ip) for the prefill node
#   DECODE_NODE     SSH target (user@ip) for the decode node
#   PREFILL_IP      RoCE-routable IP of PREFILL_NODE (router/decode reach it here)
#   DECODE_IP       RoCE-routable IP of DECODE_NODE
#   CONTAINER       docker container name on both nodes (pre-launched, sglang+rocm)
#   REMOTE_WORKDIR  path inside container where scripts get rsynced
#
# Optional:
#   ROUTER_NODE     where to launch the proxy (default: PREFILL_NODE)
#   MODEL_PATH      default /data/models/DeepSeek-V4-Flash-MXFP4 (MXFP4; MI35x/gfx950)
#   TP_SIZE         default 8
#   TRANSFER_BACKEND  default mori; set "mooncake" for the mooncake KV path
#                     (needs an image with the #27730 mooncake bump)
#   BENCH_CONC      default "4"  (max-concurrency for cache_bench)
#   BENCH_NPROMPTS  default 32
#   BENCH_OUTLEN    default 200
#   RESULTS_DIR     local dir for collected artifacts (default ./results-<timestamp>)

set -euo pipefail

: "${PREFILL_NODE:?}"
: "${DECODE_NODE:?}"
: "${PREFILL_IP:?}"
: "${DECODE_IP:?}"
: "${CONTAINER:?}"
: "${REMOTE_WORKDIR:?}"

ROUTER_NODE="${ROUTER_NODE:-$PREFILL_NODE}"
MODEL_PATH="${MODEL_PATH:-/data/models/DeepSeek-V4-Flash-MXFP4}"
TP_SIZE="${TP_SIZE:-8}"
# KV transfer backend: mori (default) or mooncake. mooncake needs an image with
# the #27730 mooncake bump (d8f35569 / #2346 QP-teardown segfault fix).
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mori}"
BENCH_CONC="${BENCH_CONC:-4}"
BENCH_NPROMPTS="${BENCH_NPROMPTS:-32}"
BENCH_OUTLEN="${BENCH_OUTLEN:-200}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-./results-${TS}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RESULTS_DIR"

ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

stage() { echo; echo "==== [$(date +%T)] $* ===="; }

# ---- 0. Preflight: fail loudly on env issues BEFORE we launch anything -------
# Skip with SKIP_PREFLIGHT=1 if you need to re-run after a known-acceptable
# warning (don't make this a habit).
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    stage "preflight"
    bash "$SCRIPT_DIR/preflight_nodes.sh"
else
    stage "preflight SKIPPED (SKIP_PREFLIGHT=1)"
fi

# ---- 1. Ensure containers are up on every target node -------------------------
stage "setup containers"
bash "$SCRIPT_DIR/setup_container.sh"

# ---- 2. Push scripts to both nodes and into their containers ------------------
# cache_bench.py runs on ROUTER_NODE (stage 6) and is not part of the sglang
# image, so it must be pushed too. It is data, not an executable -> no chmod.
stage "push role scripts"
for pair in "$PREFILL_NODE:prefill_node.sh" "$DECODE_NODE:decode_node.sh" "$ROUTER_NODE:proxy_node.sh"; do
    tgt="${pair%%:*}"; f="${pair##*:}"
    scp $ssh_opts "$SCRIPT_DIR/$f" "${tgt}:/tmp/$f"
    ssh $ssh_opts "$tgt" "docker cp /tmp/$f ${CONTAINER}:${REMOTE_WORKDIR}/$f && \
                          docker exec ${CONTAINER} chmod +x ${REMOTE_WORKDIR}/$f"
done
scp $ssh_opts "$SCRIPT_DIR/cache_bench.py" "${ROUTER_NODE}:/tmp/cache_bench.py"
ssh $ssh_opts "$ROUTER_NODE" "docker cp /tmp/cache_bench.py ${CONTAINER}:${REMOTE_WORKDIR}/cache_bench.py"

# ---- 3. Launch prefill + decode in detached docker exec, capture logs ---------
stage "launch prefill on $PREFILL_NODE (HIP all 8 GPUs, listen $PREFILL_IP:30025)"
ssh $ssh_opts "$PREFILL_NODE" "\
    docker exec -d -e HOST_IP=${PREFILL_IP} -e MODEL_PATH=${MODEL_PATH} -e TP_SIZE=${TP_SIZE} \
        -e TRANSFER_BACKEND=${TRANSFER_BACKEND} \
        -w ${REMOTE_WORKDIR} ${CONTAINER} \
        bash -c './prefill_node.sh > /tmp/prefill.log 2>&1'"

stage "launch decode on $DECODE_NODE (listen $DECODE_IP:30100)"
ssh $ssh_opts "$DECODE_NODE" "\
    docker exec -d -e HOST_IP=${DECODE_IP} -e MODEL_PATH=${MODEL_PATH} -e TP_SIZE=${TP_SIZE} \
        -e TRANSFER_BACKEND=${TRANSFER_BACKEND} \
        -w ${REMOTE_WORKDIR} ${CONTAINER} \
        bash -c './decode_node.sh > /tmp/decode.log 2>&1'"

# ---- 4. Wait for both to be live ----------------------------------------------
wait_port() {
    local tgt="$1" ip="$2" port="$3" timeout="${4:-600}"
    local t=0
    while ! ssh $ssh_opts "$tgt" "exec 3<>/dev/tcp/${ip}/${port}" 2>/dev/null; do
        ((t+=5)); sleep 5
        if (( t >= timeout )); then
            echo "TIMEOUT waiting for ${ip}:${port}" >&2
            return 1
        fi
    done
    echo "  -> ${ip}:${port} live after ${t}s"
}
stage "wait for prefill 30025 + decode 30100 (up to 10 min each)"
wait_port "$PREFILL_NODE" "$PREFILL_IP" 30025 600
wait_port "$DECODE_NODE" "$DECODE_IP" 30100 600

# ---- 5. Launch router on ROUTER_NODE ------------------------------------------
stage "launch router on $ROUTER_NODE (8000 -> prefill+decode)"
ssh $ssh_opts "$ROUTER_NODE" "\
    docker exec -d -e PREFILL_IP=${PREFILL_IP} -e DECODE_IP=${DECODE_IP} \
        -w ${REMOTE_WORKDIR} ${CONTAINER} \
        bash -c './proxy_node.sh > /tmp/proxy.log 2>&1'"

stage "wait for router 8000"
ROUTER_IP="$(ssh $ssh_opts "$ROUTER_NODE" hostname -I | awk '{print $1}')"
wait_port "$ROUTER_NODE" "$ROUTER_IP" 8000 120

# ---- 6. Run benchmark client (from router node, against 127.0.0.1:8000) -------
stage "run cache_bench (n=${BENCH_NPROMPTS} outlen=${BENCH_OUTLEN} c=${BENCH_CONC})"
CSV_REMOTE="${REMOTE_WORKDIR}/bench_crossnode_${TS}.csv"
LOG_REMOTE="${REMOTE_WORKDIR}/bench_crossnode_${TS}.log"
ssh $ssh_opts "$ROUTER_NODE" "\
    docker exec -w ${REMOTE_WORKDIR} ${CONTAINER} \
        python3 cache_bench.py \
          --host 127.0.0.1 --port 8000 \
          --model ${MODEL_PATH} \
          --total-tokens 70000 --output-len ${BENCH_OUTLEN} \
          --num-prompts ${BENCH_NPROMPTS} \
          --pcts 0,20,40,60,80,90,92,95,97,99 \
          --tp-size ${TP_SIZE} --max-concurrency ${BENCH_CONC} \
          --csv-out ${CSV_REMOTE} 2>&1 | tee ${LOG_REMOTE}"

# ---- 7. Collect artifacts -----------------------------------------------------
stage "collect logs + csv into $RESULTS_DIR"
ssh $ssh_opts "$ROUTER_NODE"  "docker cp ${CONTAINER}:${CSV_REMOTE} /tmp/ && docker cp ${CONTAINER}:${LOG_REMOTE} /tmp/"
scp $ssh_opts "$ROUTER_NODE:/tmp/$(basename "$CSV_REMOTE")" "$RESULTS_DIR/"
scp $ssh_opts "$ROUTER_NODE:/tmp/$(basename "$LOG_REMOTE")" "$RESULTS_DIR/"
for pair in "$PREFILL_NODE:prefill.log" "$DECODE_NODE:decode.log" "$ROUTER_NODE:proxy.log"; do
    tgt="${pair%%:*}"; f="${pair##*:}"
    ssh $ssh_opts "$tgt" "docker exec ${CONTAINER} cat /tmp/$f" > "$RESULTS_DIR/${tgt//[@:]/_}_$f" || true
done

# ---- 8. Tear down servers (best-effort) ---------------------------------------
stage "teardown sglang processes in containers"
for tgt in "$PREFILL_NODE" "$DECODE_NODE" "$ROUTER_NODE"; do
    # sglang renames workers via setproctitle to 'sglang::scheduler/router/detokenizer',
    # so the launch-command pattern never matches. Use bracket-trick on the proctitle
    # (the '[s]glang' regex also prevents pkill from matching its own command line).
    ssh $ssh_opts "$tgt" "docker exec ${CONTAINER} pkill -9 -f '[s]glang' || true"
done

echo
echo "DONE. Artifacts in $RESULTS_DIR"
ls -la "$RESULTS_DIR"

#!/bin/bash
# Reproduce the SGLang MiniMax-M3 AgentX points: launch one server, run the requested concurrencies, summarize.
# Usage: bash reproduce.sh real|lossy [CONCS="1 8 24 32"] [GPUS=0,1,2,3] [PORT=30000] [MODELS_DIR=/scratch/models]
#   real  = recommended config (real EAGLE3 acceptance; the model's own outputs)
#   lossy = ATOM-parity performance-only config (forced acceptance length 2.78; outputs are not the model's)
# Durations follow the benchmark: 1800 s for c<=8, 3600 s otherwise. Results: /scratch/results/aiperf_<mode>_c<N>.
set -uo pipefail
MODE=${1:?real|lossy}; : "${CONCS:=1 8 24 32}" "${GPUS:=0,1,2,3}" "${PORT:=30000}" "${MODELS_DIR:=/scratch/models}"
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export MODEL=$MODELS_DIR/MiniMax-M3-MXFP4 REPO=$(cd "$HERE/../.." && pwd)
case $MODE in
  real)  source ./best_config.sh; EX="--max-running-requests 48 $EXTRA2" ;;
  lossy) source ./best_lossy_config.sh; EX="$EXTRA2 --schedule-policy lpm --max-running-requests 32" ;;
  *) echo "mode must be real or lossy"; exit 2 ;;
esac
bash ./kill_server.sh "$PORT" >/dev/null 2>&1 || true
TAG=$MODE GPUS=$GPUS PORT=$PORT SPEC_ATTN=decode EXTRA2="$EX" \
  ENVS2="NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1 $ENVS2" setsid nohup bash ./launch_v2.sh >/dev/null 2>&1 &
for i in $(seq 1 180); do curl -s -m 2 "localhost:$PORT/health" >/dev/null && break; sleep 10; done
curl -s -m 2 "localhost:$PORT/health" >/dev/null || { echo "server did not come up; see /scratch/logs/server_$MODE.log"; exit 1; }
echo "server up: $(grep -m1 GIT: /scratch/logs/server_$MODE.log)"
for C in $CONCS; do
  DUR=3600; [ "$C" -le 8 ] && DUR=1800
  TAG=$MODE CONC=$C DURATION=$DUR PORT=$PORT bash ./run_sa_point.sh > "/scratch/logs/bench_${MODE}_c$C.log" 2>&1
  python3 ./summarize.py "/scratch/results/aiperf_${MODE}_c$C" | tail -1
  python3 ./window_rate.py "/scratch/results/aiperf_${MODE}_c$C"
done
bash ./kill_server.sh "$PORT" >/dev/null 2>&1 || true

#!/usr/bin/env bash
# Emulate a 2-node TP=2 SGLang deployment on a single physical GPU.
#
# Technique: stas00/ml-engineering "Emulating multiple GPUs with a single GPU"
# (training/emulate-multi-node.md). The MIG variant is impossible inside this
# devbox pod (no CAP_SYS_ADMIN -> nvidia-smi -mig 1 fails), so both emulated
# "nodes" share the whole H200. The enabling mechanism is identical:
#   NCCL >= 2.31 (LD_PRELOAD over torch's bundled 2.29.7)
#   NCCL_MULTI_RANK_GPU_ENABLE=1   (allow >1 rank per physical GPU)
#   NCCL_NVLS_ENABLE=0             (no NVLink SHARP across emulated ranks)
# SGLang side (--tp is GLOBAL): --nnodes 2 --node-rank {0,1} gives one
# scheduler rank per process; each binds cuda:0 of the same GPU.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT0=30000
PORT1=30001
DIST_INIT=127.0.0.1:25000
LOG_DIR=/root/mig-demo
# Two server processes share one 141GB H200; keep each one's static pool small.
MEM_FRAC="${MEM_FRAC:-0.35}"

export LD_PRELOAD=/root/nccl-new/nvidia/nccl/lib/libnccl.so.2
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0   # both emulated nodes see the same whole GPU

pkill -f sglang.launch_server || true
sleep 2

common=(--model-path "$MODEL" --nnodes 2 --tp 2
        --dist-init-addr "$DIST_INIT"
        --disable-custom-all-reduce
        --mem-fraction-static "$MEM_FRAC"
        --host 127.0.0.1)

echo ">>> starting node-rank 0 (HTTP :$PORT0)"
nohup python -m sglang.launch_server "${common[@]}" \
    --node-rank 0 --port "$PORT0" \
    > "$LOG_DIR/node0.log" 2>&1 &
echo $! > "$LOG_DIR/node0.pid"

echo ">>> starting node-rank 1 (dummy health :$PORT1)"
nohup python -m sglang.launch_server "${common[@]}" \
    --node-rank 1 --port "$PORT1" \
    > "$LOG_DIR/node1.log" 2>&1 &
echo $! > "$LOG_DIR/node1.pid"

echo ">>> waiting for node 0 health"
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT0/health" > /dev/null 2>&1; then
        echo "READY after ~$((i * 5))s"
        exit 0
    fi
    if ! kill -0 "$(cat "$LOG_DIR/node0.pid")" 2>/dev/null; then
        echo "node0 died; tail of log:"
        tail -30 "$LOG_DIR/node0.log"
        exit 1
    fi
    sleep 5
done
echo "TIMEOUT waiting for health; tail of logs:"
tail -20 "$LOG_DIR/node0.log" "$LOG_DIR/node1.log"
exit 1

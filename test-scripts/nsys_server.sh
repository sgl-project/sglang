#!/usr/bin/env bash
# Launch a glm server UNDER nsys, gated on the cudaProfilerApi capture range.
# nsys only records between the server's cudaProfilerStart()/Stop(), which are
# triggered by /start_profile (activities=[CUDA_PROFILER]) and /stop_profile.
#
#   MODE=sparse bash test-scripts/nsys_server.sh   # current branch (glm-sparse)
#   MODE=dense  bash test-scripts/nsys_server.sh   # main worktree, no sparse config
#
# Produces: $REP_DIR/glm_<mode>.nsys-rep
# Then drive it with: MODE=<mode> bash test-scripts/nsys_drive.sh
set -euo pipefail
source /root/paddlejob/inference-public/denghaodong/code/sglang/.venv/bin/activate

MODE="${MODE:?MODE required (dense|sparse)}"
PORT="${PORT:-30000}"
MODEL_PATH="/root/paddlejob/inference-public/denghaodong/code/model/GLM_v2"
DENSE_DIR="/root/paddlejob/inference-public/denghaodong/code/sglang-main"
SPARSE_DIR="/root/paddlejob/inference-public/denghaodong/code/sglang"
REP_DIR="${REP_DIR:-/tmp/glm_nsys}"
mkdir -p "$REP_DIR"

# ---- pick worktree + (sparse only) write sparse config ----
if [[ "$MODE" == "dense" ]]; then
  SGLANG_DIR="$DENSE_DIR"
else
  SGLANG_DIR="$SPARSE_DIR"
  # same sparse defaults as bench_one.sh
  ENABLED=true TOPK=2048 FORCE_LEFT=64 FORCE_RIGHT=128 FREQ=4 \
  PATTERN="FFFSSSSSSSSSSSSSSSFSSFSSSSSFSFFSSFSSSSFSSSFFSS" \
  python3 - "$MODEL_PATH/config.json" <<'PYEOF'
import json, os, sys
p = sys.argv[1]; c = json.load(open(p))
def as_bool(s): return str(s).strip().lower() in ("1","true","yes","on")
pat = os.environ["PATTERN"]
c["glm_sparse_indexer_enabled"] = as_bool(os.environ["ENABLED"])
c["glm_sparse_indexer_topk"]    = int(os.environ["TOPK"])
c["glm_sparse_force_left"]      = int(os.environ["FORCE_LEFT"])
c["glm_sparse_force_right"]     = int(os.environ["FORCE_RIGHT"])
c["glm_sparse_index_topk_freq"] = int(os.environ["FREQ"])
c["glm_sparse_index_topk_pattern"] = None if pat == "None" else pat
json.dump(c, open(p,"w"), indent=2)
print("[nsys_server] sparse config written")
PYEOF
fi

cd "$SGLANG_DIR"
export PYTHONPATH="$SGLANG_DIR/python"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_USE_CUDA_NORM=1
echo "[nsys_server] mode=$MODE worktree=$SGLANG_DIR ($(git rev-parse --abbrev-ref HEAD))"

REP="$REP_DIR/glm_${MODE}"
# --capture-range=cudaProfilerApi: only record between cudaProfilerStart/Stop
# --capture-range-end=stop        : keep server alive after Stop (don't kill it)
# -t cuda,nvtx,cudnn,cublas        : CUDA kernels + NVTX ranges (op markers)
exec nsys profile \
  --output "$REP" \
  --force-overwrite true \
  --trace=cuda,nvtx,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-graph-trace=node \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 8 \
    --context-length 131072 \
    --max-running-requests "${MAX_RUNNING:-32}" \
    --quantization fp8 \
    --reasoning-parser glm45

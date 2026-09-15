# Shared locations for the MiniMax-M3 AgentX reproduction. Every script sources this; override any variable in the environment.
# Layout: the sglang checkout is the git root of this file; everything else lives under M3_WORK (default /scratch when
# writable, else ~/m3-agentx): models/, results/, logs/, aiter/, aiperf-sa/, aiperf-sa-venv/, .cache/huggingface/.
M3_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M3_ROOT=$(cd "$M3_HERE/.." && pwd)  # directory of reproduce.sh and the tuned csv / patch files
: "${SGLANG_DIR:=$(git -C "$M3_HERE" rev-parse --show-toplevel)}"
if [ -z "${M3_WORK:-}" ]; then if [ -d /scratch ] && [ -w /scratch ]; then M3_WORK=/scratch; else M3_WORK=$HOME/m3-agentx; fi; fi
: "${MODELS_DIR:=$M3_WORK/models}" "${RESULTS_DIR:=$M3_WORK/results}" "${LOGS_DIR:=$M3_WORK/logs}"
: "${AITER_DIR:=$M3_WORK/aiter}" "${AIPERF_DIR:=$M3_WORK/aiperf-sa}" "${AIPERF_VENV:=$M3_WORK/aiperf-sa-venv}"
: "${AIPERF_BIN:=$AIPERF_VENV/bin/aiperf}" "${PYTHON:=python3}"
: "${MODEL:=$MODELS_DIR/MiniMax-M3-MXFP4}" "${DRAFT_MODEL:=$MODELS_DIR/MiniMax-M3-EAGLE3-GQA}" "${TOKENIZER:=$MODEL}"
: "${HF_HOME:=$M3_WORK/.cache/huggingface}"
export M3_HERE M3_ROOT SGLANG_DIR M3_WORK MODELS_DIR RESULTS_DIR LOGS_DIR AITER_DIR AIPERF_DIR AIPERF_VENV AIPERF_BIN PYTHON MODEL DRAFT_MODEL TOKENIZER HF_HOME
mkdir -p "$RESULTS_DIR" "$LOGS_DIR" 2>/dev/null || true
# directory of the importable aiter package (for its stock tuned-GEMM csv); empty if aiter is not installed
aiter_pkg_dir() { "$PYTHON" -c "import os, aiter; print(os.path.dirname(aiter.__file__))" 2>/dev/null; }

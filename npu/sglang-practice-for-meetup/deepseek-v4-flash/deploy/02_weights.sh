#!/usr/bin/env bash
# 02_weights.sh — locate or fetch the Ascend-validated W8A8 (+MTP) checkpoint.
#
# The documented checkpoint is Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp (~300 GB) on ModelScope.
# This script is idempotent: if a complete-looking checkpoint is already present it does nothing.
#
# Usage: bash deploy/02_weights.sh [--check-only]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$HERE/.." && pwd)
# shellcheck source=../env/00_common.sh
source "$PROJ/env/00_common.sh"

CHECK_ONLY=0
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1

MODEL_ID=${MODEL_ID:-Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp}
NEED_GB=${WEIGHTS_NEED_GB:-320}      # 300 GB weights + index/temp headroom

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---- 1. already there? -----------------------------------------------------
find_existing() {
  local c
  for c in "$MODEL_DIR" /models/DeepSeek-V4-Flash-w8a8-mtp /data/models/DeepSeek-V4-Flash-w8a8-mtp \
           "$HOME/models/DeepSeek-V4-Flash-w8a8-mtp"; do
    [ -d "$c" ] || continue
    if [ -f "$c/config.json" ] && ls "$c"/*.safetensors >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if found=$(find_existing); then
  MODEL_DIR="$found"
  count=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)
  log "checkpoint already present: $MODEL_DIR (${count} shards, ${size})"
  log "if a previous download was interrupted, re-run the same download command to resume."
  export MODEL_DIR
  exit 0
fi

log "no checkpoint found in $MODEL_DIR or the usual provisioning paths"
[ "$CHECK_ONLY" -eq 1 ] && { log "check-only: exiting"; exit 2; }

# ---- 2. disk check --------------------------------------------------------
PARENT=$(dirname "$MODEL_DIR")
mkdir -p "$PARENT"
FREE_GB=$(df -Pk "$PARENT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
log "free disk on $PARENT: ${FREE_GB} GB (need ~${NEED_GB} GB)"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  log "NOT ENOUGH DISK — stop and ask the TA for a larger volume or a pre-provisioned weights path"
  exit 3
fi

# ---- 3. download (ModelScope first, HF mirror fallback) --------------------
if python3 -c "import modelscope" >/dev/null 2>&1; then
  log "downloading from ModelScope: $MODEL_ID"
  modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
elif command -v hf >/dev/null 2>&1 || python3 -c "import huggingface_hub" >/dev/null 2>&1; then
  log "ModelScope CLI not available; using huggingface_hub with the mirror"
  export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_ID", local_dir="$MODEL_DIR", max_workers=8)
PY
else
  log "neither modelscope nor huggingface_hub is installed."
  log "option A (recommended): pip install -U modelscope && rerun this script"
  log "option B: copy the checkpoint from the shared volume the organizers provide"
  exit 4
fi

# ---- 4. verify ------------------------------------------------------------
log "downloaded. sanity checks:"
[ -f "$MODEL_DIR/config.json" ] || { log "MISSING config.json — download incomplete"; exit 5; }
shards=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
[ "$shards" -gt 0 ] || { log "MISSING safetensors shards — download incomplete"; exit 5; }
python3 - <<PY
import json, os, glob
d = "$MODEL_DIR"
cfg = json.load(open(os.path.join(d, "config.json")))
n = len(glob.glob(os.path.join(d, "*.safetensors")))
total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, "*.safetensors")))
print(f"  model_type       : {cfg.get('model_type')}  ({cfg.get('architectures')})")
print(f"  layers / experts : {cfg.get('num_hidden_layers')} / {cfg.get('n_routed_experts')}")
print(f"  context          : {cfg.get('max_position_embeddings')}")
print(f"  shards           : {n}   weights on disk: {total/1e9:.1f} GB")
assert cfg.get("model_type") == "deepseek_v4", "unexpected model_type — wrong checkpoint?"
assert total > 2.5e11, "checkpoint looks too small — expected ~300 GB / 70 shards; download incomplete?"
print(f"  note: the native 159.6 GB FP4+FP8 checkpoint is NOT usable on A2/A3 (see troubleshooting: 'Unexpected type fp8')")
PY
# The W8A8 mirror does not always ship the `encoding/` helper that V4 needs instead of a chat template.
if [ ! -d "$MODEL_DIR/encoding" ]; then
  log "NOTE: no encoding/ directory in this checkpoint (V4 has no Jinja chat template)."
  log "      If /v1/chat/completions returns a template/400 error, either use /generate, or fetch it:"
  log "      HF_ENDPOINT=https://hf-mirror.com hf download deepseek-ai/DeepSeek-V4-Flash --include 'encoding/*' --local-dir $MODEL_DIR"
fi
log "checkpoint ready: $MODEL_DIR"

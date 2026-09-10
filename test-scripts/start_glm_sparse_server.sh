#!/usr/bin/env bash

# 1号机 http://10.52.101.26:30000 - V5
# 2号机 http://10.52.100.149:30000 - V3 baseline
# 3号机 http://10.52.101.14:30000  -  V5_opt
# 4号机 http://10.52.100.141:30000 -  dense baseline
# 5号机 http://10.52.98.151:30000 - V3 qwen tool parser tp8
# 6号机 http://10.52.106.215:30000 - V4

# pkill -9 -f "sglang.launch_server"
# tail -f /tmp/glm_sparse_test/server_freq4.log

set -euo pipefail
# source .venv/bin/activate

SCRIPT_LOG="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_LOG="${SCRIPT_LOG%.sh}_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee "$SCRIPT_LOG") 2>&1

MODEL_PATH="/root/paddlejob/inference-public/denghaodong/code/model/GLM_v2"
SGLANG_DIR="/root/paddlejob/inference-public/denghaodong/code/sglang"

# 稀疏配置（写入 config.json）
ENABLED="${ENABLED:-true}"
TOPK="${TOPK:-2048}"
FORCE_LEFT="${FORCE_LEFT:-64}"
FORCE_RIGHT="${FORCE_RIGHT:-128}"
FREQ="${FREQ:-4}"   
# 经过校准后的Pattern
# pattern 优先级高于 freq    
# PATTERN="${PATTERN:-None}"      
PATTERN="${PATTERN:-FSFSSFSFSSSFSSSSFFFSFSSSSSSFSSSSSFSSSSSSFSSSSS}" 


# 1) 写稀疏配置进 config.json（sglang 通过 getattr(config, ...) 读取，不走 CLI flag）
ENABLED="$ENABLED" TOPK="$TOPK" FORCE_LEFT="$FORCE_LEFT" FORCE_RIGHT="$FORCE_RIGHT" \
FREQ="$FREQ" PATTERN="$PATTERN" \
python3 - "$MODEL_PATH/config.json" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
c = json.load(open(p))

def as_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "on")

pat = os.environ["PATTERN"]
c["glm_sparse_indexer_enabled"] = as_bool(os.environ["ENABLED"])
c["glm_sparse_indexer_topk"] = int(os.environ["TOPK"])
c["glm_sparse_force_left"] = int(os.environ["FORCE_LEFT"])
c["glm_sparse_force_right"] = int(os.environ["FORCE_RIGHT"])
c["glm_sparse_index_topk_freq"] = int(os.environ["FREQ"])
c["glm_sparse_index_topk_pattern"] = None if pat == "None" else pat
json.dump(c, open(p, "w"), indent=2)
print("config.json updated:", {k: c[k] for k in (
    "glm_sparse_indexer_enabled","glm_sparse_indexer_topk",
    "glm_sparse_force_left","glm_sparse_force_right",
    "glm_sparse_index_topk_freq","glm_sparse_index_topk_pattern")})
PYEOF

# 2) 起 server
cd "$SGLANG_DIR"
export PYTHONPATH=/root/paddlejob/inference-public/denghaodong/code/sglang/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_USE_CUDA_NORM=1
nohup \
python3 -m sglang.launch_server \
    --model-path /root/paddlejob/inference-public/denghaodong/code/model/GLM_v2 \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 8 \
    --context-length 131072 \
    --max-running-requests 64 \
    --quantization fp8 \
    --reasoning-parser glm45 \
    --tool-call-parser qwen \
    > /tmp/glm_server_v5_opt.log 2>&1 &


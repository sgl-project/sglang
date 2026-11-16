#!/bin/bash
# Run on BOTH nodes of the 2-node RDMA cluster. Captures env + clones DeepEP sources.
# Usage: bash cluster_setup.sh <rank>
set -euxo pipefail
RANK=$1
mkdir -p /workspace/derisk/logs
cd /workspace/derisk

{
  echo "=== rank $RANK env capture ==="
  date -u
  hostname
  nvidia-smi -L
  nvcc --version | tail -2 || true
  python3 -c "import deep_ep; print('deep_ep file:', deep_ep.__file__)"
  pip list 2>/dev/null | grep -Ei 'deep.?ep|nccl|torch |mooncake|nixl' || true
  python3 - <<'EOF'
import ctypes
v = ctypes.c_int()
ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(v))
print("NCCL runtime version int:", v.value)  # 23007 = 2.30.7, 23102 = 2.31.2
EOF
  pip show nvidia-nccl-cu13 2>/dev/null | head -3 || true
  (ibv_devices || rdma link || true) 2>&1
  ls /dev/infiniband 2>/dev/null || true
  (ip -brief addr || ifconfig || hostname -I || true) 2>&1 | head -20
  env | grep -Ei 'nccl|efa|fabric|mooncake|nixl|rdma' || true
} 2>&1 | tee logs/env_rank${RANK}.log

# DeepEP base branch (tests matching the installed wheel) and PR #12 head
if [ ! -d deepep-base ]; then
  git clone --filter=blob:none https://github.com/sgl-project/DeepEP.git deepep-base
  cd deepep-base && git checkout sgl-deepep && cd ..
fi
if [ ! -d deepep-pr12 ]; then
  git clone --filter=blob:none https://github.com/sgl-project/DeepEP.git deepep-pr12
  cd deepep-pr12
  git fetch origin pull/12/head:pr12 || git fetch https://github.com/sgl-project/DeepEP.git pull/12/head:pr12
  git checkout 9f86f3147f9a6a2b02a8162429f356a556fd901f
  cd ..
fi
echo "SETUP_OK rank $RANK"

#!/bin/bash
# R5/R6 checks inside the PR image on a non-EFA (IB) node. Run: bash check_image.sh 2>&1 | tee /tmp/check_image.out
set -uo pipefail

echo "===== OS / arch ====="
uname -m

echo "===== pip NCCL + torch ====="
pip list 2>/dev/null | grep -Ei 'nccl|torch |nixl|mooncake|sgl' | head -20

echo "===== runtime NCCL version (torch-preloaded) ====="
python3 -c "import torch,ctypes; v=ctypes.c_int(); ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v)); print('nccl runtime:', v.value)"

echo "===== R5a: libfabric locations ====="
ls -la /opt/amazon/efa/lib 2>/dev/null | grep -E 'libfabric|libefa' || echo "no /opt/amazon/efa/lib"
ldconfig -p | grep -i fabric || echo "no libfabric in ldconfig"
find / -name 'libfabric.so*' -not -path '*/proc/*' 2>/dev/null | head -10
echo "--- LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "===== R5b: fi_info provider list ====="
FI=$(command -v fi_info || echo /opt/amazon/efa/bin/fi_info)
$FI -l 2>&1
echo "--- verbs provider usable?"
$FI -p verbs 2>&1 | head -15 || true

echo "===== R5c: python resolution of libfabric ====="
python3 - <<'EOF'
import ctypes, ctypes.util
p = ctypes.util.find_library('fabric')
print('find_library(fabric) ->', p)
try:
    lib = ctypes.CDLL('libfabric.so.1')
    print('loaded OK')
except OSError as e:
    print('load failed:', e)
EOF

echo "===== R5d: aws-ofi-nccl plugin presence ====="
find /opt /usr -name 'libnccl-net*.so' 2>/dev/null | head
ls /opt/amazon/efa 2>/dev/null

echo "===== R5e: NCCL net selection on IB (1 GPU ring self-test) ====="
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET python3 - <<'EOF' 2>&1 | grep -E 'NCCL INFO (NET|Using)|Bootstrap|nccl runtime' | head -20
import torch, torch.distributed as dist, os
os.environ.setdefault('MASTER_ADDR','127.0.0.1'); os.environ.setdefault('MASTER_PORT','29611')
torch.cuda.set_device(0)
dist.init_process_group('nccl', rank=0, world_size=1)
t = torch.zeros(1, device='cuda')
dist.all_reduce(t)
print('single-rank allreduce OK, nccl', torch.cuda.nccl.version())
dist.destroy_process_group()
EOF

echo "===== R6a: mooncake ====="
pip list 2>/dev/null | grep -i mooncake
python3 -c "import mooncake; print('mooncake import OK', getattr(mooncake,'__file__',''))" 2>&1 | tail -2
MC=$(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))" 2>/dev/null)
[ -n "$MC" ] && find "$MC" -name '*.so' | head -5 | while read so; do echo "--- $so"; ldd "$so" 2>/dev/null | grep -Ei 'fabric|ibverbs|rdmacm|ucp|not found' ; done

echo "===== R6b: NIXL backends ====="
python3 - <<'EOF' 2>&1 | tail -30
try:
    from nixl._api import nixl_agent, nixl_agent_config
    cfg = nixl_agent_config(backends=[])
    a = nixl_agent("probe", cfg)
    print("nixl agent OK; available backends:", a.get_available_backends() if hasattr(a,'get_available_backends') else 'n/a')
except Exception as e:
    print("nixl probe failed:", type(e).__name__, e)
# try LIBFABRIC backend explicitly
try:
    from nixl._api import nixl_agent
    a2 = nixl_agent("lf")
    params, mems = a2.get_backend_params("LIBFABRIC") if hasattr(a2,'get_backend_params') else (None,None)
    a2.create_backend("LIBFABRIC", {})
    print("NIXL LIBFABRIC backend init OK")
except Exception as e:
    print("NIXL LIBFABRIC init:", type(e).__name__, str(e)[:200])
EOF

echo "===== R6c: mooncake transfer engine init over IB ====="
python3 - <<'EOF' 2>&1 | tail -15
try:
    from mooncake.engine import TransferEngine
    te = TransferEngine()
    rc = te.initialize("127.0.0.1:12001", "P2PHANDSHAKE", "rdma", "")
    print("TransferEngine.initialize rc =", rc)
except Exception as e:
    print("mooncake TE:", type(e).__name__, str(e)[:300])
EOF

echo "===== DONE ====="

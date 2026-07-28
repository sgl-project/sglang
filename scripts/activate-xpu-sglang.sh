#!/bin/bash
# activate-xpu-sglang.sh — source this before running SGLang on XPU with pyxccl
#
# Usage:
#   source scripts/activate-xpu-sglang.sh
#   python -m sglang.launch_server \
#       --model-path facebook/opt-125m --device xpu --tp 2 \
#       --attention-backend intel_xpu
#
# See docs/platforms/xpu.pyxccl.md for the full setup guide.

# ---------------------------------------------------------------------------
# 1. Intel oneAPI compiler & SYCL runtime
# ---------------------------------------------------------------------------
source /work/compiler/setvars.sh

# ---------------------------------------------------------------------------
# 2. Runtime shared libraries
# ---------------------------------------------------------------------------
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/compiler/lib/

_CCL=/work/libraries.performance.communication.oneccl-v2/build-xpu-release/_install
export LD_LIBRARY_PATH=${_CCL}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${_CCL}/opt/mpi/libfabric/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# 3. oneCCL / OFI transport
# ---------------------------------------------------------------------------
export FI_PROVIDER_PATH=${_CCL}/opt/mpi/libfabric/lib/prov
export CCL_ATL_TRANSPORT=ofi
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0

# ---------------------------------------------------------------------------
# 4. SGLang pyxccl integration
# ---------------------------------------------------------------------------
# Opt into the direct-oneCCL path (pyxccl is off by default; XPU otherwise uses
# torch.distributed / XCCL).
export SGLANG_ENABLE_PYXCCL=1
export SGLANG_PYXCCL_SO_PATH=${_CCL}/lib/libccl.so.1

unset _CCL

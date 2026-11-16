#!/bin/bash
# Run on BOTH ranks. Builds DeepEP#12 wheel against NCCL 2.31.2 headers,
# WITHOUT touching the runtime env (side-staged NCCL) so baseline tests are unaffected.
set -euxo pipefail
cd /workspace/derisk

# Side-stage NCCL 2.31.2 (headers + lib) for the build only
if [ ! -d nccl2312/nvidia/nccl ]; then
  pip download nvidia-nccl-cu13==2.31.2 --no-deps -d /tmp/nccldl
  mkdir -p nccl2312 && cd nccl2312
  python3 -c "import zipfile,glob; zipfile.ZipFile(glob.glob('/tmp/nccldl/nvidia_nccl*.whl')[0]).extractall('.')"
  cd ..
fi
NCCL_DIR=/workspace/derisk/nccl2312/nvidia/nccl
test -f "$NCCL_DIR/include/nccl.h"
grep -E 'NCCL_VERSION_CODE|define NCCL_MAJOR|define NCCL_MINOR|define NCCL_PATCH' "$NCCL_DIR/include/nccl.h" | head -5

cd /workspace/derisk/deepep-pr12
rm -rf /workspace/derisk/wheels-pr12 build dist
EP_NCCL_ROOT_DIR="$NCCL_DIR" \
CPATH="$NCCL_DIR/include:${CPATH:-}" \
C_INCLUDE_PATH="$NCCL_DIR/include:${C_INCLUDE_PATH:-}" \
CPLUS_INCLUDE_PATH="$NCCL_DIR/include:${CPLUS_INCLUDE_PATH:-}" \
LIBRARY_PATH="$NCCL_DIR/lib:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="$NCCL_DIR/lib:${LD_LIBRARY_PATH:-}" \
MAX_JOBS=32 \
  bash sgl_deep_ep/build_sgl_deep_ep.sh /workspace/derisk/wheels-pr12 13.0 x86_64
ls -la /workspace/derisk/wheels-pr12/
echo "WHEEL_BUILD_OK"

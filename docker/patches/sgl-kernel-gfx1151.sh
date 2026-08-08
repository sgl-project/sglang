#!/bin/sh
# Teach python/sglang/kernels/aot/setup_rocm.py to build for gfx1151 (Strix Halo).
# Applied at image-build time by docker/rocm-gfx1151.Dockerfile; the repo files are
# left untouched because gfx1151 is not a supported SGLang target.
#
# Three changes:
#   1. Lift the {gfx942, gfx950, gfx1250} allowlist, which otherwise sys.exit(1)s.
#   2. Fix the TopK dynamic-LDS budget. The upstream expression is keyed on
#      "am I gfx942", so every other arch falls through to the 128KB branch.
#      gfx1151 reports lds_size_in_kb 64 and needs the 48KB budget. Re-key it
#      onto an explicit large-LDS allowlist so gfx950 and gfx1250 keep 128KB
#      while gfx942 and gfx1151 both get 48KB.
#   3. Force a single WARP_SIZE across the host and device compiler passes.
#      include/utils.h resolves WARP_SIZE to 64 whenever __HIP_DEVICE_COMPILE__
#      is undefined -- i.e. on the host pass -- and to 32 on a non-__GFX9__
#      device pass. On CDNA both come out 64 and nothing is wrong, which is why
#      upstream never sees this. On gfx1151 the two passes disagree, and the MoE
#      TopK kernels use WARP_SIZE on both sides of the launch:
#         moe_topk_softmax_kernels.cu  __launch_bounds__(WARPS_PER_CTA * WARP_SIZE)  -> device, 4*32 = 128
#         moe_topk_softmax_kernels.cu  dim3 block_dim(WARP_SIZE, WARPS_PER_TB)       -> host,   64*4 = 256
#      Launching 256 threads into a 128-thread bound fails with
#      hipErrorLaunchFailure, poisons the queue, and typically surfaces as a
#      page fault in whatever kernel runs next (moe_align_block_size_kernel),
#      which makes it easy to misattribute. The same pattern is in
#      moe_topk_sigmoid_kernels.cu. It also desynchronizes the launcher's
#      TopkConstants math (ROWS_PER_WARP, VECs_PER_THREAD) from the kernel's.
#      Only MoE models hit this; dense models never call these kernels.
#      32 is simply correct here -- gfx1151 is a wave32 part -- so the override
#      pins both passes to 32 rather than renaming the symbol per call site.
#
# Each edit is guarded: if the upstream line has changed, fail rather than
# silently produce an image whose kernels were built with the wrong limits.
set -e

FILE="${1:?usage: sgl-kernel-gfx1151.sh <path to setup_rocm.py>}"
UTILS="$(dirname "${FILE}")/include/utils.h"

GATE_OLD='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250"]:'
GATE_NEW='if amdgpu_target not in ["gfx942", "gfx950", "gfx1250", "gfx1151"]:'

LDS_OLD='topk_dynamic_smem_bytes = 48 * 1024 if amdgpu_target == "gfx942" else 32 * 1024 * 4'
LDS_NEW='topk_dynamic_smem_bytes = (
    32 * 1024 * 4 if amdgpu_target in ("gfx950", "gfx1250") else 48 * 1024
)'

FLAGS_OLD='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",'
FLAGS_NEW='    f"-DSGL_TOPK_DYNAMIC_SMEM_BYTES={topk_dynamic_smem_bytes}",
    # gfx1151 is wave32; pin both compiler passes to it (see utils.h below).
    *(["-DSGL_ROCM_WARP_SIZE=32"] if amdgpu_target == "gfx1151" else []),'

WARP_OLD='#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'
WARP_NEW='#if defined(SGL_ROCM_WARP_SIZE)
#define WARP_SIZE SGL_ROCM_WARP_SIZE
#elif defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
#define WARP_SIZE 64'

for pattern in "${GATE_OLD}" "${LDS_OLD}" "${FLAGS_OLD}"; do
    if ! grep -qF "${pattern}" "${FILE}"; then
        echo "ERROR: expected line not found in ${FILE}:" >&2
        echo "  ${pattern}" >&2
        echo "setup_rocm.py changed upstream; re-check this patch before building." >&2
        exit 1
    fi
done

if ! grep -qF "${WARP_OLD}" "${UTILS}"; then
    echo "ERROR: expected WARP_SIZE block not found in ${UTILS}." >&2
    echo "utils.h changed upstream; re-check the wave32 fix before building." >&2
    exit 1
fi

python3 - "${UTILS}" "${WARP_OLD}" "${WARP_NEW}" <<'PY'
import sys

path, old, new = sys.argv[1:4]
with open(path) as f:
    src = f.read()
with open(path, "w") as f:
    f.write(src.replace(old, new, 1))
PY

python3 - "${FILE}" "${GATE_OLD}" "${GATE_NEW}" "${LDS_OLD}" "${LDS_NEW}" "${FLAGS_OLD}" "${FLAGS_NEW}" <<'PY'
import sys

path, gate_old, gate_new, lds_old, lds_new, flags_old, flags_new = sys.argv[1:8]
with open(path) as f:
    src = f.read()
src = src.replace(gate_old, gate_new).replace(lds_old, lds_new)
src = src.replace(flags_old, flags_new, 1)
with open(path, "w") as f:
    f.write(src)
PY

echo "Patched ${FILE} for gfx1151:"
grep -nF -e "${GATE_NEW}" -e 'gfx950", "gfx1250")' -e "SGL_ROCM_WARP_SIZE" "${FILE}"
echo "Patched ${UTILS} for wave32:"
grep -nF "SGL_ROCM_WARP_SIZE" "${UTILS}"

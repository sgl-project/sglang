#!/bin/bash
# Qwen38 DeepEP v2 patch + build -- replaces the base image's stock DeepEP.
#
# The base image ships DeepEP at the legacy pin, which does not serve GB300 on
# CUDA 13. This installs the v2 tree instead:
#   - v2 moved to a CMake/JIT layout, so the cross-node timeout knobs now live
#     in csrc/kernels/legacy/compiled.cuh behind LEGACY_-prefixed macros
#   - the wheel links the nvidia-nvshmem/nccl wheels rather than a system NCCL
#   - CUDA 13 relocated the cccl headers; the legacy setuptools path needs an
#     extra include dir, the CMake path resolves them itself
#
# Both timeout bumps are asserted after the fact because `sed -i` exits 0 on no
# match: shipping the stock 100s CPU timeout is invisible at build time and
# surfaces only as GB300 multi-node init aborts in production.
#
# Env knobs (all optional):
#   DEEPEP_V2_COMMIT       pinned commit to build
#   TORCH_CUDA_ARCH_LIST   arches to compile cubins for
#   MAX_JOBS               nvcc build parallelism
#   DEEPEP_DIR             where the source tree lands in the image
set -euo pipefail

: "${DEEPEP_V2_COMMIT:=01dc3aaac82068020353dce2c302e38153c0bfaa}"
: "${TORCH_CUDA_ARCH_LIST:=9.0;10.0;10.3}"
: "${MAX_JOBS:=8}"
: "${DEEPEP_DIR:=/sgl-workspace/DeepEP}"
: "${CUDA_HOME:=/usr/local/cuda}"

BUILD_DIR=/build/DeepEP
TIMEOUT_HEADER=csrc/kernels/legacy/compiled.cuh

rm -rf "${BUILD_DIR}"
git clone https://github.com/deepseek-ai/DeepEP.git "${BUILD_DIR}"
cd "${BUILD_DIR}"
git checkout "${DEEPEP_V2_COMMIT}"

# --- Cross-node timeout headroom ---
sed -i \
  's/#define LEGACY_NUM_CPU_TIMEOUT_SECS 100/#define LEGACY_NUM_CPU_TIMEOUT_SECS 1000/' \
  "${TIMEOUT_HEADER}"
sed -i \
  's/#define LEGACY_NUM_TIMEOUT_CYCLES 200000000000ull/#define LEGACY_NUM_TIMEOUT_CYCLES 2000000000000ull/' \
  "${TIMEOUT_HEADER}"

if ! grep -q '#define LEGACY_NUM_CPU_TIMEOUT_SECS 1000' "${TIMEOUT_HEADER}"; then
  echo "ERROR: CPU timeout bump did not apply; the macro moved or was reformatted" >&2
  exit 1
fi
if ! grep -q '#define LEGACY_NUM_TIMEOUT_CYCLES 2000000000000ull' "${TIMEOUT_HEADER}"; then
  echo "ERROR: cycle timeout bump did not apply; the macro moved or was reformatted" >&2
  exit 1
fi

# --- CUDA 13 cccl include dir (legacy setuptools path only) ---
# A missing anchor is the expected outcome on a CMake tree, so report which
# path was taken rather than letting sed no-op silently.
if grep -q "^    include_dirs = \['csrc/'\]" setup.py; then
  sed -i \
    "/^    include_dirs = \['csrc\/'\]/a\\    include_dirs.append('${CUDA_HOME}/include/cccl')" \
    setup.py
  echo "Applied the CUDA 13 cccl include fix to setup.py"
else
  echo "setup.py has no legacy include_dirs anchor; relying on the CMake build to resolve cccl"
fi

# --- Build and swap in ---
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
  python3 setup.py bdist_wheel -d /wheels

# Remove whatever currently owns the deep_ep module, by asking the metadata
# rather than by name. The base image's provider has changed name: v0.5.17
# shipped a distribution literally called `deep_ep`, while the dev images ship
# `sgl-deep-ep`, which installs the same deep_ep/ path (plus its own
# deep_ep_cpp*.so). `pip uninstall deep_ep` exits 0 with only a warning when no
# such distribution exists, so hardcoding either name leaves the other in place
# and two distributions then contend for deep_ep/ -- whichever wrote the files
# last wins, which is not a decision this build should leave to chance.
#
# packages_distributions() reads metadata and does not import deep_ep, so the
# packaged provider's driver check cannot fire here.
#
# Run from / rather than the build tree: `setup.py bdist_wheel` above leaves a
# deep_ep.egg-info in ${BUILD_DIR}, and `python3 -` puts the cwd first on
# sys.path, so importlib.metadata would report that build artifact as an
# installed distribution named deep_ep on top of the real one.
mapfile -t DEEP_EP_DISTS < <(cd / && python3 - <<'PY'
from importlib.metadata import packages_distributions

for dist in sorted(set(packages_distributions().get("deep_ep", ()))):
    print(dist)
PY
)
if [ "${#DEEP_EP_DISTS[@]}" -gt 0 ]; then
    echo "Removing existing deep_ep provider(s): ${DEEP_EP_DISTS[*]}"
    python3 -m pip uninstall -y "${DEEP_EP_DISTS[@]}"
else
    echo "No installed distribution provides deep_ep; nothing to remove"
fi

python3 -m pip install /wheels/*.whl

# Exactly one provider must remain. Two would mean the removal above missed a
# name and the source build is now sharing deep_ep/ with a packaged copy.
# From / for the same reason as above -- the egg-info left in the build tree is a
# build artifact, not an installed distribution, and counting it here reported
# ['deep_ep', 'deep_ep'] and failed a build that was in fact correct.
( cd / && python3 - <<'PY'
from importlib.metadata import packages_distributions

providers = sorted(set(packages_distributions().get("deep_ep", ())))
if len(providers) != 1:
    raise SystemExit(
        f"expected exactly one distribution to provide deep_ep, found {providers}"
    )
print(f"deep_ep is provided by {providers[0]} alone")
PY
)

rm -rf "${DEEPEP_DIR}" /wheels "${BUILD_DIR}/build" "${BUILD_DIR}/dist" /root/.cache/pip
mv "${BUILD_DIR}" "${DEEPEP_DIR}"

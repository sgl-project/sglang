#!/usr/bin/env bash
# Apply the aiter dependencies required by Muse Spark on MI355X.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="${AITER_ROOT:-/sgl-workspace/aiter}"
ENABLE_PLAN_C="${ENABLE_PLAN_C:-0}"

git -C "${AITER_ROOT}" rev-parse --is-inside-work-tree >/dev/null
git -C "${AITER_ROOT}" submodule update --init --recursive 3rdparty/composable_kernel

apply_patch() {
  local patch="$1"
  shift
  if git -C "${AITER_ROOT}" apply "$@" --check "${patch}" 2>/dev/null; then
    git -C "${AITER_ROOT}" apply "$@" "${patch}"
    echo "applied: $(basename "${patch}")"
  elif git -C "${AITER_ROOT}" apply "$@" --reverse --check "${patch}" 2>/dev/null; then
    echo "already applied: $(basename "${patch}")"
  else
    echo "cannot apply $(basename "${patch}") to ${AITER_ROOT}" >&2
    echo "verify the pinned revisions in ${ROOT}/README.md" >&2
    exit 1
  fi
}

apply_patch "${ROOT}/patches/01-aiter-fix-page-scatter-vm-fault.patch"
apply_patch "${ROOT}/patches/03-plan-b-aiter-unified-attention-3d.patch" \
  --include="aiter/**"
apply_patch "${ROOT}/patches/04-aiter-muse-moe-flydsl.patch"

if [[ "${ENABLE_PLAN_C}" == "1" ]]; then
  apply_patch \
    "${ROOT}/patches/experimental/aiter-muse-flydsl-swa-decode.patch"
fi

# These JIT modules embed the CK FMHA headers and must be rebuilt after patch 01.
for mask in mask nmask; do
  module="mha_batch_prefill_bf16_nlogits_nbias_${mask}_nlse_ndropout_nqscale_nsink"
  rm -f "${AITER_ROOT}/aiter/jit/${module}.so"
  rm -rf "${AITER_ROOT}/aiter/jit/build/${module}"
done

echo "aiter Muse Spark dependencies are ready"

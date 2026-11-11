#!/usr/bin/env bash
# Apply the aiter dependencies required by Muse Spark on MI355X.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="${AITER_ROOT:-/sgl-workspace/aiter}"

git -C "${AITER_ROOT}" rev-parse --is-inside-work-tree >/dev/null

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

apply_patch "${ROOT}/patches/05-aiter-unified-attention-muse.patch"
apply_patch "${ROOT}/patches/06-aiter-muse-spark-bf16-tuned-gemm.patch"

echo "aiter Muse Spark dependencies are ready"

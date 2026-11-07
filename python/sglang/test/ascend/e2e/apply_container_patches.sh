#!/bin/bash
set -euo pipefail

# Apply declarative container patch sets inside a pod before the test starts.
#
# Patch sets are committed under ./container_patches/<name>.sh and selected by
# SGLANG_NPU_CONTAINER_PATCHES (comma separated) or by the first positional
# argument. An empty value is a no-op, so jobs that do not ask for a patch
# behave exactly as before.
#
# Notes:
#   * The set name comes from a workflow input, so only names matching
#     ^[a-z0-9_]+$ that resolve to a file in the patch directory are executed.
#   * Patch resources are cached on the shared disk and installed offline-first,
#     mirroring the KUBERNETES_PKG_PATH_SOURCE pattern in run_npu_testcase.sh,
#     so the pods of one run reuse the first download.
#   * A per-set marker keyed by the file checksum makes re-runs idempotent.
#   * Any failure aborts the caller: a half patched cluster produces results
#     that cannot be trusted.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${SCRIPT_DIR}/container_patches"
PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-/root/.cache/.cache/npu_patches}"
PATCH_STATE_FILE="${PATCH_STATE_FILE:-/tmp/sglang_npu_patches_applied.json}"
REQUESTED="${1:-${SGLANG_NPU_CONTAINER_PATCHES:-}}"

if [ -z "${REQUESTED}" ]; then
    echo "No container patch set requested, skip."
    exit 0
fi

APPLIED_JSON=""
NORMALIZED=""
while IFS= read -r set_name; do
    set_name="$(echo "${set_name}" | tr -d '[:space:]')"
    if [ -z "${set_name}" ]; then
        continue
    fi

    if ! echo "${set_name}" | grep -Eq '^[a-z0-9_]+$'; then
        echo "Invalid container patch set name: ${set_name}" >&2
        exit 1
    fi

    patch_file="${PATCH_DIR}/${set_name}.sh"
    if [ ! -f "${patch_file}" ]; then
        echo "Container patch set not found: ${patch_file}" >&2
        exit 1
    fi

    patch_sha="$(sha256sum "${patch_file}" | awk '{print $1}')"
    patch_version="$(grep -m1 '^# PATCH_VERSION:' "${patch_file}" | sed 's/^# PATCH_VERSION:[[:space:]]*//' || true)"
    patch_version="${patch_version:-unknown}"
    marker="/tmp/.sglang_npu_patch_${set_name}_${patch_sha}"

    if [ -f "${marker}" ]; then
        echo "Container patch set already applied, skip: ${set_name}"
        patch_status="skipped"
    else
        echo "===== Apply container patch set: ${set_name} - Begin ====="
        mkdir -p "${PATCH_CACHE_DIR}/${set_name}"
        PATCH_CACHE_DIR="${PATCH_CACHE_DIR}/${set_name}" bash "${patch_file}"
        echo "===== Apply container patch set: ${set_name} - End ====="
        touch "${marker}"
        patch_status="applied"
    fi

    if [ -n "${APPLIED_JSON}" ]; then
        APPLIED_JSON="${APPLIED_JSON},"
        NORMALIZED="${NORMALIZED},"
    fi
    NORMALIZED="${NORMALIZED}${set_name}"
    APPLIED_JSON="${APPLIED_JSON}{\"set\": \"${set_name}\", \"version\": \"${patch_version}\", \"sha256\": \"${patch_sha}\", \"status\": \"${patch_status}\"}"
done <<< "$(echo "${REQUESTED}" | tr ',' '\n')"

cat > "${PATCH_STATE_FILE}" <<EOF
{
  "requested": "${NORMALIZED}",
  "host": "${HOSTNAME:-unknown}",
  "recorded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "patches": [${APPLIED_JSON}]
}
EOF

echo "Container patch state written to ${PATCH_STATE_FILE}"
cat "${PATCH_STATE_FILE}"

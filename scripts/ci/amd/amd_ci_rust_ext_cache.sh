#!/bin/bash
# Host-side entry point for amd_ci_rust_ext_cache.py, run from the workspace
# root with ci_sglang already started and sglang installed.
#
# Usage:
#   amd_ci_rust_ext_cache.sh export DIR   # stage-a: build, then fill DIR for upload
#   amd_ci_rust_ext_cache.sh import DIR   # later stages: seed the cache from DIR
#
# DIR is relative to the workspace, which the container mounts at
# /sglang-checkout. It must stay outside rust/: the loader fingerprints every
# file under the Cargo workspace.
#
# DIR/image-id pins the modules to the exact image that built them. The loader's
# fingerprint already covers the torch version and ABI, but not the libtorch
# bytes, which differ between e.g. the mi30x and mi35x images of one torch
# release. Never fails the job: a module that is not imported is built at
# runtime, as it was before this existed.

set -uo pipefail

ACTION="${1:?export or import}"
DIR="${2:?directory relative to the workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_ID=$(docker inspect --format '{{.Image}}' ci_sglang 2>/dev/null || true)
if [[ -z "${IMAGE_ID}" ]]; then
  echo "::warning::could not read ci_sglang's image; skipping Rust extension ${ACTION}"
  exit 0
fi

run_helper() {
  bash "${SCRIPT_DIR}/amd_ci_exec.sh" -w /sglang-checkout \
    python3 /sglang-checkout/scripts/ci/amd/amd_ci_rust_ext_cache.py "$@"
}

case "${ACTION}" in
  export)
    rm -rf "${DIR}"
    # Created here so the runner owns it; the container (root) writes inside.
    mkdir -p "${DIR}"
    if run_helper export "/sglang-checkout/${DIR}"; then
      echo "${IMAGE_ID}" > "${DIR}/image-id"
    else
      echo "::warning::Rust extension export failed; later stages will build their own"
      rm -rf "${DIR}"
    fi
    ;;
  import)
    if [[ ! -f "${DIR}/image-id" ]]; then
      echo "No prebuilt Rust extensions in ${DIR}; they will be built at runtime"
      exit 0
    fi
    BUILT_ON=$(cat "${DIR}/image-id")
    if [[ "${BUILT_ON}" != "${IMAGE_ID}" ]]; then
      echo "Prebuilt Rust extensions come from image ${BUILT_ON}, this container runs ${IMAGE_ID}; building at runtime instead"
      exit 0
    fi
    run_helper import "/sglang-checkout/${DIR}" \
      || echo "::warning::Rust extension import failed; they will be built at runtime"
    ;;
  *)
    echo "Unknown action '${ACTION}'; expected export or import" >&2
    exit 1
    ;;
esac
exit 0

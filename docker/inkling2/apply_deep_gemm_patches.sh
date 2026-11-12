#!/usr/bin/env bash
# Apply the Inkling2 MegaMoE patch (SiTU for NVFP4 / FP8xFP8, and the `inkling2` weight
# layout that reads the loaded expert tensors in place), then build a wheel from this
# exact dev revision:
#
#   git clone https://github.com/sgl-project/DeepGEMM /scratch/DeepGEMM
#   git -C /scratch/DeepGEMM checkout --detach 0f96f7291e8ee459c725b3ea893611491083d669
#   git -C /scratch/DeepGEMM submodule update --init --recursive
#   docker/inkling2/apply_deep_gemm_patches.sh /scratch/DeepGEMM
#   scripts/build_sgl_deep_gemm.sh 3.12 13.0 /scratch/DeepGEMM
set -euo pipefail

readonly DEEP_GEMM_VERSION="0.2.6+inkling2.0f96f72.1"
readonly SOURCE_REVISION="0f96f7291e8ee459c725b3ea893611491083d669"
readonly PATCH_SHA256="e03cdc481c04a41af506a7272df6e931d1dcf733a5fdddce450cd7512d0652d6"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PATCH_FILE="${SCRIPT_DIR}/deep_gemm.patch"

die() { echo "apply_deep_gemm_patches.sh: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: $0 DEEPGEMM_SOURCE_DIR"
readonly SOURCE_DIR="$(cd -- "$1" && pwd -P)"

[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${SOURCE_REVISION}" ]] || \
  die "source must be exact revision ${SOURCE_REVISION} (v${DEEP_GEMM_VERSION})"
[[ -z "$(git -C "${SOURCE_DIR}" status --porcelain --untracked-files=all)" ]] || \
  die "source has untracked or modified files"

actual="$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"
[[ "${actual}" == "${PATCH_SHA256}" ]] || die "SHA256 mismatch for ${PATCH_FILE}"
git -C "${SOURCE_DIR}" apply --check --whitespace=error-all "${PATCH_FILE}"
git -C "${SOURCE_DIR}" apply --whitespace=error-all "${PATCH_FILE}"
echo "DeepGEMM ${SOURCE_REVISION} patched; build with scripts/build_sgl_deep_gemm.sh"

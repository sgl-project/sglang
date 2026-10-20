#!/usr/bin/env bash
# Apply the residual Muse Spark patch on top of upstream TRT-LLM weight/scale
# sharing (DeepGEMM PR #92), then build a wheel from this exact dev revision:
#
#   git clone https://github.com/sgl-project/DeepGEMM /scratch/DeepGEMM
#   git -C /scratch/DeepGEMM checkout --detach cbe1727eacd9ad704795584ce1421dd1f5a8d67e
#   docker/muse_spark/apply_deep_gemm_patches.sh /scratch/DeepGEMM
#   scripts/build_sgl_deep_gemm.sh 3.12 13.0 /scratch/DeepGEMM
set -euo pipefail

readonly DEEP_GEMM_VERSION="0.2.1.post1+muse.cbe1727"
readonly SOURCE_REVISION="cbe1727eacd9ad704795584ce1421dd1f5a8d67e"
readonly PATCH_SHA256="2add36aa8264c338afc2a9013dd4b3a2dbf73a5b9746dc1ce82915a2fa1fadc1"
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

#!/bin/bash
# Install the standard CUDA CI dependencies plus Kimi-K3's FlashInfer assets.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Source (not bash) so the generic install's Python/venv selection remains
# active while patching the FlashInfer package it just installed.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/ci_install_dependency.sh" "$@"

TRTLLM_GEN_MOE_CUBIN_URL="https://github.com/sgl-project/whl/releases/download/trtllm_gen_moe_cubin_20260617/trtllm_gen_moe_cubin_pool_20260617_v0613rc1.zip"
TRTLLM_GEN_MOE_CUBIN_SHA256="4900501cbe782a76b08a5858f9f07152287b97cb68114466dac286366b66c192"
TRTLLM_GEN_MOE_CUBIN_ARCHIVE_ROOT="trtllm_gen_moe_cubin_pool_20260617_v0613rc1"
export SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL="/opt/trtllm_gen_moe_cubin_pool"

install_required_tools() {
    local missing=()
    local tool
    for tool in patch unzip wget; do
        command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        return
    fi

    apt-get update || true
    apt-get install -y --no-install-recommends "${missing[@]}"
}

cubin_pool_is_valid() {
    [ -d "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" ] &&
        [ "$(find "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" -type f -name '*.cubin' | wc -l)" -eq 1696 ] &&
        [ -f "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}/flashinferMetaInfo.h" ] &&
        [ -d "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}/local" ] &&
        [ -d "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}/overlay/csrc" ]
}

install_trtllm_gen_moe_cubin_pool() (
    if cubin_pool_is_valid; then
        echo "Reusing validated TRT-LLM Gen MoE cubin pool"
        return
    fi

    local cubin_archive
    local cubin_extract_dir
    local extracted_pool
    cubin_archive="$(mktemp /tmp/trtllm_gen_moe_cubin_pool.XXXXXX.zip)"
    cubin_extract_dir="$(mktemp -d /tmp/trtllm_gen_moe_cubin_extract.XXXXXX)"
    extracted_pool="${cubin_extract_dir}/${TRTLLM_GEN_MOE_CUBIN_ARCHIVE_ROOT}"
    trap 'rm -f "${cubin_archive}"; rm -rf "${cubin_extract_dir}"' EXIT

    wget --no-verbose --output-document="${cubin_archive}" \
        "${TRTLLM_GEN_MOE_CUBIN_URL}"
    echo "${TRTLLM_GEN_MOE_CUBIN_SHA256}  ${cubin_archive}" | \
        sha256sum --check --strict -
    unzip -q "${cubin_archive}" -d "${cubin_extract_dir}"
    test "$(find "${extracted_pool}" -type f -name '*.cubin' | wc -l)" -eq 1696

    rm -rf "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}"
    mv "${extracted_pool}" "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}"
    cubin_pool_is_valid
)

apply_flashinfer_dcp_patch() {
    local flashinfer_dcp_patch
    local flashinfer_site_packages
    flashinfer_dcp_patch="${REPO_ROOT}/docker/kimi_k3/flashinfer-perkz-dcp-0.6.15.txt"
    flashinfer_site_packages="$(python3 -c 'from pathlib import Path; import flashinfer; print(Path(flashinfer.__file__).resolve().parent.parent)')"

    sed '/^diff --git a\/tests\//,$d' "${flashinfer_dcp_patch}" | \
        patch --dry-run --batch --forward --strip=1 \
            --directory="${flashinfer_site_packages}"
    sed '/^diff --git a\/tests\//,$d' "${flashinfer_dcp_patch}" | \
        patch --batch --forward --strip=1 \
            --directory="${flashinfer_site_packages}"

    rm -rf /root/.cache/flashinfer /root/.cache/pip
    python3 -c 'import inspect; from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla; assert "enable_dcp" in inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters'
}

apply_transformers_symlink_patch() {
    # transformers 5.12.1 resolves custom-code symlinks out of the HF snapshot
    # and into blobs/, then looks for relative imports by filename in blobs/.
    # Mirror the hot patch in docker/rocm.Dockerfile until upstream PR #46618
    # reaches a transformers release.
    python3 - <<'PY'
import importlib
import pathlib
import tempfile

import transformers.dynamic_module_utils as dynamic_module_utils

marks = ["Path(resolved_module_file).resolve()", "Path(source_file).resolve()"]
path = pathlib.Path(dynamic_module_utils.__file__)
src = path.read_text()
if not any(mark in src for mark in marks):
    print("transformers dynamic_module_utils already fixed; no patch needed")
else:
    patched = (
        src.replace(
            "Path(resolved_module_file).resolve()", "Path(resolved_module_file)"
        ).replace("Path(source_file).resolve()", "Path(source_file)")
    )
    assert patched != src, "FATAL: transformers symlink patch matched nothing"
    path.write_text(patched)
    print("patched transformers dynamic_module_utils.py (symlink hash fix)")

# Exercise the exact HF-cache layout that failed in the B300 Kimi-K3 run.
dynamic_module_utils = importlib.reload(dynamic_module_utils)
with tempfile.TemporaryDirectory() as tmp:
    cache = pathlib.Path(tmp) / "models--org--model"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / "revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "modeling-blob").write_text("from .media_utils import VALUE\n")
    (blobs / "media-blob").write_text("VALUE = 1\n")
    (snapshot / "modeling.py").symlink_to("../../blobs/modeling-blob")
    (snapshot / "media_utils.py").symlink_to("../../blobs/media-blob")
    source_hash = dynamic_module_utils._compute_local_source_files_hash(
        snapshot, snapshot / "modeling.py"
    )
    assert len(source_hash) == 16, source_hash
PY
}

install_required_tools
install_trtllm_gen_moe_cubin_pool
apply_flashinfer_dcp_patch
apply_transformers_symlink_patch

# The install runs in its own shell. Persist the pool path for later workflow
# steps so Kimi-K3 selects the FlashInfer MXFP4 MoE runner instead of failing
# its startup validation.
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL=${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" >> "${GITHUB_ENV}"
fi

#!/bin/bash
# Build the sgl-kernel PPU wheel from source inside the CI container and install
# it over whatever the base image ships.
#
# Called unconditionally by ppu_install_dependency.sh: on PPU we build the
# repo's own kernel sources (dist name sglang-kernel, currently 0.4.6.post1) and
# force-reinstall it over the image's 0.4.3+v0.1.0.ppu2.1.1, so the kernel under
# test matches the source tree of the commit being tested (tested == code).
#
# v0.5.18 note: the compiled kernel no longer lives in a top-level sgl-kernel/
# tree (that was v0.5.13). It moved under python/sglang/kernels/aot/, and its
# build entry point is setup_ppu.py there. Pointing at the old sgl-kernel/ path
# was the reason the former build machinery went dead; this script targets the
# aot tree directly.
#
# Op-surface caveat (measured on the base image, 2026-09): the source 0.4.6 tree
# registers 5 fewer sgl_kernel:: ops than the image's 0.4.3 (bmm_fp8, moe_fused_gate,
# kimi_k2_moe_fused_gate, qserve_w4a8_*). Those are deliberately retired in 0.4.6
# (moe gate/topk routes through the unified Triton router; bmm_fp8 goes through
# flashinfer; qserve is not vendored), NOT a build defect. Overriding therefore
# switches the whole run onto the 0.4.6 generation by design.
#
# Coverage: this script is the single kernel-install entry point for EVERY PPU
# suite, because they all reach it through ppu_install_dependency.sh --
# pr-test-ppu (the three stages a/b/c) and nightly-test-ppu call it directly,
# while the answer / perf / pd-perf / accuracy(evalscope) suites call it from
# their run_*_suite_board.sh. So the source-build override applies uniformly to
# every PPU run, not just the PR gate.
#
# Wheel cache (NAS): a from-scratch compile costs ~200s. We cache the built
# wheel on the shared NAS mount, keyed by the kernel source tree + toolchain, so
# identical sources are compiled at most once ACROSS ALL of the suites above --
# whichever job runs first compiles and backfills, the rest get a ~10s install
# with no compile. Most PRs never touch python/sglang/kernels/aot/, so the common
# case is a cross-workflow cache HIT. See the cache section below for the key
# definition and override knobs.
#
# Prerequisites (all satisfied inside the PPU CI Docker container):
#   - torch with PPU CUDA support (already installed in base image; torch.version.cuda != None)
#   - PPU_SDK at /usr/local/PPU_SDK (baked into image)
#   - ninja / cmake / nvcc (already installed)
#   - git + network reachability for third-party (cutlass/flashinfer/triton);
#     verified reachable from the PPU runner under --network host.
#
# Environment variables (optional):
#   MAX_JOBS                     — ninja parallelism (default: min(nproc*2/3, 32))
#   SGL_KERNEL_THIRD_PARTY_DIR   — reuse a pre-populated third_party dir (e.g. NAS cache)
#                                  to skip the GitHub clone (default: aot build/_deps, cloned fresh)
#   SGL_KERNEL_CUTLASS_DIR       — point at a local cutlass checkout
#   SGL_KERNEL_SKIP_THIRD_PARTY  — set to 1 if third_party is already populated
#   SGL_KERNEL_LOCAL_VERSION     — PEP 440 local segment stamped onto the wheel
#                                  (default: ppu.src.<cache-key>, or ppu.src.g<sha>
#                                  when the cache is disabled). Stamped into the
#                                  checked-out pyproject metadata before the build;
#                                  it keeps the runtime PPU kernel-version guard
#                                  satisfied (that guard keys on a 'ppu' marker in
#                                  the installed version string).
#   SGL_KERNEL_WHEEL_CACHE       — set to 0 to disable the NAS wheel cache
#   SGL_KERNEL_WHEEL_CACHE_DIR   — cache root (default /mnt/wl_nas/cache/sgl-kernel-wheels)
#   PPU_BASE_IMAGE               — declared base-image tag used by the run
#   PPU_BASE_IMAGE_DIGEST        — immutable image digest, preferred when available
#   PPU_SDK_VERSION              — explicit PPU SDK release (for example, 2.2.0)

set -euo pipefail

REPO_ROOT="${GITHUB_WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
KERNEL_DIR="${REPO_ROOT}/python/sglang/kernels/aot"

echo "========================================"
echo "  PPU sgl-kernel: build-or-cache"
echo "========================================"
echo "KERNEL_DIR: ${KERNEL_DIR}"

if [ ! -f "${KERNEL_DIR}/setup_ppu.py" ]; then
    echo "ERROR: ${KERNEL_DIR}/setup_ppu.py not found; the aot kernel tree is missing."
    echo "       (v0.5.18 relocated the kernel from the old top-level sgl-kernel/ path.)"
    exit 1
fi

cd "${KERNEL_DIR}"

# Force-reinstall the given wheel over the image's build and verify the import.
# Returns non-zero on any failure so callers can fall back (e.g. rebuild on a
# corrupt cache entry) instead of aborting the whole run.
install_and_verify() {
    local wheel="$1"
    echo "Installing (override): ${wheel}"
    python3 -m pip install --force-reinstall --no-deps "${wheel}" || return 1
    python3 -c "import sgl_kernel; print(f'sgl_kernel {sgl_kernel.__version__} ready (override)')" || return 1
    return 0
}

# ==================== Wheel cache key ==================== #
# AOT also compiles shared speculative headers from jit/include.
# Prefer the immutable image digest, while retaining the declared tag as a
# fallback for workflows that do not resolve a digest. Image and SDK identity
# are mandatory for cache use: an incomplete key must compile from source
# rather than risk installing a wheel built against another SDK generation.
CACHE_KEY=""
_CACHE_UNAVAILABLE_REASON=""
_IMAGE_IDENTITY=""
_IMAGE_IDENTITY_SOURCE=""
_SDK_VERSION="${PPU_SDK_VERSION:-}"

if [ -n "${PPU_BASE_IMAGE_DIGEST:-}" ]; then
    _IMAGE_IDENTITY="${PPU_BASE_IMAGE_DIGEST}"
    _IMAGE_IDENTITY_SOURCE="digest"
elif [ -n "${PPU_BASE_IMAGE:-}" ]; then
    _IMAGE_IDENTITY="${PPU_BASE_IMAGE}"
    _IMAGE_IDENTITY_SOURCE="tag"
fi

if [ -n "${_IMAGE_IDENTITY}" ]; then
    echo "Cache image identity (${_IMAGE_IDENTITY_SOURCE}): ${_IMAGE_IDENTITY}"
else
    echo "Cache image identity: <missing>"
fi
echo "PPU SDK version: ${_SDK_VERSION:-<missing>}"

if [ "${SGL_KERNEL_WHEEL_CACHE:-1}" != "0" ]; then
    if [ -z "${_IMAGE_IDENTITY}" ]; then
        _CACHE_UNAVAILABLE_REASON="missing image identity"
    elif [ -z "${_SDK_VERSION}" ]; then
        _CACHE_UNAVAILABLE_REASON="missing PPU_SDK_VERSION"
    else
        _AOT_TREE=$(git -C "${REPO_ROOT}" rev-parse --verify "HEAD:python/sglang/kernels/aot" 2>/dev/null || echo "")
        _JIT_INCLUDE_TREE=$(git -C "${REPO_ROOT}" rev-parse --verify "HEAD:python/sglang/kernels/jit/include" 2>/dev/null || echo "")
        if [ -z "${_AOT_TREE}" ]; then
            _CACHE_UNAVAILABLE_REASON="missing AOT git tree"
        elif [ -z "${_JIT_INCLUDE_TREE}" ]; then
            _CACHE_UNAVAILABLE_REASON="missing JIT include git tree"
        else
            _TOOLCHAIN=$(python3 -c 'import sys
try:
    import torch
    tv, cu = torch.__version__, torch.version.cuda
except Exception:
    tv = cu = "none"
print(f"{tv}|{cu}|cp{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || echo "")
            if [ -z "${_TOOLCHAIN}" ]; then
                _CACHE_UNAVAILABLE_REASON="missing toolchain fingerprint"
            else
                CACHE_KEY=$(printf '%s|%s|%s|%s|%s' \
                    "${_AOT_TREE}" "${_JIT_INCLUDE_TREE}" "${_TOOLCHAIN}" \
                    "${_IMAGE_IDENTITY}" "${_SDK_VERSION}" | sha256sum | cut -c1-16)
                echo "Wheel cache key: ${CACHE_KEY}"
            fi
        fi
    fi
fi

WHEEL_CACHE_DIR="${SGL_KERNEL_WHEEL_CACHE_DIR:-/mnt/wl_nas/cache/sgl-kernel-wheels}"
KEY_DIR=""
if [ -n "${CACHE_KEY}" ]; then
    KEY_DIR="${WHEEL_CACHE_DIR}/${CACHE_KEY}"
fi

# Stamp a PEP 440 local version segment so the resulting wheel is identifiable
# as a PPU source-build and satisfies the runtime PPU kernel-version guard. The
# cache key keeps HIT and MISS versions identical; unavailable cache identity
# falls back to the tested commit.
if [ -z "${SGL_KERNEL_LOCAL_VERSION:-}" ]; then
    if [ -n "${CACHE_KEY}" ]; then
        SGL_KERNEL_LOCAL_VERSION="ppu.src.${CACHE_KEY}"
    else
        _SHA=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "")
        if [ -n "${_SHA}" ]; then
            SGL_KERNEL_LOCAL_VERSION="ppu.src.g${_SHA}"
        else
            SGL_KERNEL_LOCAL_VERSION="ppu.src"
        fi
    fi
fi
export SGL_KERNEL_LOCAL_VERSION
echo "SGL_KERNEL_LOCAL_VERSION: ${SGL_KERNEL_LOCAL_VERSION}"

# ==================== Fast path: cache HIT ==================== #
if [ -n "${KEY_DIR}" ]; then
    # An empty cache dir makes the glob match nothing and ls exit non-zero;
    # the '|| true' keeps that out of set -e / pipefail so a MISS is not fatal.
    CACHED_WHEEL=$( (ls -t "${KEY_DIR}"/sglang_kernel-*.whl 2>/dev/null || true) | head -1)
    if [ -n "${CACHED_WHEEL}" ] && [ -r "${CACHED_WHEEL}" ]; then
        echo "Cache HIT (key ${CACHE_KEY}): ${CACHED_WHEEL}"
        if install_and_verify "${CACHED_WHEEL}"; then
            echo "========================================"
            echo "  PPU sgl-kernel installed from cache (no compile)"
            echo "========================================"
            exit 0
        fi
        echo "WARNING: cached wheel failed to install; rebuilding from source."
    else
        echo "Cache MISS (key ${CACHE_KEY}); building from source."
    fi
elif [ "${SGL_KERNEL_WHEEL_CACHE:-1}" = "0" ]; then
    echo "Wheel cache disabled (SGL_KERNEL_WHEEL_CACHE=0); building from source."
else
    echo "Wheel cache unavailable (${_CACHE_UNAVAILABLE_REASON:-incomplete identity}); building from source."
fi

# ==================== Slow path: build from source ==================== #
# Parallelism: be conservative on shared runners (default 2/3 of cores, cap 32)
if [ -z "${MAX_JOBS:-}" ]; then
    MAX_JOBS=$(python3 -c "import os; print(min(os.cpu_count() * 2 // 3, 32))")
fi
export MAX_JOBS

# Generic CUDA metadata pins a different torch than the PPU SDK provides.
cp "${KERNEL_DIR}/pyproject_ppu.toml" "${KERNEL_DIR}/pyproject.toml"

# setuptools reads [project].version from pyproject.toml; a value passed to
# setup() is ignored when that field is static. So stamp the local segment
# directly into the checked-out project metadata (an ephemeral CI checkout --
# not committed). Idempotent: the [0-9][^"+]* match skips a version that already
# carries a local segment. Both files are patched so setup_ppu.py's own
# _get_version() (which reads pyproject_ppu.toml) stays consistent.
for _pp in pyproject.toml pyproject_ppu.toml; do
    [ -f "${_pp}" ] || continue
    sed -i -E "s/^(version = \"[0-9][^\"+]*)\"/\1+${SGL_KERNEL_LOCAL_VERSION}\"/" "${_pp}"
    echo "  ${_pp}: $(grep -E '^version = ' "${_pp}" | head -1)"
done

# Build the wheel
echo "Building wheel (MAX_JOBS=${MAX_JOBS})..."
BUILD_START=$(date +%s)

if python3 setup_ppu.py bdist_wheel 2>&1 | tee /tmp/sgl_kernel_build.log | \
    grep -E "^(Building|running|creating|Cloning|copying|nvcc|error:|warning:.*error)"; then
    BUILD_RC=0
else
    BUILD_RC=${PIPESTATUS[0]}
fi

BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

if [ "${BUILD_RC}" -ne 0 ]; then
    echo "ERROR: setup_ppu.py failed (exit ${BUILD_RC}) after ${BUILD_ELAPSED}s. Full log:"
    cat /tmp/sgl_kernel_build.log
    exit "${BUILD_RC}"
fi
echo "Build completed in ${BUILD_ELAPSED}s"

# Find the wheel ('|| true' so a missing wheel yields the explicit error below
# instead of a bare set -e / pipefail abort on ls's non-zero exit).
WHEEL=$( (ls -t dist/sglang_kernel-*.whl 2>/dev/null || true) | head -1)
if [ -z "${WHEEL}" ]; then
    echo "ERROR: No wheel produced. Full build log:"
    cat /tmp/sgl_kernel_build.log
    exit 1
fi

# Install it over the image's build
if ! install_and_verify "${WHEEL}"; then
    echo "ERROR: freshly built wheel failed to install: ${WHEEL}"
    exit 1
fi

# ==================== Cache backfill (best-effort) ==================== #
# Atomic publish: copy into a temp file on the cache filesystem, then rename it
# into place so concurrent readers never observe a half-written wheel. All
# failures here are non-fatal -- the run already has the wheel installed.
if [ -n "${KEY_DIR}" ]; then
    if mkdir -p "${KEY_DIR}" 2>/dev/null; then
        _tmp=$(mktemp "${KEY_DIR}/.tmp.XXXXXX" 2>/dev/null || echo "")
        if [ -n "${_tmp}" ] && cp "${WHEEL}" "${_tmp}" && \
           mv -f "${_tmp}" "${KEY_DIR}/$(basename "${WHEEL}")"; then
            echo "Cache backfill: ${KEY_DIR}/$(basename "${WHEEL}")"
        else
            [ -n "${_tmp}" ] && rm -f "${_tmp}" 2>/dev/null || :
            echo "WARNING: cache backfill failed (non-fatal)."
        fi
    else
        echo "WARNING: cannot create cache dir ${KEY_DIR} (non-fatal)."
    fi
fi

echo "========================================"
echo "  PPU sgl-kernel build SUCCESS (${BUILD_ELAPSED}s)"
echo "========================================"

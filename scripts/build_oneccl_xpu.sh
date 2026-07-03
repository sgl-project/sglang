#!/bin/bash
# =============================================================================
# build_oneccl_xpu.sh — Build oneCCL-v2 with XPU (SYCL/dpcpp) support
#
# Produces the SYCL-built libccl.so.1 that SGLang's PyXcclCommunicator loads
# (via SGLANG_PYXCCL_SO_PATH) to call Intel oneCCL directly for XPU tensor-
# parallel collectives. See docs/platforms/xpu.pyxccl.md.
#
# USAGE
#   ./build_oneccl_xpu.sh [OPTIONS]
#
# OPTIONS
#   -j N            Parallel jobs (default: nproc)
#   --build-dir DIR Build directory (default: <repo>/build-xpu-release)
#   --debug         Debug build instead of Release
#   --install       Run cmake --install after build
#   --clean         Wipe build directory before configure
#   --with-tests    Also build functional/regression tests (slower)
#   --with-examples Also build examples (slower)
#   --device-code MODE
#                   SYCL device-code-split: per_kernel|per_source|off
#                   default: per_kernel (best for deployment)
#   -h, --help      Show this message
#
# WHAT IT BUILDS
#   libccl.so.2  — oneCCL v2 XPU library (SYCL/dpcpp, via icpx)
#   libccl.so.1  — legacy CCL shim (XPU path, links libccl.so.2)
#   libccl.so    — symlink -> libccl.so.2
#
#   CPU plugin is disabled by default (XPU-only build).
#
# COMPILER
#   Activates Intel oneAPI compiler via: source /work/compiler/setvars.sh
#
# ENVIRONMENT PREREQUISITES (first run only)
#   1. System packages:
#        apt-get install -y gcc g++ libopenmpi-dev
#   2. Build-tools uv environment (cmake + ninja):
#        uv venv /work/.build-env --python 3.12
#        uv pip install --python /work/.build-env/bin/python cmake ninja
#
# SOURCE PATCHES APPLIED TO THE oneCCL-v2 REPO
#   1. CMakeLists.txt (top-level):
#        LIBCCL_CMAKE_ARGS gets -DCMAKE_CXX_STANDARD=17 so the inner
#        ExternalProject's FindIntelSYCL_level_zero.cmake check_cxx_compiler_flag
#        test for -fsycl compiles with C++17 (SYCL requires C++17).
#   2. deps/libccl/CMakeLists.txt:
#        set(CMAKE_CXX_STANDARD 11) guarded with if(NOT CMAKE_CXX_STANDARD)
#        so the dpcpp -DCMAKE_CXX_STANDARD=17 override is not clobbered.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO_ROOT="${ONECCL_REPO_ROOT:-/work/libraries.performance.communication.oneccl-v2}"
BUILD_TYPE="Release"
BUILD_DIR=""          # resolved below after parsing args
JOBS="$(nproc)"
DO_INSTALL=0
DO_CLEAN=0
BUILD_TESTS=0
BUILD_EXAMPLES=0
DEVICE_CODE="per_kernel"
BUILD_CPU_PLUGIN=OFF
SETVARS="${ONEAPI_SETVARS:-/work/compiler/setvars.sh}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    sed -n '/^# USAGE/,/^# =/p' "${BASH_SOURCE[0]}" | head -n -1 | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j)          JOBS="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --debug)     BUILD_TYPE="Debug"; shift ;;
        --install)   DO_INSTALL=1; shift ;;
        --clean)     DO_CLEAN=1; shift ;;
        --with-tests)    BUILD_TESTS=1; shift ;;
        --with-examples) BUILD_EXAMPLES=1; shift ;;
        --device-code)   DEVICE_CODE="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Default build dir: inside the repo
[[ -z "$BUILD_DIR" ]] && BUILD_DIR="${REPO_ROOT}/build-xpu-$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
INSTALL_DIR="${BUILD_DIR}/_install"

# ---------------------------------------------------------------------------
# Activate compiler
# ---------------------------------------------------------------------------
echo "[build_oneccl_xpu] Activating Intel oneAPI compiler..."
# setvars.sh references variables that may be unset; temporarily relax -u/-e
set +euo pipefail
# shellcheck source=/dev/null
source "$SETVARS"
set -euo pipefail

# Locate cmake — prefer the build-tools venv, then fall back to known locations
CMAKE_BIN=""
for _cmake_candidate in \
    /work/.build-env/lib/python3.12/site-packages/cmake/data/bin/cmake \
    /work/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake \
    /usr/local/bin/cmake /usr/bin/cmake; do
    if [[ -x "$_cmake_candidate" ]]; then
        CMAKE_BIN="$_cmake_candidate"
        break
    fi
done

if [[ -z "$CMAKE_BIN" ]]; then
    echo "[build_oneccl_xpu] ERROR: cmake not found. Install it via:" >&2
    echo "  uv pip install --python /work/.build-env/bin/python cmake ninja" >&2
    exit 1
fi
echo "[build_oneccl_xpu] cmake: $("$CMAKE_BIN" --version | head -1)  [$CMAKE_BIN]"

# Locate ninja — prefer the build-tools venv
NINJA_BIN=""
for _ninja_candidate in \
    /work/.build-env/bin/ninja \
    /work/.venv/bin/ninja \
    /usr/local/bin/ninja /usr/bin/ninja; do
    if [[ -x "$_ninja_candidate" ]]; then
        NINJA_BIN="$_ninja_candidate"
        break
    fi
done

if [[ -n "$NINJA_BIN" ]]; then
    CMAKE_GENERATOR="-GNinja -DCMAKE_MAKE_PROGRAM=$NINJA_BIN"
    # Also add to PATH so ExternalProject sub-builds inherit it
    export PATH="$(dirname "$NINJA_BIN"):$PATH"
    echo "[build_oneccl_xpu] ninja: $("$NINJA_BIN" --version)  [$NINJA_BIN]"
else
    CMAKE_GENERATOR=""
    echo "[build_oneccl_xpu] WARNING: ninja not found, falling back to Unix Makefiles"
fi

# Verify icpx is available
if ! command -v icpx &>/dev/null; then
    echo "[build_oneccl_xpu] ERROR: icpx not found after sourcing $SETVARS" >&2
    exit 1
fi
ICPX_VER=$(icpx --version 2>&1 | head -1)
echo "[build_oneccl_xpu] Compiler: $ICPX_VER"

# Ensure the Intel compiler's bundled linker (ld.lld) is on PATH so that
# icpx can resolve 'ld' when linking. The oneAPI package ships ld.lld in
# compiler/latest/bin/compiler/ but does not always install a system 'ld'.
INTEL_COMPILER_BIN="$(dirname "$(command -v icpx)")"/compiler
if [[ -x "$INTEL_COMPILER_BIN/ld.lld" ]]; then
    # Create a temp dir with an 'ld' symlink so the system linker path resolves
    _LD_SHIM_DIR="$(mktemp -d)"
    ln -sf "$INTEL_COMPILER_BIN/ld.lld" "$_LD_SHIM_DIR/ld"
    export PATH="$INTEL_COMPILER_BIN:$_LD_SHIM_DIR:$PATH"
    echo "[build_oneccl_xpu] linker: $INTEL_COMPILER_BIN/ld.lld (shimmed as ld)"
fi

# ---------------------------------------------------------------------------
# Initialize git submodules (idempotent)
# ---------------------------------------------------------------------------
echo "[build_oneccl_xpu] Checking git submodules..."
cd "$REPO_ROOT"
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# Optional clean
# ---------------------------------------------------------------------------
if [[ $DO_CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "[build_oneccl_xpu] Cleaning build dir: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------------------
# cmake configure
# ---------------------------------------------------------------------------
ON_OFF_TESTS=$([ "$BUILD_TESTS" -eq 1 ] && echo ON || echo OFF)
ON_OFF_EXAMPLES=$([ "$BUILD_EXAMPLES" -eq 1 ] && echo ON || echo OFF)

echo "[build_oneccl_xpu] Configuring..."
echo "[build_oneccl_xpu]   BUILD_TYPE    = $BUILD_TYPE"
echo "[build_oneccl_xpu]   DEVICE_CODE   = $DEVICE_CODE"
echo "[build_oneccl_xpu]   CPU_PLUGIN    = $BUILD_CPU_PLUGIN"
echo "[build_oneccl_xpu]   TESTS         = $ON_OFF_TESTS"
echo "[build_oneccl_xpu]   EXAMPLES      = $ON_OFF_EXAMPLES"
echo "[build_oneccl_xpu]   BUILD_DIR     = $BUILD_DIR"
echo "[build_oneccl_xpu]   INSTALL_DIR   = $INSTALL_DIR"
echo "[build_oneccl_xpu]   JOBS          = $JOBS"

"$CMAKE_BIN" \
    ${CMAKE_GENERATOR} \
    -S "$REPO_ROOT" \
    -B "$BUILD_DIR" \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DONECCL_LIBCCL_BUILD_FT="$ON_OFF_TESTS" \
    -DONECCL_LIBCCL_BUILD_REG_TESTS="$ON_OFF_TESTS" \
    -DONECCL_LIBCCL_BUILD_EXAMPLES="$ON_OFF_EXAMPLES" \
    -DONECCL_BUILD_TESTS="$ON_OFF_TESTS" \
    -DONECCL_LIBCCL_SYCL_DEVICE_CODE_SPLIT="$DEVICE_CODE" \
    -DONECCL_LIBCCL_EXTERN_TEMPLATE_INST=ON \
    -DONECCL_BUILD_CPU_PLUGIN="$BUILD_CPU_PLUGIN" \
    -DONECCL_ENABLE_ITT=ON

# Sanity check: ensure SYCL support was detected
if grep -q '^COMPILER_SUPPORTS_SYCL:INTERNAL=FALSE' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    echo "[build_oneccl_xpu] ERROR: COMPILER_SUPPORTS_SYCL=FALSE — icpx does not support -fsycl!" >&2
    echo "[build_oneccl_xpu] Check that setvars.sh was sourced correctly and icpx supports SYCL." >&2
    exit 1
fi
echo "[build_oneccl_xpu] SYCL support confirmed"

# ---------------------------------------------------------------------------
# cmake build
# ---------------------------------------------------------------------------
echo "[build_oneccl_xpu] Building (jobs=$JOBS)..."
"$CMAKE_BIN" --build "$BUILD_DIR" --parallel "$JOBS"

# ---------------------------------------------------------------------------
# cmake install (optional)
# ---------------------------------------------------------------------------
if [[ $DO_INSTALL -eq 1 ]]; then
    echo "[build_oneccl_xpu] Installing to $INSTALL_DIR..."
    "$CMAKE_BIN" --install "$BUILD_DIR"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[build_oneccl_xpu] ========================================="
echo "[build_oneccl_xpu] Build complete!"
echo "[build_oneccl_xpu]   Build dir:   $BUILD_DIR"
if [[ $DO_INSTALL -eq 1 ]]; then
    echo "[build_oneccl_xpu]   Install dir: $INSTALL_DIR"
fi
echo ""
echo "[build_oneccl_xpu] Key outputs (build dir):"
find "$BUILD_DIR" -maxdepth 4 \( -name "libccl.so*" -o -name "libccl.so.2*" \) \
    ! -name "*.cmake" 2>/dev/null | sort | sed 's/^/  /'
echo ""
echo "[build_oneccl_xpu] To use in your environment:"
echo "  export CCL_ROOT=$INSTALL_DIR"
echo "  export LD_LIBRARY_PATH=\$CCL_ROOT/lib:\$LD_LIBRARY_PATH"
echo "  export SGLANG_PYXCCL_SO_PATH=\$CCL_ROOT/lib/libccl.so.1"
echo "[build_oneccl_xpu] ========================================="

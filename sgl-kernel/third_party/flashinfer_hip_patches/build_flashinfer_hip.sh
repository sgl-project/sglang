#!/bin/bash
# Build FlashInfer with HIP/ROCm support
# Usage: ./build_flashinfer_hip.sh [--install] [--clean]
#
# This script:
# 1. Clones FlashInfer (or uses existing clone)
# 2. Applies HIP compatibility patches
# 3. Builds and optionally installs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-/tmp/flashinfer-hip-build}"
FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/flashinfer-ai/flashinfer.git}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-main}"

# Parse arguments
INSTALL=0
CLEAN=0
for arg in "$@"; do
    case $arg in
        --install) INSTALL=1 ;;
        --clean) CLEAN=1 ;;
        --help|-h)
            echo "Usage: $0 [--install] [--clean]"
            echo "  --install  Install FlashInfer after building"
            echo "  --clean    Remove existing build and start fresh"
            exit 0
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" -eq 1 ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build..."
    rm -rf "$BUILD_DIR"
fi

# Clone or update FlashInfer
if [ ! -d "$BUILD_DIR" ]; then
    echo "Cloning FlashInfer..."
    git clone "$FLASHINFER_REPO" "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Checkout specific version and reset
echo "Checking out FlashInfer version: $FLASHINFER_VERSION"
git fetch origin
git checkout "$FLASHINFER_VERSION"
git reset --hard "origin/$FLASHINFER_VERSION" 2>/dev/null || git reset --hard "$FLASHINFER_VERSION"

# Check for patches
PATCH_COUNT=$(ls -1 "$PATCHES_DIR"/*.patch 2>/dev/null | wc -l)
if [ "$PATCH_COUNT" -eq 0 ]; then
    echo "Error: No patches found in $PATCHES_DIR"
    exit 1
fi

# Apply patches
echo ""
echo "Applying $PATCH_COUNT HIP patches..."

# First try git am (preserves commits)
if git am "$PATCHES_DIR"/*.patch 2>/dev/null; then
    echo "Patches applied successfully with git am"
else
    # Fall back to git apply
    git am --abort 2>/dev/null || true
    
    # Check if patches apply cleanly
    if git apply --check "$PATCHES_DIR"/*.patch 2>/dev/null; then
        git apply "$PATCHES_DIR"/*.patch
        echo "Patches applied successfully with git apply"
    else
        echo ""
        echo "Warning: Patches don't apply cleanly. Trying with --3way..."
        if git apply --3way "$PATCHES_DIR"/*.patch 2>/dev/null; then
            echo "Patches applied with 3-way merge"
        else
            echo "Error: Failed to apply patches. Manual intervention required."
            echo "Conflicts in:"
            git apply --check "$PATCHES_DIR"/*.patch 2>&1 || true
            exit 1
        fi
    fi
fi

# Initialize submodules if needed
if [ -f ".gitmodules" ]; then
    echo ""
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

# Set up environment for HIP build
export HIP_PLATFORM=amd
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Detect GPU architecture
if [ -z "$AMDGPU_TARGET" ]; then
    if command -v rocm_agent_enumerator &>/dev/null; then
        AMDGPU_TARGET=$(rocm_agent_enumerator | grep gfx | head -1)
    fi
    AMDGPU_TARGET="${AMDGPU_TARGET:-gfx942}"
fi
export AMDGPU_TARGET
echo ""
echo "Building for GPU architecture: $AMDGPU_TARGET"

# Build
echo ""
echo "Building FlashInfer..."
if [ "$INSTALL" -eq 1 ]; then
    pip install . --no-build-isolation -v
    echo ""
    echo "FlashInfer installed successfully!"
else
    pip wheel . --no-build-isolation -w dist/
    echo ""
    echo "FlashInfer wheel built in: $BUILD_DIR/dist/"
    ls -la dist/*.whl
fi

echo ""
echo "Done!"

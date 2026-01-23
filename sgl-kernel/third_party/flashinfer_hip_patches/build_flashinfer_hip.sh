#!/bin/bash
# Install FlashInfer from AMD-compatible fork
#
# Usage:
#   ./build_flashinfer_hip.sh           # Install from fork
#   ./build_flashinfer_hip.sh --dev     # Install in editable mode
#
set -e

FLASHINFER_FORK="https://github.com/sunxxuns/flashinfer.git"
FLASHINFER_BRANCH="${FLASHINFER_BRANCH:-main}"
BUILD_DIR="${BUILD_DIR:-/tmp/flashinfer-hip-build}"

echo "=== Installing FlashInfer from AMD fork ==="
echo "Fork: $FLASHINFER_FORK"
echo "Branch: $FLASHINFER_BRANCH"

# Check if already installed
if python -c "import flashinfer; print(f'FlashInfer {flashinfer.__version__} already installed')" 2>/dev/null; then
    read -p "FlashInfer already installed. Reinstall? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Clone or update
if [ -d "$BUILD_DIR" ]; then
    echo "Updating existing clone..."
    cd "$BUILD_DIR"
    git fetch origin
    git checkout "$FLASHINFER_BRANCH"
    git pull origin "$FLASHINFER_BRANCH"
else
    echo "Cloning fork..."
    git clone --branch "$FLASHINFER_BRANCH" "$FLASHINFER_FORK" "$BUILD_DIR"
    cd "$BUILD_DIR"
fi

# Install
if [ "$1" == "--dev" ]; then
    echo "Installing in editable mode..."
    pip install -e . -v
else
    echo "Installing..."
    pip install . -v
fi

echo ""
echo "=== Done! ==="
python -c "import flashinfer; print(f'FlashInfer {flashinfer.__version__} installed successfully')"

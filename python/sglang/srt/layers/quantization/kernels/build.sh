#!/bin/bash
# Build script for MXFP4 grouped kernels with backend selection

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
BACKEND="stub"
BUILD_TYPE="Release"
CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --cutlass)
            BACKEND="cutlass"
            shift
            ;;
        --cutlass-advanced)
            BACKEND="cutlass-advanced"
            shift
            ;;
        --flashinfer)
            BACKEND="flashinfer"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--cutlass|--cutlass-advanced|--flashinfer] [--debug] [--clean]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Building MXFP4 Grouped GEMM Kernels${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "Backend: ${YELLOW}${BACKEND}${NC}"
echo -e "Build type: ${YELLOW}${BUILD_TYPE}${NC}"
echo ""

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf build _mxfp4_kernels*.so
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found. Please install CMake first.${NC}"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake based on backend
echo -e "${GREEN}Configuring with CMake...${NC}"
CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

case $BACKEND in
    cutlass)
        CMAKE_FLAGS="${CMAKE_FLAGS} -DUSE_CUTLASS_FP4=ON"
        echo -e "${YELLOW}Note: Make sure CUTLASS is installed or in third_party/${NC}"
        ;;
    cutlass-advanced)
        CMAKE_FLAGS="${CMAKE_FLAGS} -DUSE_CUTLASS_ADVANCED=ON"
        echo -e "${YELLOW}Note: Using advanced CUTLASS implementation with tuned tiles${NC}"
        ;;
    flashinfer)
        CMAKE_FLAGS="${CMAKE_FLAGS} -DUSE_FLASHINFER_BACKEND=ON"
        echo -e "${YELLOW}Note: Make sure FlashInfer is installed${NC}"
        ;;
    *)
        echo -e "${YELLOW}Using stub implementation (build sanity only)${NC}"
        ;;
esac

cmake ${CMAKE_FLAGS} ..

# Build
echo -e "${GREEN}Building...${NC}"
cmake --build . -j$(nproc)

# Copy the built module
if [ -f _mxfp4_kernels.so ]; then
    cp _mxfp4_kernels.so ../
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "  Module location: ${YELLOW}$(pwd)/../_mxfp4_kernels.so${NC}"
else
    echo -e "${RED}✗ Build failed - module not found${NC}"
    exit 1
fi

cd ..

# Test import
echo ""
echo -e "${GREEN}Testing module import...${NC}"
if python3 -c "import _mxfp4_kernels; print('✓ Module imported successfully')" 2>/dev/null; then
    echo -e "${GREEN}✓ Import test passed${NC}"
else
    echo -e "${YELLOW}⚠ Import test failed - this is expected if not in PYTHONPATH${NC}"
    echo -e "  Add this directory to PYTHONPATH or copy the .so file next to your Python code"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. Copy _mxfp4_kernels.so to your Python module directory"
echo "2. Or add $(pwd) to PYTHONPATH"
echo "3. Test with: python3 -c 'import _mxfp4_kernels'"
echo ""

if [ "$BACKEND" == "cutlass" ] || [ "$BACKEND" == "cutlass-advanced" ]; then
    echo -e "${YELLOW}Performance tip for CUTLASS:${NC}"
    echo "  - Profile with: nsys profile --stats=true python your_script.py"
    echo "  - Look for high TMA utilization and low bank conflicts"
    echo "  - Tune TileShape if needed (current: 128x256x64)"
fi
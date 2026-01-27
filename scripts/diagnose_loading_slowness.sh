#!/bin/bash
# Diagnose Model Loading Slowness - Disk Speed Test
# Compare disk read speeds on MI325 vs MI35x
#
# Usage:
#   bash scripts/diagnose_loading_slowness.sh

set -e

echo "========================================"
echo "Disk Speed Diagnosis for Model Loading"
echo "========================================"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo ""

# GPU Architecture
echo "=== GPU Architecture ==="
if command -v rocminfo &> /dev/null; then
    rocminfo 2>/dev/null | grep -E "gfx[0-9]+" | head -1 || echo "Unknown"
fi
echo ""

# Storage Info
echo "=== Storage Info ==="
df -h 2>/dev/null | grep -E "Filesystem|overlay|sgl-data|home" || df -h
echo ""
lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null | head -15 || echo "lsblk not available"
echo ""

# Find model weights location
echo "=== Finding Model Weights ==="

# Common paths to search
SEARCH_PATHS=(
    "/sgl-data/hf-cache/hub"
    "/home/runner/sgl-data/hf-cache/hub"
    "${HF_HOME:-}/hub"
    "${HOME}/.cache/huggingface/hub"
    "/root/.cache/huggingface/hub"
)

MODEL_DIR=""
SAFETENSOR_FILE=""

# Look for DeepSeek-V3.2 or Kimi-K2
for path in "${SEARCH_PATHS[@]}"; do
    if [ -z "$path" ]; then continue; fi

    # DeepSeek-V3.2
    dsv32="$path/models--deepseek-ai--DeepSeek-V3.2"
    if [ -d "$dsv32" ]; then
        echo "Found DeepSeek-V3.2 at: $dsv32"
        MODEL_DIR="$dsv32"
        SAFETENSOR_FILE=$(find "$dsv32/snapshots" -name "*.safetensors" -type f 2>/dev/null | head -1)
        break
    fi

    # Kimi-K2
    kimi="$path/models--moonshotai--Kimi-K2-Instruct"
    if [ -d "$kimi" ]; then
        echo "Found Kimi-K2-Instruct at: $kimi"
        MODEL_DIR="$kimi"
        SAFETENSOR_FILE=$(find "$kimi/snapshots" -name "*.safetensors" -type f 2>/dev/null | head -1)
        break
    fi

    # Also check variant names
    kimi2="$path/models--moonshotai--Kimi-K2-Instruct-0905"
    if [ -d "$kimi2" ]; then
        echo "Found Kimi-K2-Instruct-0905 at: $kimi2"
        MODEL_DIR="$kimi2"
        SAFETENSOR_FILE=$(find "$kimi2/snapshots" -name "*.safetensors" -type f 2>/dev/null | head -1)
        break
    fi
done

# Also try find command as fallback
if [ -z "$SAFETENSOR_FILE" ]; then
    echo "Searching for any safetensor files..."
    SAFETENSOR_FILE=$(find /sgl-data /home/runner -name "*deepseek*.safetensors" -o -name "*kimi*.safetensors" 2>/dev/null | head -1)
fi

if [ -z "$SAFETENSOR_FILE" ]; then
    echo "Searching for any large safetensor files..."
    SAFETENSOR_FILE=$(find /sgl-data /home/runner /root -name "*.safetensors" -size +100M 2>/dev/null | head -1)
fi

echo ""

# Disk Speed Tests
echo "=== Disk Speed Tests ==="

# Determine test directory
if [ -d "/sgl-data" ]; then
    TEST_DIR="/sgl-data"
elif [ -d "/home/runner" ]; then
    TEST_DIR="/home/runner"
else
    TEST_DIR="/tmp"
fi

TEST_FILE="$TEST_DIR/disk_test_$$"
echo "Test directory: $TEST_DIR"
echo ""

# Write test (10GB)
echo "--- Sequential Write (10GB) ---"
sync
WRITE_OUT=$(dd if=/dev/zero of="$TEST_FILE" bs=1M count=10240 conv=fdatasync 2>&1)
echo "$WRITE_OUT" | tail -1
rm -f "$TEST_FILE"
echo ""

# Create test file for read
echo "Creating 10GB test file..."
dd if=/dev/zero of="$TEST_FILE" bs=1M count=10240 2>/dev/null

# Drop cache before read test
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "(Cannot drop cache)"

# Read test (10GB)
echo "--- Sequential Read (10GB) ---"
READ_OUT=$(dd if="$TEST_FILE" of=/dev/null bs=1M 2>&1)
echo "$READ_OUT" | tail -1
rm -f "$TEST_FILE"
echo ""

# Test actual model file if found
if [ -n "$SAFETENSOR_FILE" ] && [ -f "$SAFETENSOR_FILE" ]; then
    echo "=== Model File Read Test ==="
    FILE_SIZE=$(stat -c%s "$SAFETENSOR_FILE" 2>/dev/null || stat -f%z "$SAFETENSOR_FILE" 2>/dev/null || echo "0")
    FILE_SIZE_GB=$(awk "BEGIN {printf \"%.2f\", $FILE_SIZE/1024/1024/1024}")
    echo "File: $(basename "$SAFETENSOR_FILE")"
    echo "Size: ${FILE_SIZE_GB} GB"
    echo "Path: $SAFETENSOR_FILE"
    echo ""

    # Drop cache
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

    echo "--- Reading model file ---"
    MODEL_OUT=$(dd if="$SAFETENSOR_FILE" of=/dev/null bs=1M 2>&1)
    echo "$MODEL_OUT" | tail -1
else
    echo "=== Model File Read Test ==="
    echo "No model files found to test."
    echo "Models may not be cached on this machine."
fi
echo ""

# List what's actually in common paths
echo "=== Available Cache Contents ==="
for path in "/sgl-data" "/sgl-data/hf-cache" "/home/runner/sgl-data"; do
    if [ -d "$path" ]; then
        echo "Contents of $path:"
        ls -la "$path" 2>/dev/null | head -10
        echo ""
    fi
done

echo "========================================"
echo "Diagnosis Complete"
echo "========================================"

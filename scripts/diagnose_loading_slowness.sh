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

MODEL_DIR=""
SAFETENSOR_FILE=""

# Search for DeepSeek-V3.2 directory
echo "Searching for DeepSeek-V3.2..."
DSV32_DIR=$(find / -type d -iname "*deepseek-v3.2*" 2>/dev/null | head -1)
if [ -n "$DSV32_DIR" ]; then
    echo "Found DeepSeek-V3.2 at: $DSV32_DIR"
    MODEL_DIR="$DSV32_DIR"
    SAFETENSOR_FILE=$(find "$DSV32_DIR" -name "*.safetensors" -type f 2>/dev/null | head -1)
fi

# If not found, search for Kimi-K2
if [ -z "$SAFETENSOR_FILE" ]; then
    echo "Searching for Kimi-K2..."
    KIMI_DIR=$(find / -type d -iname "*kimi-k2*" 2>/dev/null | head -1)
    if [ -n "$KIMI_DIR" ]; then
        echo "Found Kimi-K2 at: $KIMI_DIR"
        MODEL_DIR="$KIMI_DIR"
        SAFETENSOR_FILE=$(find "$KIMI_DIR" -name "*.safetensors" -type f 2>/dev/null | head -1)
    fi
fi

# If still not found, search for any large safetensor file
if [ -z "$SAFETENSOR_FILE" ]; then
    echo "Searching for any large safetensor files (>1GB)..."
    SAFETENSOR_FILE=$(find / -name "*.safetensors" -size +1G -type f 2>/dev/null | head -1)
fi

# Show what we found and disk info
if [ -n "$SAFETENSOR_FILE" ]; then
    echo "Found safetensor file: $SAFETENSOR_FILE"
    echo ""

    # Get mount point and disk info for this file
    echo "=== Disk Info for Model File ==="
    FILE_DIR=$(dirname "$SAFETENSOR_FILE")

    # Get mount point
    MOUNT_INFO=$(df "$SAFETENSOR_FILE" 2>/dev/null | tail -1)
    echo "Mount info: $MOUNT_INFO"

    # Extract device name
    DEVICE=$(echo "$MOUNT_INFO" | awk '{print $1}')
    MOUNT_POINT=$(echo "$MOUNT_INFO" | awk '{print $NF}')
    echo "Device: $DEVICE"
    echo "Mount point: $MOUNT_POINT"

    # Get disk details using lsblk
    if command -v lsblk &> /dev/null; then
        # Extract base device name (e.g., nvme0n1 from /dev/nvme0n1p1)
        BASE_DEVICE=$(echo "$DEVICE" | sed 's|/dev/||' | sed 's/p[0-9]*$//' | sed 's/[0-9]*$//')
        echo ""
        echo "Disk details:"
        lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,TRAN,SCHED "/dev/$BASE_DEVICE" 2>/dev/null || \
        lsblk -o NAME,TYPE,SIZE,ROTA,MODEL "/dev/$BASE_DEVICE" 2>/dev/null || \
        lsblk -d -o NAME,TYPE,SIZE,ROTA,MODEL 2>/dev/null | grep -E "NAME|$BASE_DEVICE"

        # Show if it's SSD or HDD
        ROTA=$(lsblk -d -n -o ROTA "/dev/$BASE_DEVICE" 2>/dev/null)
        if [ "$ROTA" = "0" ]; then
            echo "Disk type: SSD/NVMe (non-rotational)"
        elif [ "$ROTA" = "1" ]; then
            echo "Disk type: HDD (rotational)"
        fi
    fi

    # Try to get more NVMe info if applicable
    if [[ "$DEVICE" == *nvme* ]] && command -v nvme &> /dev/null; then
        echo ""
        echo "NVMe details:"
        nvme list 2>/dev/null | head -5 || true
    fi

    # Show filesystem type
    FS_TYPE=$(df -T "$SAFETENSOR_FILE" 2>/dev/null | tail -1 | awk '{print $2}')
    echo "Filesystem type: $FS_TYPE"
else
    echo "No model files found."
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

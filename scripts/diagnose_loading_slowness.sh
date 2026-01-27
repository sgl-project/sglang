#!/bin/bash
# Diagnose Model Loading Slowness
# Run this script on both MI325 and MI355 CI machines to compare
#
# Usage (inside CI container):
#   bash scripts/diagnose_loading_slowness.sh
#   bash scripts/diagnose_loading_slowness.sh /sgl-data  # specify test dir

set -e

echo "========================================"
echo "Model Loading Diagnosis"
echo "========================================"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo ""

# 1. GPU Info
echo "=== GPU Information ==="
if command -v rocminfo &> /dev/null; then
    echo "GPU Architecture:"
    rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:|gfx" | head -10 || echo "rocminfo failed"
elif command -v rocm-smi &> /dev/null; then
    rocm-smi --showproductname
else
    echo "No ROCm tools found"
fi
echo ""

# 2. Disk Info
echo "=== Storage Information ==="
echo "Mount points:"
df -h 2>/dev/null | grep -E "Filesystem|/sgl-data|/home|overlay" || df -h
echo ""

echo "Disk type (if lsblk available):"
lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null || echo "lsblk not available (inside container)"
echo ""

# 3. HF Cache - CI specific paths
echo "=== HuggingFace Cache ==="

# CI container uses /sgl-data/hf-cache (set by HF_HOME env var)
# See scripts/ci/amd/amd_ci_start_container.sh
if [ -n "$HF_HOME" ]; then
    HF_CACHE="$HF_HOME/hub"
    echo "HF_HOME is set: $HF_HOME"
elif [ -d "/sgl-data/hf-cache" ]; then
    HF_CACHE="/sgl-data/hf-cache/hub"
    echo "Using CI cache path: /sgl-data/hf-cache"
else
    HF_CACHE="${HOME}/.cache/huggingface/hub"
    echo "Using default HF cache path"
fi

echo "HF cache location: $HF_CACHE"

if [ -d "$HF_CACHE" ]; then
    echo "Cache size: $(du -sh $HF_CACHE 2>/dev/null | cut -f1 || echo 'unknown')"
    echo "Number of models: $(ls -d $HF_CACHE/models--* 2>/dev/null | wc -l || echo '0')"
else
    echo "WARNING: HF cache directory does not exist!"
fi

# Check for DeepSeek model
DEEPSEEK_CACHE="$HF_CACHE/models--deepseek-ai--DeepSeek-V3.2"
if [ -d "$DEEPSEEK_CACHE" ]; then
    echo "DeepSeek-V3.2 cache exists: $(du -sh $DEEPSEEK_CACHE 2>/dev/null | cut -f1)"
    # Check if snapshots exist
    if [ -d "$DEEPSEEK_CACHE/snapshots" ]; then
        SNAPSHOT_COUNT=$(ls -1 "$DEEPSEEK_CACHE/snapshots" 2>/dev/null | wc -l)
        echo "  Snapshots: $SNAPSHOT_COUNT"
        # Check for safetensors files
        SAFETENSOR_COUNT=$(find "$DEEPSEEK_CACHE/snapshots" -name "*.safetensors" 2>/dev/null | wc -l)
        echo "  Safetensor files: $SAFETENSOR_COUNT"
    fi
else
    echo "DeepSeek-V3.2 NOT in cache"
fi

# Check for Kimi-K2
KIMI_CACHE="$HF_CACHE/models--moonshotai--Kimi-K2-Instruct-0905"
if [ -d "$KIMI_CACHE" ]; then
    echo "Kimi-K2 cache exists: $(du -sh $KIMI_CACHE 2>/dev/null | cut -f1)"
    if [ -d "$KIMI_CACHE/snapshots" ]; then
        SNAPSHOT_COUNT=$(ls -1 "$KIMI_CACHE/snapshots" 2>/dev/null | wc -l)
        echo "  Snapshots: $SNAPSHOT_COUNT"
        SAFETENSOR_COUNT=$(find "$KIMI_CACHE/snapshots" -name "*.safetensors" 2>/dev/null | wc -l)
        echo "  Safetensor files: $SAFETENSOR_COUNT"
    fi
else
    echo "Kimi-K2 NOT in cache"
fi
echo ""

# 4. Disk Speed Test
echo "=== Disk Speed Test ==="

# Use /sgl-data if available (CI), otherwise use provided arg or /tmp
if [ -d "/sgl-data" ]; then
    TEST_DIR="${1:-/sgl-data}"
else
    TEST_DIR="${1:-/tmp}"
fi
TEST_FILE="$TEST_DIR/disk_test_$$"

echo "Testing in: $TEST_DIR"

# Write test (1GB)
echo "Write test (1GB)..."
start_time=$(date +%s.%N)
dd if=/dev/zero of=$TEST_FILE bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
end_time=$(date +%s.%N)
write_time=$(echo "$end_time - $start_time" | bc)
write_speed=$(echo "scale=2; 1024 / $write_time" | bc 2>/dev/null || echo "N/A")
echo "Write speed: ${write_speed} MB/s (write_time: ${write_time}s)"

# Clear cache
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "(Cannot drop cache, need root)"

# Read test
echo "Read test (1GB)..."
start_time=$(date +%s.%N)
dd if=$TEST_FILE of=/dev/null bs=1M 2>&1 | tail -1
end_time=$(date +%s.%N)
read_time=$(echo "$end_time - $start_time" | bc)
read_speed=$(echo "scale=2; 1024 / $read_time" | bc 2>/dev/null || echo "N/A")
echo "Read speed: ${read_speed} MB/s (read_time: ${read_time}s)"

rm -f $TEST_FILE
echo ""

# Test actual model file read speed if cached
echo "=== Model File Read Test ==="
if [ -d "$DEEPSEEK_CACHE/snapshots" ]; then
    # Find a safetensor file to test
    SAFETENSOR_FILE=$(find "$DEEPSEEK_CACHE/snapshots" -name "*.safetensors" -type f 2>/dev/null | head -1)
    if [ -n "$SAFETENSOR_FILE" ] && [ -f "$SAFETENSOR_FILE" ]; then
        FILE_SIZE=$(stat -c%s "$SAFETENSOR_FILE" 2>/dev/null || stat -f%z "$SAFETENSOR_FILE" 2>/dev/null)
        FILE_SIZE_GB=$(echo "scale=2; $FILE_SIZE / 1024 / 1024 / 1024" | bc)
        echo "Testing read speed on: $(basename $SAFETENSOR_FILE)"
        echo "File size: ${FILE_SIZE_GB} GB"

        # Drop cache first
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

        start_time=$(date +%s.%N)
        dd if="$SAFETENSOR_FILE" of=/dev/null bs=1M 2>&1 | tail -1
        end_time=$(date +%s.%N)
        read_time=$(echo "$end_time - $start_time" | bc)
        read_speed=$(echo "scale=2; $FILE_SIZE / 1024 / 1024 / $read_time" | bc 2>/dev/null || echo "N/A")
        echo "Model file read speed: ${read_speed} MB/s"
    else
        echo "No safetensor files found to test"
    fi
else
    echo "DeepSeek model not cached, skipping model file test"
fi
echo ""

# 5. MIOpen Cache (AMD specific)
echo "=== MIOpen Cache ==="
MIOPEN_CACHE="${MIOPEN_USER_DB_PATH:-/sgl-data/miopen-cache}"
echo "MIOpen cache path: $MIOPEN_CACHE"
if [ -d "$MIOPEN_CACHE" ]; then
    echo "MIOpen cache size: $(du -sh $MIOPEN_CACHE 2>/dev/null | cut -f1)"
else
    echo "MIOpen cache not found"
fi
echo ""

# 6. Network Test (HuggingFace)
echo "=== Network Test (HuggingFace) ==="
echo "Testing HF Hub connectivity..."
HF_URL="https://huggingface.co"
if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 $HF_URL | grep -q "200\|301\|302"; then
    echo "HF Hub reachable"
    # Test download speed with a small file
    echo "Testing API latency..."
    start_time=$(date +%s.%N)
    curl -s -o /dev/null "https://huggingface.co/api/models" 2>&1
    end_time=$(date +%s.%N)
    latency=$(echo "$end_time - $start_time" | bc)
    echo "API latency: ${latency}s"

    # Test actual download speed with a small model file
    echo "Testing download speed (10MB sample)..."
    start_time=$(date +%s.%N)
    curl -s -L -o /dev/null "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt" 2>&1
    end_time=$(date +%s.%N)
    download_time=$(echo "$end_time - $start_time" | bc)
    echo "Small file download time: ${download_time}s"
else
    echo "HF Hub NOT reachable or slow"
fi
echo ""

# 7. Memory Info
echo "=== Memory Information ==="
free -h
echo ""

# 8. CPU Info
echo "=== CPU Information ==="
echo "CPU cores: $(nproc)"
echo "CPU model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo ""

# 9. Environment Variables
echo "=== Relevant Environment Variables ==="
echo "HF_HOME: ${HF_HOME:-not set}"
echo "HF_TOKEN: ${HF_TOKEN:+set (hidden)}"
echo "SGLANG_USE_AITER: ${SGLANG_USE_AITER:-not set}"
echo "SGLANG_AMD_CI: ${SGLANG_AMD_CI:-not set}"
echo "MIOPEN_USER_DB_PATH: ${MIOPEN_USER_DB_PATH:-not set}"
echo ""

# 10. Summary
echo "========================================"
echo "Diagnosis Summary"
echo "========================================"
echo ""
echo "Quick checks:"
if [ -d "$DEEPSEEK_CACHE/snapshots" ]; then
    echo "  [OK] DeepSeek-V3.2 is cached"
else
    echo "  [!!] DeepSeek-V3.2 NOT cached - will need download"
fi
if [ -d "$KIMI_CACHE/snapshots" ]; then
    echo "  [OK] Kimi-K2 is cached"
else
    echo "  [!!] Kimi-K2 NOT cached - will need download"
fi
if [ -d "/sgl-data" ]; then
    echo "  [OK] /sgl-data volume is mounted"
else
    echo "  [!!] /sgl-data NOT mounted - using container storage"
fi
echo ""
echo "========================================"
echo "Diagnosis Complete"
echo "========================================"

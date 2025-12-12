#!/bin/bash
# Check disk I/O bandwidth on HF cache directory
# This helps detect storage performance issues that can cause slow model loading

set -e

# Default values
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
CI_CACHE_DIR="/sgl-data/hf-cache/hub"
TEST_SIZE_MB=1024  # 1GB test
MIN_BW_MB_S=500    # Minimum acceptable bandwidth in MB/s

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo "=============================================="
    echo "       Disk I/O Bandwidth Benchmark"
    echo "=============================================="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
}

# Find the best cache directory to test
find_cache_dir() {
    if [ -d "$CI_CACHE_DIR" ]; then
        echo "$CI_CACHE_DIR"
    elif [ -d "$HF_CACHE_DIR" ]; then
        echo "$HF_CACHE_DIR"
    else
        # Fallback to /tmp
        echo "/tmp"
    fi
}

# Test read bandwidth using dd
test_read_bandwidth() {
    local test_dir="$1"
    local test_file="$test_dir/.disk_io_test_$$"
    local size_mb="$TEST_SIZE_MB"
    
    echo ""
    echo "Testing read bandwidth on: $test_dir"
    echo "Test size: ${size_mb}MB"
    echo ""
    
    # Create test file
    echo "Creating test file..."
    dd if=/dev/zero of="$test_file" bs=1M count="$size_mb" conv=fdatasync 2>/dev/null
    
    # Clear page cache if running as root
    if [ "$(id -u)" = "0" ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
    
    # Test read bandwidth
    echo "Testing read bandwidth..."
    local output
    output=$(dd if="$test_file" of=/dev/null bs=1M 2>&1)
    
    # Clean up
    rm -f "$test_file"
    
    # Parse bandwidth from dd output
    # dd output format varies, try to extract bandwidth
    local bw_line=$(echo "$output" | grep -E 'bytes|copied')
    echo "Raw dd output: $bw_line"
    
    # Extract bandwidth - handle different dd output formats
    local bw_value=""
    local bw_unit=""
    
    # Try to extract GB/s or MB/s
    if echo "$bw_line" | grep -qE '[0-9.]+ GB/s'; then
        bw_value=$(echo "$bw_line" | grep -oE '[0-9.]+ GB/s' | grep -oE '[0-9.]+')
        bw_unit="GB/s"
        bw_mb_s=$(echo "$bw_value * 1024" | bc 2>/dev/null || echo "0")
    elif echo "$bw_line" | grep -qE '[0-9.]+ MB/s'; then
        bw_value=$(echo "$bw_line" | grep -oE '[0-9.]+ MB/s' | grep -oE '[0-9.]+')
        bw_unit="MB/s"
        bw_mb_s="$bw_value"
    else
        # Fallback: calculate from bytes and time
        local bytes=$(echo "$bw_line" | grep -oE '^[0-9]+' | head -1)
        local time=$(echo "$bw_line" | grep -oE '[0-9.]+ s' | grep -oE '[0-9.]+')
        if [ -n "$bytes" ] && [ -n "$time" ]; then
            bw_mb_s=$(echo "scale=2; $bytes / $time / 1024 / 1024" | bc 2>/dev/null || echo "0")
            bw_value="$bw_mb_s"
            bw_unit="MB/s"
        fi
    fi
    
    echo ""
    echo "=============================================="
    echo "         DISK I/O BENCHMARK RESULTS"
    echo "=============================================="
    echo "Test Directory: $test_dir"
    echo "Read Bandwidth: ${bw_value} ${bw_unit} (${bw_mb_s} MB/s)"
    echo "Minimum Required: ${MIN_BW_MB_S} MB/s"
    echo "=============================================="
    
    # Check if bandwidth is acceptable
    if [ -n "$bw_mb_s" ] && [ "$(echo "$bw_mb_s >= $MIN_BW_MB_S" | bc 2>/dev/null)" = "1" ]; then
        echo -e "${GREEN}✓ Disk I/O bandwidth is acceptable${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ Disk I/O bandwidth may be degraded${NC}"
        echo "Expected >= ${MIN_BW_MB_S} MB/s, got ${bw_mb_s} MB/s"
        # Don't fail, just warn
        return 0
    fi
}

# Test safetensors-like read pattern (multiple small files)
test_safetensors_pattern() {
    local test_dir="$1"
    local test_subdir="$test_dir/.safetensors_test_$$"
    local num_files=4
    local file_size_mb=256  # 256MB per file, like typical safetensors shards
    
    echo ""
    echo "Testing safetensors-like read pattern..."
    echo "Pattern: $num_files files x ${file_size_mb}MB each"
    
    mkdir -p "$test_subdir"
    
    # Create test files
    for i in $(seq 1 $num_files); do
        dd if=/dev/zero of="$test_subdir/shard_$i.bin" bs=1M count="$file_size_mb" conv=fdatasync 2>/dev/null
    done
    
    # Clear cache if root
    if [ "$(id -u)" = "0" ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    fi
    
    # Time reading all files sequentially
    local start_time=$(date +%s.%N)
    for i in $(seq 1 $num_files); do
        dd if="$test_subdir/shard_$i.bin" of=/dev/null bs=1M 2>/dev/null
    done
    local end_time=$(date +%s.%N)
    
    # Clean up
    rm -rf "$test_subdir"
    
    # Calculate bandwidth
    local total_mb=$((num_files * file_size_mb))
    local duration=$(echo "$end_time - $start_time" | bc)
    local bw_mb_s=$(echo "scale=2; $total_mb / $duration" | bc 2>/dev/null || echo "0")
    local time_per_shard=$(echo "scale=2; $duration / $num_files" | bc 2>/dev/null || echo "0")
    
    echo ""
    echo "=============================================="
    echo "     SAFETENSORS PATTERN BENCHMARK RESULTS"
    echo "=============================================="
    echo "Total Data: ${total_mb}MB in $num_files files"
    echo "Total Time: ${duration}s"
    echo "Bandwidth: ${bw_mb_s} MB/s"
    echo "Time per shard: ${time_per_shard}s"
    echo "=============================================="
    
    # Compare with expected ~1s/shard for healthy system
    if [ "$(echo "$time_per_shard <= 2.0" | bc 2>/dev/null)" = "1" ]; then
        echo -e "${GREEN}✓ Safetensors loading speed is healthy (<= 2s/shard)${NC}"
    else
        echo -e "${RED}✗ Safetensors loading is SLOW (${time_per_shard}s/shard, expected <= 2s)${NC}"
        echo "This may cause CI timeouts during model loading!"
    fi
}

# Main
main() {
    print_header
    
    local cache_dir=$(find_cache_dir)
    echo "Using cache directory: $cache_dir"
    
    # Check if directory exists and is writable
    if [ ! -d "$cache_dir" ]; then
        echo "Warning: Cache directory does not exist, creating..."
        mkdir -p "$cache_dir" || {
            echo "Error: Cannot create cache directory"
            exit 1
        }
    fi
    
    if [ ! -w "$cache_dir" ]; then
        echo "Error: Cache directory is not writable"
        exit 1
    fi
    
    # Run benchmarks
    test_read_bandwidth "$cache_dir"
    test_safetensors_pattern "$cache_dir"
    
    echo ""
    echo "Disk I/O benchmark complete."
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

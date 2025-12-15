#!/bin/bash
# Validate HuggingFace cache integrity
# This script checks that cached models have actual safetensor files, not just metadata
# If a model cache is incomplete/corrupted, it will be cleared to force re-download

set -e

# Default cache directory
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
CI_CACHE_DIR="/sgl-data/hf-cache/hub"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find the cache directory
find_cache_dir() {
    if [ -d "$CI_CACHE_DIR" ]; then
        echo "$CI_CACHE_DIR"
    elif [ -d "$HF_CACHE_DIR" ]; then
        echo "$HF_CACHE_DIR"
    else
        echo ""
    fi
}

# Validate a single model cache
# Returns 0 if valid, 1 if invalid/incomplete
validate_model_cache() {
    local model_dir="$1"
    local model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g')
    
    # Check if snapshots directory exists
    if [ ! -d "$model_dir/snapshots" ]; then
        echo -e "${YELLOW}⚠ No snapshots directory for $model_name${NC}"
        return 1
    fi
    
    # Find the latest snapshot
    local snapshot_dir=$(ls -td "$model_dir/snapshots"/*/ 2>/dev/null | head -1)
    if [ -z "$snapshot_dir" ]; then
        echo -e "${YELLOW}⚠ No snapshot found for $model_name${NC}"
        return 1
    fi
    
    # Check for actual model files (safetensors, bin, or pt)
    local model_files=$(find "$snapshot_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) 2>/dev/null | wc -l)
    
    if [ "$model_files" -eq 0 ]; then
        echo -e "${YELLOW}⚠ No model files (*.safetensors, *.bin, *.pt) found for $model_name${NC}"
        echo "  Snapshot: $snapshot_dir"
        return 1
    fi
    
    # Check if files are actual files (not empty or broken symlinks)
    local valid_files=0
    local total_size=0
    while IFS= read -r -d '' f; do
        if [ -f "$f" ] && [ -s "$f" ]; then
            valid_files=$((valid_files + 1))
            local fsize=$(stat -c%s "$f" 2>/dev/null || echo 0)
            total_size=$((total_size + fsize))
        fi
    done < <(find "$snapshot_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -print0 2>/dev/null)
    
    if [ "$valid_files" -eq 0 ]; then
        echo -e "${YELLOW}⚠ All model files are empty or broken for $model_name${NC}"
        return 1
    fi
    
    # Convert to MB for display
    local size_mb=$((total_size / 1024 / 1024))
    echo -e "${GREEN}✓ $model_name: $valid_files files, ${size_mb}MB${NC}"
    return 0
}

# Clear a model cache
clear_model_cache() {
    local model_dir="$1"
    local model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//g')
    
    echo -e "${YELLOW}Clearing cache for $model_name...${NC}"
    rm -rf "$model_dir"
    echo -e "${GREEN}✓ Cleared $model_name${NC}"
}

# Main
main() {
    echo "=============================================="
    echo "     HuggingFace Cache Validation"
    echo "=============================================="
    echo "Date: $(date)"
    
    local cache_dir=$(find_cache_dir)
    
    if [ -z "$cache_dir" ] || [ ! -d "$cache_dir" ]; then
        echo "No HF cache directory found, nothing to validate."
        exit 0
    fi
    
    echo "Cache directory: $cache_dir"
    echo ""
    
    local invalid_count=0
    local valid_count=0
    local cleared_count=0
    
    # Find all model directories
    for model_dir in "$cache_dir"/models--*; do
        if [ -d "$model_dir" ]; then
            if validate_model_cache "$model_dir"; then
                valid_count=$((valid_count + 1))
            else
                invalid_count=$((invalid_count + 1))
                # Clear invalid cache to force re-download
                clear_model_cache "$model_dir"
                cleared_count=$((cleared_count + 1))
            fi
        fi
    done
    
    echo ""
    echo "=============================================="
    echo "           VALIDATION SUMMARY"
    echo "=============================================="
    echo "Valid caches: $valid_count"
    echo "Invalid/incomplete caches: $invalid_count"
    echo "Caches cleared: $cleared_count"
    echo "=============================================="
    
    if [ "$cleared_count" -gt 0 ]; then
        echo -e "${YELLOW}Note: $cleared_count model caches were cleared and will be re-downloaded on next use.${NC}"
    fi
}

# Run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

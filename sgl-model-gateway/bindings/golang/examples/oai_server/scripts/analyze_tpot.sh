#!/bin/bash

# TPOT performance bottleneck analysis script
# Specifically designed to analyze why Go Router is twice as slow as Rust Router
#
# Usage:
#   ./scripts/analyze_tpot.sh [options]
#
# Options:
#   --duration SECONDS     CPU profile duration (default: 60)
#   --requests NUM        Number of requests (default: 100)
#   --concurrency NUM     Concurrency level (default: 20)
#   --pprof-port PORT     pprof port (default: 6060)
#   --server-url URL      Server URL (default: http://localhost:8080)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_DIR="${PROJECT_ROOT}/profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROFILE_DIR}/tpot_analysis_${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
DURATION=${DURATION:-60}
NUM_REQUESTS=${NUM_REQUESTS:-100}
CONCURRENCY=${CONCURRENCY:-20}
PPROF_PORT=${PPROF_PORT:-6060}
SERVER_URL=${SERVER_URL:-http://localhost:8080}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --pprof-port)
            PPROF_PORT="$2"
            shift 2
            ;;
        --server-url)
            SERVER_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Check for graphviz (optional, needed for some pprof visualizations)
HAS_GRAPHVIZ=false
if command -v dot >/dev/null 2>&1; then
    HAS_GRAPHVIZ=true
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TPOT Performance Bottleneck Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Duration:    ${DURATION}s"
echo "  Requests:   $NUM_REQUESTS"
echo "  Concurrency: $CONCURRENCY"
echo "  Server URL:  $SERVER_URL"
echo "  pprof Port:  $PPROF_PORT"
echo "  Output Dir:  $OUTPUT_DIR"
if [ "$HAS_GRAPHVIZ" = "false" ]; then
    echo ""
    echo -e "${YELLOW}Note: graphviz not found. Some pprof visualizations may not work.${NC}"
    echo -e "${YELLOW}To install graphviz:${NC}"
    echo -e "${YELLOW}  macOS:   brew install graphviz${NC}"
    echo -e "${YELLOW}  Ubuntu:  sudo apt-get install graphviz${NC}"
    echo -e "${YELLOW}  CentOS:  sudo yum install graphviz${NC}"
    echo -e "${YELLOW}Text reports will still be generated without graphviz.${NC}"
fi
echo ""

# Check if server is running
echo -e "${YELLOW}[Check] Verifying server is running...${NC}"
if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Server not responding at ${SERVER_URL}${NC}"
    echo ""
    echo "Please start the server first with profiling enabled:"
    echo "  ./run.sh --profile --pprof-port $PPROF_PORT"
    echo "  or"
    echo "  PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT make run"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo ""

# Check if pprof is enabled
echo -e "${YELLOW}[Check] Verifying pprof is enabled...${NC}"
if ! curl -s "http://localhost:${PPROF_PORT}/debug/pprof/" > /dev/null 2>&1; then
    echo -e "${RED}Error: pprof not accessible at http://localhost:${PPROF_PORT}/debug/pprof/${NC}"
    echo ""
    echo "Please start the server with profiling enabled:"
    echo "  ./run.sh --profile --pprof-port $PPROF_PORT"
    exit 1
fi
echo -e "${GREEN}✓ pprof is enabled${NC}"
echo ""

# ============================================
# Step 1: Collect baseline profiles
# ============================================
echo -e "${GREEN}[Step 1/8] Collecting baseline profiles...${NC}"

# Baseline memory
go tool pprof -proto -output="${OUTPUT_DIR}/heap_before.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" > /dev/null 2>&1 || true

# Baseline goroutine
go tool pprof -proto -output="${OUTPUT_DIR}/goroutine_before.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/goroutine" > /dev/null 2>&1 || true

echo -e "${GREEN}✓ Baseline profiles collected${NC}"
echo ""

# ============================================
# Step 2: Start CPU profile collection
# ============================================
echo -e "${GREEN}[Step 2/8] Starting CPU profile collection (${DURATION}s)...${NC}"
go tool pprof -proto -output="${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=${DURATION}" &
CPU_PID=$!
sleep 2
echo -e "${GREEN}✓ CPU profile collection started${NC}"
echo ""

# ============================================
# Step 3: Run load test with streaming requests
# ============================================
echo -e "${GREEN}[Step 3/8] Running load test ($NUM_REQUESTS streaming requests, concurrency=$CONCURRENCY)...${NC}"

# Function to run a single streaming request
run_streaming_request() {
    local request_id=$1
    local start_time=$(date +%s)
    local start_nanos=$(date +%N 2>/dev/null || echo "000000000")

    curl -N -s -X POST "${SERVER_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"default\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Write a 500-word story with character dialogue and scene descriptions\"}],
            \"stream\": true,
            \"max_tokens\": 300,
            \"temperature\": 0.7
        }" > /dev/null

    local end_time=$(date +%s)
    local end_nanos=$(date +%N 2>/dev/null || echo "000000000")
    local duration=$((end_time - start_time))
    echo "$duration" >> "${OUTPUT_DIR}/request_times.txt"
}

# Run requests with controlled concurrency
# Use a temporary file to track job PIDs to avoid conflicts with CPU_PID
JOB_PIDS_FILE="${OUTPUT_DIR}/.job_pids_$$"
> "$JOB_PIDS_FILE"

for i in $(seq 1 $NUM_REQUESTS); do
    # Wait if we've reached concurrency limit
    while [ $(wc -l < "$JOB_PIDS_FILE" 2>/dev/null || echo 0) -ge $CONCURRENCY ]; do
        # Check and remove completed jobs
        while IFS= read -r pid; do
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                # Process completed, remove from file
                grep -v "^${pid}$" "$JOB_PIDS_FILE" > "${JOB_PIDS_FILE}.tmp" && \
                    mv "${JOB_PIDS_FILE}.tmp" "$JOB_PIDS_FILE" || true
            fi
        done < "$JOB_PIDS_FILE"
        sleep 0.1
    done

    # Start new request
    run_streaming_request $i &
    echo $! >> "$JOB_PIDS_FILE"

    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Progress: $i/$NUM_REQUESTS requests sent..."
    fi
done

# Wait for all remaining jobs (excluding CPU_PID)
while IFS= read -r pid; do
    if [ -n "$pid" ] && [ "$pid" != "$CPU_PID" ]; then
        wait "$pid" 2>/dev/null || true
    fi
done < "$JOB_PIDS_FILE"

# Clean up
rm -f "$JOB_PIDS_FILE" "${JOB_PIDS_FILE}.tmp" 2>/dev/null || true

echo -e "${GREEN}✓ Load test completed${NC}"
echo ""

# ============================================
# Step 4: Wait for CPU profile to complete
# ============================================
echo -e "${GREEN}[Step 4/8] Waiting for CPU profile to complete...${NC}"
# Wait for the process, but handle the case where it might have already completed
if kill -0 $CPU_PID 2>/dev/null; then
    wait $CPU_PID 2>/dev/null || true
else
    # Process already completed, just wait a bit to ensure file is written
    sleep 1
fi
echo -e "${GREEN}✓ CPU profile collection completed${NC}"
echo ""

# ============================================
# Step 5: Collect final profiles
# ============================================
echo -e "${GREEN}[Step 5/8] Collecting final profiles...${NC}"

# Final memory
go tool pprof -proto -output="${OUTPUT_DIR}/heap_after.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" > /dev/null 2>&1 || true

# Final goroutine
go tool pprof -proto -output="${OUTPUT_DIR}/goroutine_after.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/goroutine" > /dev/null 2>&1 || true

# Mutex profile
go tool pprof -proto -output="${OUTPUT_DIR}/mutex.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/mutex" > /dev/null 2>&1 || true

# Block profile
go tool pprof -proto -output="${OUTPUT_DIR}/block.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/block" > /dev/null 2>&1 || true

echo -e "${GREEN}✓ Final profiles collected${NC}"
echo ""

# ============================================
# Step 6: Generate analysis reports
# ============================================
echo -e "${GREEN}[Step 6/8] Generating analysis reports...${NC}"

# CPU analysis
echo "  Generating CPU reports..."
go tool pprof -top -cum "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" > "${OUTPUT_DIR}/01_cpu_top_cum.txt" 2>&1 || true
go tool pprof -top "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" > "${OUTPUT_DIR}/02_cpu_top_flat.txt" 2>&1 || true

# Memory analysis
echo "  Generating memory reports..."
if [ -f "${OUTPUT_DIR}/heap_after.pb.gz" ]; then
    go tool pprof -top -alloc_space "${OUTPUT_DIR}/heap_after.pb.gz" > "${OUTPUT_DIR}/03_memory_alloc_space.txt" 2>&1 || true
    go tool pprof -top -alloc_objects "${OUTPUT_DIR}/heap_after.pb.gz" > "${OUTPUT_DIR}/04_memory_alloc_objects.txt" 2>&1 || true
    go tool pprof -top -inuse_space "${OUTPUT_DIR}/heap_after.pb.gz" > "${OUTPUT_DIR}/05_memory_inuse_space.txt" 2>&1 || true
fi

# Memory growth
if [ -f "${OUTPUT_DIR}/heap_before.pb.gz" ] && [ -f "${OUTPUT_DIR}/heap_after.pb.gz" ]; then
    go tool pprof -top -base="${OUTPUT_DIR}/heap_before.pb.gz" \
        "${OUTPUT_DIR}/heap_after.pb.gz" > "${OUTPUT_DIR}/06_memory_growth.txt" 2>&1 || true
fi

# FFI/CGO analysis
echo "  Analyzing FFI/CGO calls..."
go tool pprof -top "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" 2>&1 | \
    grep -iE "(block_on|CGO|FFI|ffi|runtime\.cgo|_Cfunc)" > "${OUTPUT_DIR}/07_ffi_cgo_analysis.txt" || \
    echo "No FFI/CGO related functions found" > "${OUTPUT_DIR}/07_ffi_cgo_analysis.txt"

# JSON serialization analysis
echo "  Analyzing JSON serialization..."
go tool pprof -top "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" 2>&1 | \
    grep -iE "(json|Marshal|Unmarshal|Encode|Decode|sonic|jsoniter)" > "${OUTPUT_DIR}/08_json_analysis.txt" || \
    echo "No JSON related functions found" > "${OUTPUT_DIR}/08_json_analysis.txt"

# Goroutine analysis
if [ -f "${OUTPUT_DIR}/goroutine_after.pb.gz" ]; then
    echo "  Analyzing goroutines..."
    go tool pprof -top "${OUTPUT_DIR}/goroutine_after.pb.gz" > "${OUTPUT_DIR}/09_goroutine_analysis.txt" 2>&1 || true
fi

# Mutex analysis
if [ -f "${OUTPUT_DIR}/mutex.pb.gz" ]; then
    echo "  Analyzing mutex contention..."
    go tool pprof -top "${OUTPUT_DIR}/mutex.pb.gz" > "${OUTPUT_DIR}/10_mutex_analysis.txt" 2>&1 || true
fi

# Block analysis
if [ -f "${OUTPUT_DIR}/block.pb.gz" ]; then
    echo "  Analyzing blocking operations..."
    go tool pprof -top "${OUTPUT_DIR}/block.pb.gz" > "${OUTPUT_DIR}/11_block_analysis.txt" 2>&1 || true
fi

# Request timing statistics
if [ -f "${OUTPUT_DIR}/request_times.txt" ] && [ -s "${OUTPUT_DIR}/request_times.txt" ]; then
    echo "  Calculating request timing statistics..."
    {
        echo "Request Timing Statistics"
        echo "========================"
        echo ""
        echo "Total requests: $(wc -l < "${OUTPUT_DIR}/request_times.txt" | tr -d ' ')"
        echo ""
        awk '{
            sum+=$1
            sumsq+=$1*$1
            if(NR==1 || $1<min) min=$1
            if(NR==1 || $1>max) max=$1
        } END {
            if(NR > 0) {
                mean=sum/NR
                variance=(sumsq/NR - mean*mean)
                stddev=sqrt(variance)
                print "Min:    " min "s"
                print "Max:    " max "s"
                print "Mean:   " mean "s"
                print "StdDev: " stddev "s"
            }
        }' "${OUTPUT_DIR}/request_times.txt"
    } > "${OUTPUT_DIR}/12_request_timing.txt"
fi

echo -e "${GREEN}✓ Analysis reports generated${NC}"
echo ""

# ============================================
# Step 7: Generate summary report
# ============================================
echo -e "${GREEN}[Step 7/8] Generating summary report...${NC}"

SUMMARY_FILE="${OUTPUT_DIR}/00_SUMMARY.md"
cat > "$SUMMARY_FILE" <<EOF
# TPOT Performance Analysis Summary

**Analysis Date:** $(date)
**Duration:** ${DURATION}s
**Requests:** $NUM_REQUESTS
**Concurrency:** $CONCURRENCY

## Key Findings

### 1. CPU Hotspots (Top 10 Cumulative Time)

\`\`\`
$(head -15 "${OUTPUT_DIR}/01_cpu_top_cum.txt" | tail -10)
\`\`\`

### 2. CPU Hotspots (Top 10 Flat Time)

\`\`\`
$(head -15 "${OUTPUT_DIR}/02_cpu_top_flat.txt" | tail -10)
\`\`\`

### 3. FFI/CGO Overhead

\`\`\`
$(cat "${OUTPUT_DIR}/07_ffi_cgo_analysis.txt")
\`\`\`

### 4. JSON Serialization Overhead

\`\`\`
$(cat "${OUTPUT_DIR}/08_json_analysis.txt")
\`\`\`

### 5. Memory Allocation (Top 10 by Space)

\`\`\`
$(head -15 "${OUTPUT_DIR}/03_memory_alloc_space.txt" | tail -10)
\`\`\`

### 6. Memory Allocation (Top 10 by Objects)

\`\`\`
$(head -15 "${OUTPUT_DIR}/04_memory_alloc_objects.txt" | tail -10)
\`\`\`

### 7. Mutex Contention

\`\`\`
$(head -15 "${OUTPUT_DIR}/10_mutex_analysis.txt" | tail -10 2>/dev/null || echo "No significant mutex contention detected")
\`\`\`

### 8. Blocking Operations

\`\`\`
$(head -15 "${OUTPUT_DIR}/11_block_analysis.txt" | tail -10 2>/dev/null || echo "No significant blocking detected")
\`\`\`

## Performance Bottlenecks Identified

### High Priority Issues

1. **FFI/CGO Overhead**
   - Check: \`cat ${OUTPUT_DIR}/07_ffi_cgo_analysis.txt\`
   - Impact: FFI calls add overhead compared to native Rust code
   - Recommendation: Minimize FFI calls, batch operations

2. **JSON Serialization**
   - Check: \`cat ${OUTPUT_DIR}/08_json_analysis.txt\`
   - Impact: JSON marshaling/unmarshaling can be expensive
   - Recommendation: Use faster JSON library (jsoniter), reduce serialization frequency

3. **Memory Allocations**
   - Check: \`cat ${OUTPUT_DIR}/03_memory_alloc_space.txt\`
   - Impact: Frequent allocations cause GC pressure
   - Recommendation: Use object pools, pre-allocate buffers

### Medium Priority Issues

4. **Goroutine Overhead**
   - Check: \`cat ${OUTPUT_DIR}/09_goroutine_analysis.txt\`
   - Impact: Too many goroutines can cause scheduling overhead
   - Recommendation: Limit goroutine count, use worker pools

5. **Lock Contention**
   - Check: \`cat ${OUTPUT_DIR}/10_mutex_analysis.txt\`
   - Impact: Lock contention reduces parallelism
   - Recommendation: Reduce lock granularity, use lock-free structures

## Comparison with Rust Router

### Expected Differences

1. **FFI Overhead**: Go → Rust FFI calls add ~100-500ns per call
2. **GC Overhead**: Go's GC can cause pauses (usually <1ms)
3. **JSON Library**: Go's standard library is slower than Rust's serde
4. **Memory Layout**: Go's GC affects cache locality

### Optimization Opportunities

1. **Reduce FFI Calls**
   - Batch token processing
   - Use async FFI (if possible)
   - Cache frequently used FFI results

2. **Optimize JSON**
   - Use jsoniter (already implemented)
   - Pre-allocate JSON buffers
   - Reduce serialization frequency

3. **Memory Management**
   - Use sync.Pool for frequently allocated objects
   - Pre-allocate slices with known capacity
   - Avoid unnecessary string copies

4. **Concurrency**
   - Use worker pools instead of spawning goroutines per request
   - Limit concurrent FFI calls
   - Use channels efficiently

## Next Steps

1. Review detailed reports in this directory
2. Use interactive pprof: \`go tool pprof -http=:8081 ${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz\`
3. Compare with Rust router profiles (if available)
4. Implement optimizations based on findings
5. Re-run analysis to measure improvements

## Files Generated

- \`00_SUMMARY.md\` - This summary
- \`01_cpu_top_cum.txt\` - CPU top functions (cumulative)
- \`02_cpu_top_flat.txt\` - CPU top functions (flat)
- \`03_memory_alloc_space.txt\` - Memory allocation by space
- \`04_memory_alloc_objects.txt\` - Memory allocation by objects
- \`05_memory_inuse_space.txt\` - Memory in use by space
- \`06_memory_growth.txt\` - Memory growth during test
- \`07_ffi_cgo_analysis.txt\` - FFI/CGO overhead analysis
- \`08_json_analysis.txt\` - JSON serialization analysis
- \`09_goroutine_analysis.txt\` - Goroutine analysis
- \`10_mutex_analysis.txt\` - Mutex contention analysis
- \`11_block_analysis.txt\` - Blocking operations analysis
- \`12_request_timing.txt\` - Request timing statistics
- \`*.pb.gz\` - Raw profile files for interactive analysis

EOF

echo -e "${GREEN}✓ Summary report generated${NC}"
echo ""

# ============================================
# Step 8: Display summary
# ============================================
echo -e "${GREEN}[Step 8/8] Analysis Complete!${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Top CPU Hotspots (Cumulative):${NC}"
head -12 "${OUTPUT_DIR}/01_cpu_top_cum.txt" | tail -10
echo ""
echo -e "${YELLOW}FFI/CGO Overhead:${NC}"
cat "${OUTPUT_DIR}/07_ffi_cgo_analysis.txt"
echo ""
echo -e "${YELLOW}JSON Serialization Overhead:${NC}"
cat "${OUTPUT_DIR}/08_json_analysis.txt"
echo ""
echo -e "${YELLOW}Top Memory Allocations:${NC}"
head -12 "${OUTPUT_DIR}/03_memory_alloc_space.txt" | tail -10
echo ""
if [ -f "${OUTPUT_DIR}/12_request_timing.txt" ]; then
    echo -e "${YELLOW}Request Timing:${NC}"
    cat "${OUTPUT_DIR}/12_request_timing.txt"
    echo ""
fi
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Detailed Reports:${NC}"
echo "  Summary:      cat ${OUTPUT_DIR}/00_SUMMARY.md"
echo "  CPU (cum):    cat ${OUTPUT_DIR}/01_cpu_top_cum.txt"
echo "  CPU (flat):   cat ${OUTPUT_DIR}/02_cpu_top_flat.txt"
echo "  FFI/CGO:      cat ${OUTPUT_DIR}/07_ffi_cgo_analysis.txt"
echo "  JSON:         cat ${OUTPUT_DIR}/08_json_analysis.txt"
echo "  Memory:       cat ${OUTPUT_DIR}/03_memory_alloc_space.txt"
echo ""
echo -e "${BLUE}Interactive Analysis:${NC}"
echo "  Run: go tool pprof -http=:8081 ${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz"
echo "  Then visit:"
echo "    - http://localhost:8081/ui/flamegraph (Flame Graph - no graphviz needed)"
echo "    - http://localhost:8081/ui/top (Top Functions - no graphviz needed)"
if [ "$HAS_GRAPHVIZ" = "true" ]; then
    echo "    - http://localhost:8081/ui/graph (Call Graph - requires graphviz)"
else
    echo "    - http://localhost:8081/ui/graph (Call Graph - requires graphviz, not available)"
fi
echo ""
if [ "$HAS_GRAPHVIZ" = "false" ]; then
    echo -e "${YELLOW}Note: Install graphviz to enable call graph visualization:${NC}"
    echo -e "${YELLOW}  macOS:   brew install graphviz${NC}"
    echo -e "${YELLOW}  Ubuntu:  sudo apt-get install graphviz${NC}"
    echo -e "${YELLOW}  CentOS:  sudo yum install graphviz${NC}"
    echo ""
fi
echo -e "${GREEN}All files saved to: ${OUTPUT_DIR}${NC}"
echo ""

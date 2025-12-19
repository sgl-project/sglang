#!/bin/bash

# TPOT performance analysis script
# Quickly collect and analyze TPOT-related performance data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_DIR="${PROJECT_ROOT}/profiles"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROFILE_DIR}/${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
PPROF_PORT=${PPROF_PORT:-6060}
SERVER_URL=${SERVER_URL:-http://localhost:8080}
DURATION=${DURATION:-30}
NUM_REQUESTS=${NUM_REQUESTS:-20}

mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}TPOT Performance Analysis${NC}"
echo "=========================="
echo "Profile directory: $OUTPUT_DIR"
echo "Duration: ${DURATION}s"
echo "Requests: $NUM_REQUESTS"
echo ""

# Check if server is running
if ! curl -s "${SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Server not responding at ${SERVER_URL}${NC}"
    echo "Please start the server first with profiling enabled:"
    echo "  PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT make run"
    exit 1
fi

# Collect baseline memory
echo -e "${GREEN}[1/5] Collecting baseline memory profile...${NC}"
go tool pprof -proto -output="${OUTPUT_DIR}/heap_before.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" > /dev/null 2>&1 || true

# Start CPU profile collection in background
echo -e "${GREEN}[2/5] Starting CPU profile collection (${DURATION}s)...${NC}"
go tool pprof -proto -output="${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=${DURATION}" &
CPU_PID=$!

# Wait a bit for profile to start
sleep 2

# Run load test
echo -e "${GREEN}[3/5] Running load test ($NUM_REQUESTS requests)...${NC}"
for i in $(seq 1 $NUM_REQUESTS); do
  curl -N -s -X POST "${SERVER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"default\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Write a story\"}],
      \"stream\": true,
      \"max_tokens\": 200
    }" > /dev/null &

  # Limit concurrency
  if [ $((i % 5)) -eq 0 ]; then
    wait
  fi
done
wait

# Wait for CPU profile to complete
echo -e "${GREEN}[4/5] Waiting for CPU profile to complete...${NC}"
# Wait for the CPU profile process, but handle the case where it's not a child process
if kill -0 $CPU_PID 2>/dev/null; then
    # Process is still running, wait for it
    while kill -0 $CPU_PID 2>/dev/null; do
        sleep 1
    done
else
    # Process already completed or not found, just wait a bit
    sleep 2
fi

# Collect final memory
echo -e "${GREEN}[5/5] Collecting final memory profile...${NC}"
go tool pprof -proto -output="${OUTPUT_DIR}/heap_after.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" > /dev/null 2>&1 || true

# Generate reports
echo ""
echo -e "${GREEN}Generating reports...${NC}"

# CPU top (cumulative)
go tool pprof -top -cum "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" > "${OUTPUT_DIR}/cpu_top_cum.txt" 2>&1 || true

# CPU top (flat)
go tool pprof -top "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" > "${OUTPUT_DIR}/cpu_top_flat.txt" 2>&1 || true

# Memory growth
if [ -f "${OUTPUT_DIR}/heap_before.pb.gz" ] && [ -f "${OUTPUT_DIR}/heap_after.pb.gz" ]; then
    go tool pprof -top -base="${OUTPUT_DIR}/heap_before.pb.gz" \
        "${OUTPUT_DIR}/heap_after.pb.gz" > "${OUTPUT_DIR}/heap_growth.txt" 2>&1 || true
fi

# FFI/CGO related
go tool pprof -top "${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz" 2>&1 | \
    grep -E "(block_on|CGO|FFI|json|Marshal|Unmarshal)" > "${OUTPUT_DIR}/ffi_related.txt" || \
    echo "No FFI/CGO related functions found" > "${OUTPUT_DIR}/ffi_related.txt"

# Summary
echo ""
echo -e "${GREEN}=== Analysis Summary ===${NC}"
echo ""
echo -e "${YELLOW}CPU Top (Cumulative) - Top 10:${NC}"
head -12 "${OUTPUT_DIR}/cpu_top_cum.txt" | tail -10 || true

echo ""
echo -e "${YELLOW}CPU Top (Flat) - Top 10:${NC}"
head -12 "${OUTPUT_DIR}/cpu_top_flat.txt" | tail -10 || true

echo ""
echo -e "${YELLOW}FFI/CGO Related Functions:${NC}"
cat "${OUTPUT_DIR}/ffi_related.txt" || true

echo ""
echo -e "${GREEN}=== Detailed Reports ===${NC}"
echo "CPU (cumulative): cat ${OUTPUT_DIR}/cpu_top_cum.txt"
echo "CPU (flat):       cat ${OUTPUT_DIR}/cpu_top_flat.txt"
echo "Memory growth:    cat ${OUTPUT_DIR}/heap_growth.txt"
echo "FFI related:     cat ${OUTPUT_DIR}/ffi_related.txt"
echo ""
echo -e "${GREEN}=== Interactive Analysis ===${NC}"
echo "Run: go tool pprof -http=:8081 ${OUTPUT_DIR}/cpu_${DURATION}s.pb.gz"
echo "Then visit: http://localhost:8081/ui/flamegraph"
echo ""
echo "Profile files saved to: ${OUTPUT_DIR}"

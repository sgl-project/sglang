#!/bin/bash

# pprof performance analysis script
# Used to analyze performance bottlenecks of Go OpenAI server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
PPROF_PORT=${PPROF_PORT:-6060}
SERVER_PORT=${SERVER_PORT:-8080}
DURATION=${DURATION:-60}  # Performance test duration (seconds)
OUTPUT_DIR="./pprof_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "pprof Performance Analysis Tool"
echo "=========================================="
echo "PPROF_PORT: $PPROF_PORT"
echo "SERVER_PORT: $SERVER_PORT"
echo "DURATION: ${DURATION}s"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Check if go tool pprof is available
if ! command -v go &> /dev/null; then
    echo "Error: go command not found"
    exit 1
fi

# Check if server is running
check_server() {
    if curl -s "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if pprof is available
check_pprof() {
    if curl -s "http://localhost:${PPROF_PORT}/debug/pprof/" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start server (if not running)
if ! check_server; then
    echo "Server not running, please start the server first:"
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    echo "  ./oai_server"
    echo ""
    echo "Or use the following command to start:"
    echo "  PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT ./oai_server"
    echo ""
    read -p "Start server now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting server..."
        PPROF_ENABLED=true PPROF_PORT=$PPROF_PORT ./oai_server &
        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"

        # Wait for server to start
        echo "Waiting for server to start..."
        for i in {1..30}; do
            if check_server; then
                echo "Server started"
                break
            fi
            sleep 1
        done

        if ! check_server; then
            echo "Error: Server failed to start"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    else
        exit 1
    fi
fi

# Check if pprof is available
if ! check_pprof; then
    echo "Error: pprof not enabled. Please set environment variables:"
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    exit 1
fi

echo "Starting to collect performance data..."
echo ""

# 1. CPU Profile (30 seconds)
echo "[1/6] Collecting CPU Profile (30 seconds)..."
go tool pprof -proto -output="$OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30" &
CPU_PID=$!

# 2. Collect Heap Profile simultaneously
echo "[2/6] Collecting Heap Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/heap" &
HEAP_PID=$!

# 3. Collect Goroutine Profile
echo "[3/6] Collecting Goroutine Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/goroutine_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/goroutine" &
GOROUTINE_PID=$!

# 4. Collect Mutex Profile
echo "[4/6] Collecting Mutex Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/mutex_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/mutex" &
MUTEX_PID=$!

# 5. Collect Block Profile
echo "[5/6] Collecting Block Profile..."
go tool pprof -proto -output="$OUTPUT_DIR/block_${TIMESTAMP}.pb.gz" \
    "http://localhost:${PPROF_PORT}/debug/pprof/block" &
BLOCK_PID=$!

# 6. Run performance test (during CPU profile collection)
echo "[6/6] Running performance test..."
echo "Tip: Please use your performance testing tool (curl, ab, wrk, etc.) to send requests to the server"
echo "     CPU profile will collect 30 seconds of performance data"
echo ""

# Wait for CPU profile to complete
wait $CPU_PID
echo "CPU Profile collection completed"

# Wait for other profiles
wait $HEAP_PID
wait $GOROUTINE_PID
wait $MUTEX_PID
wait $BLOCK_PID

echo ""
echo "=========================================="
echo "Performance data collection completed!"
echo "=========================================="
echo ""
echo "Generated analysis files:"
ls -lh "$OUTPUT_DIR"/*_${TIMESTAMP}.* 2>/dev/null || true
echo ""

# Generate analysis report
echo "Generating analysis report..."
echo ""

# CPU Top 20
echo "=== CPU Top 20 (sorted by flat time) ===" > "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top -cum "$OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Heap Top 20
echo "=== Heap Top 20 (sorted by allocation size) ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Goroutine statistics
echo "=== Goroutine Statistics ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/goroutine_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Mutex statistics
echo "=== Mutex Wait Time ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/mutex_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true
echo "" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"

# Block statistics
echo "=== Block Wait Time ===" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
go tool pprof -top "$OUTPUT_DIR/block_${TIMESTAMP}.pb.gz" >> "$OUTPUT_DIR/analysis_${TIMESTAMP}.txt" 2>&1 || true

echo "Analysis report saved to: $OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
echo ""

# Display key information
echo "=========================================="
echo "Key Performance Metrics Summary"
echo "=========================================="
echo ""
echo "View detailed report:"
echo "  cat $OUTPUT_DIR/analysis_${TIMESTAMP}.txt"
echo ""
echo "Interactive CPU Profile view:"
echo "  go tool pprof $OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz"
echo ""
echo "Interactive Heap Profile view:"
echo "  go tool pprof $OUTPUT_DIR/heap_${TIMESTAMP}.pb.gz"
echo ""
echo "Generate flame graph (requires go-torch or pprof):"
echo "  go tool pprof -http=:8080 $OUTPUT_DIR/cpu_${TIMESTAMP}.pb.gz"
echo ""

# If server was started, ask if it should be closed
if [ -n "$SERVER_PID" ]; then
    read -p "Close server? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill $SERVER_PID 2>/dev/null || true
        echo "Server closed"
    fi
fi

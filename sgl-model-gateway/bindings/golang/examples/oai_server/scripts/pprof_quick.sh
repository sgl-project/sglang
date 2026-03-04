#!/bin/bash

# Quick pprof analysis script
# Collects 30-second CPU profile and immediately displays top results

set -e

PPROF_PORT=${PPROF_PORT:-6060}
DURATION=${DURATION:-30}

echo "=========================================="
echo "Quick pprof Analysis"
echo "=========================================="
echo "PPROF_PORT: $PPROF_PORT"
echo "DURATION: ${DURATION}s"
echo ""
echo "Tip: During data collection, please send requests to the server"
echo "     You can use: ./pprof_test.sh"
echo ""

# Check if pprof is available
if ! curl -s "http://localhost:${PPROF_PORT}/debug/pprof/" > /dev/null 2>&1; then
    echo "Error: pprof not enabled. Please set environment variables:"
    echo "  export PPROF_ENABLED=true"
    echo "  export PPROF_PORT=$PPROF_PORT"
    exit 1
fi

echo "Starting to collect CPU Profile (${DURATION} seconds)..."
echo ""

# Collect CPU profile and directly display top results
go tool pprof -top -cum "http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=${DURATION}"

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""
echo "More analysis options:"
echo "  # Interactive view"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30"
echo ""
echo "  # View heap memory"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/heap"
echo ""
echo "  # View goroutines"
echo "  go tool pprof http://localhost:${PPROF_PORT}/debug/pprof/goroutine"
echo ""
echo "  # Generate Web UI"
echo "  go tool pprof -http=:8080 http://localhost:${PPROF_PORT}/debug/pprof/profile?seconds=30"
echo ""

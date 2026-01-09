#!/bin/bash
# E2E Test Runner for Attention Explorer
#
# Usage:
#   ./run_e2e_tests.sh                    # Run against localhost:8000
#   ./run_e2e_tests.sh --server http://server:port
#   ./run_e2e_tests.sh --full-validation  # Run full test suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default settings
SERVER_URL="${SGLANG_SERVER_URL:-http://localhost:8000}"
TOP_K=32
TIMEOUT=120
FULL_VALIDATION=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server)
            SERVER_URL="$2"
            shift 2
            ;;
        --full-validation)
            FULL_VALIDATION=true
            shift
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "E2E Test Runner for Attention Explorer"
echo "============================================================"
echo "Server URL: $SERVER_URL"
echo "Top-K: $TOP_K"
echo "Full validation: $FULL_VALIDATION"
echo "============================================================"
echo ""

# Check if server is running
echo "Checking server connectivity..."
if ! curl -s --max-time 5 "$SERVER_URL/health" > /dev/null 2>&1; then
    if ! curl -s --max-time 5 "$SERVER_URL/v1/models" > /dev/null 2>&1; then
        echo "WARNING: Server may not be reachable at $SERVER_URL"
        echo "Continuing anyway - tests will skip if server is unavailable"
    fi
fi

cd "$PROJECT_DIR"

# Run tests in sequence
echo ""
echo "============================================================"
echo "1. Running Server Connection Tests"
echo "============================================================"
python -m pytest tests/e2e/test_attention_capture.py::TestServerConnection \
    -v --server "$SERVER_URL" --attention-top-k "$TOP_K" \
    || echo "Server connection tests completed (some may have skipped)"

echo ""
echo "============================================================"
echo "2. Running Basic Attention Capture Tests"
echo "============================================================"
python -m pytest tests/e2e/test_attention_capture.py::TestAttentionCapture \
    -v --server "$SERVER_URL" --attention-top-k "$TOP_K" \
    || echo "Attention capture tests completed"

echo ""
echo "============================================================"
echo "3. Running Fingerprint Generation Tests"
echo "============================================================"
python -m pytest tests/e2e/test_attention_capture.py::TestFingerprintGeneration \
    -v --server "$SERVER_URL" --attention-top-k "$TOP_K" \
    || echo "Fingerprint generation tests completed"

echo ""
echo "============================================================"
echo "4. Running Manifold Zone Tests"
echo "============================================================"
python -m pytest tests/e2e/test_attention_capture.py::TestManifoldZones \
    -v --server "$SERVER_URL" --attention-top-k "$TOP_K" \
    || echo "Manifold zone tests completed"

echo ""
echo "============================================================"
echo "5. Running Performance Metrics Tests"
echo "============================================================"
python -m pytest tests/e2e/test_attention_capture.py::TestPerformanceMetrics \
    -v --server "$SERVER_URL" --attention-top-k "$TOP_K" \
    || echo "Performance metrics tests completed"

echo ""
echo "============================================================"
echo "6. Running Fingerprint Pipeline Tests"
echo "============================================================"
python -m pytest tests/e2e/test_fingerprint_pipeline.py::TestFingerprintComputation \
    -v --server "$SERVER_URL" \
    || echo "Fingerprint pipeline tests completed"

echo ""
echo "============================================================"
echo "7. Running Full Pipeline Tests"
echo "============================================================"
python -m pytest tests/e2e/test_fingerprint_pipeline.py::TestFullPipeline \
    -v --server "$SERVER_URL" \
    || echo "Full pipeline tests completed"

if [ "$FULL_VALIDATION" = true ]; then
    echo ""
    echo "============================================================"
    echo "8. Running Full Validation Suite (slow)"
    echo "============================================================"
    python -m pytest tests/e2e/test_attention_capture.py::TestFullValidation \
        -v --server "$SERVER_URL" --attention-top-k "$TOP_K" --full-validation \
        || echo "Full validation tests completed"

    echo ""
    echo "============================================================"
    echo "9. Running Hardware Validation Tests"
    echo "============================================================"
    python -m pytest tests/e2e/test_attention_capture.py::TestHardwareValidation \
        -v --server "$SERVER_URL" --attention-top-k "$TOP_K" --full-validation \
        || echo "Hardware validation tests completed"
fi

echo ""
echo "============================================================"
echo "E2E Tests Complete!"
echo "============================================================"

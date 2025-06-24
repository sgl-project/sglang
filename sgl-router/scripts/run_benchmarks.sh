#!/bin/bash

# SGLang Router Performance Benchmark Runner
# This script runs benchmarks and generates readable output for CI/CD and development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "SGLang Router Benchmark Runner"
echo "=============================="
echo "Project: $PROJECT_DIR"
echo "Timestamp: $(date)"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found. Please install Rust toolchain."
    exit 1
fi

# Parse command line arguments
QUICK_MODE=false
SAVE_BASELINE=""
COMPARE_BASELINE=""
OUTPUT_DIR="target/criterion"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --save-baseline)
            SAVE_BASELINE="$2"
            shift 2
            ;;
        --compare-baseline)
            COMPARE_BASELINE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Run quick benchmarks only"
            echo "  --save-baseline NAME Save results as baseline"
            echo "  --compare-baseline NAME Compare with baseline"
            echo "  --output-dir DIR     Output directory for reports"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure we're in release mode for accurate benchmarks
echo "Building in release mode..."
cargo build --release --quiet

# Run benchmarks with appropriate options
BENCH_ARGS=""
if [ "$QUICK_MODE" = true ]; then
    BENCH_ARGS="$BENCH_ARGS benchmark_summary"
    echo "Running quick benchmarks..."
else
    echo "Running full benchmark suite..."
fi

if [ -n "$SAVE_BASELINE" ]; then
    BENCH_ARGS="$BENCH_ARGS --save-baseline $SAVE_BASELINE"
    echo "Saving baseline as: $SAVE_BASELINE"
fi

if [ -n "$COMPARE_BASELINE" ]; then
    BENCH_ARGS="$BENCH_ARGS --baseline $COMPARE_BASELINE"
    echo "Comparing with baseline: $COMPARE_BASELINE"
fi

# Run the benchmarks
echo ""
cargo bench --bench request_processing $BENCH_ARGS

# Generate summary report
echo ""
echo "Benchmark Summary Report"
echo "========================"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'Unknown')"
echo "Rust version: $(rustc --version)"
echo "Output directory: $OUTPUT_DIR"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Detailed reports available in: $OUTPUT_DIR"
    echo "   - Open $OUTPUT_DIR/request_processing/report/index.html for detailed analysis"
fi

# Performance regression check (if comparing with baseline)
if [ -n "$COMPARE_BASELINE" ]; then
    echo ""
    echo "Performance Regression Analysis"
    echo "==============================="
    echo "Baseline: $COMPARE_BASELINE"
    echo "Current: $(date)"
    echo ""
    echo "Check the detailed HTML report for performance comparisons"
    echo "If you see significant regressions, consider:"
    echo "   - Reviewing recent changes that might affect performance"
    echo "   - Running benchmarks multiple times to confirm results"
    echo "   - Investigating specific bottlenecks in the detailed report"
fi

echo "Benchmarks completed successfully!"

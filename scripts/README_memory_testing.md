# SGLang Memory Leak Testing Infrastructure

This directory contains test infrastructure for tracking and reproducing memory leaks in SGLang, specifically addressing issue [#9365](https://github.com/sgl-project/sglang/issues/9365).

## Overview

The memory leak testing infrastructure includes:

1. **Unit Tests** (`test/srt/test_memory_leak_tracking.py`) - Comprehensive test suite for memory tracking
2. **Memory Monitor Script** (`scripts/memory_leak_monitor.py`) - Standalone monitoring tool
3. **Test Runner** (`scripts/run_memory_tests.py`) - Easy-to-use test runner for specific scenarios

## Quick Start

### Run Specific Tests Requested in Issue Comments

The issue comments specifically requested tests for:
1. A text model
2. A VLM without initializing mm_processor

```bash
# Test 1: Text model memory tracking (addresses comment request #1)
python scripts/run_memory_tests.py --test text-model

# Test 2: VLM without mm_processor usage (addresses comment request #2)
python scripts/run_memory_tests.py --test vlm-no-mm-processor

# Run all tests
python scripts/run_memory_tests.py --test all
```

### Run Unit Tests

```bash
python -m unittest test.srt.test_memory_leak_tracking -v
```

## Detailed Usage

### Memory Monitor Script

The `memory_leak_monitor.py` script provides detailed memory tracking:

```bash
# Monitor text model
python scripts/memory_leak_monitor.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --test-type text \
    --num-requests 100

# Monitor VLM with text-only requests (no mm_processor usage)
python scripts/memory_leak_monitor.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --test-type vlm-text \
    --num-requests 100

# Monitor VLM with image processing
python scripts/memory_leak_monitor.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --test-type vlm-image \
    --num-requests 50
```

### Parameters

- `--model`: Model path (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
- `--test-type`: Type of test (`text`, `vlm-text`, `vlm-image`)
- `--num-requests`: Number of requests to send (default: 100)
- `--request-interval`: Seconds between requests (default: 0.1)
- `--mem-fraction`: Memory fraction for server (default: 0.7)

## Test Types

### 1. Text Model Test (`text`)
- **Purpose**: Track memory usage in pure text models
- **Addresses**: Comment request for "A text model" test
- **Description**: Sends text-only requests to a language model and monitors memory usage
- **Expected**: Stable memory usage with minimal growth

### 2. VLM Text-Only Test (`vlm-text`)
- **Purpose**: Track memory usage in VLM without mm_processor usage
- **Addresses**: Comment request for "A VLM without initializing mm_processor" test
- **Description**: Sends text-only requests to a VLM server (no image processing)
- **Expected**: Similar to text model, but may have higher baseline memory

### 3. VLM Image Processing Test (`vlm-image`)
- **Purpose**: Demonstrate the memory leak with image processing
- **Addresses**: Reference test to show the issue described in #9365
- **Description**: Sends image processing requests to VLM
- **Expected**: May show memory growth/leaks as described in the issue

## Output

Each test produces:

1. **Console Output**: Real-time memory usage and statistics
2. **JSON Data**: Detailed memory readings saved to `/tmp/`
3. **Memory Plots**: Visual graphs of memory usage over time (requires matplotlib)

Example output files:
- `/tmp/text_meta-llama_Llama-3.2-1B-Instruct_1234567890_memory_data.json`
- `/tmp/text_meta-llama_Llama-3.2-1B-Instruct_1234567890_memory_plot.png`

## Dependencies

Install required dependencies:

```bash
pip install matplotlib GPUtil psutil
```

## Integration with CI

The unit tests in `test_memory_leak_tracking.py` can be integrated into the existing CI pipeline:

```bash
python -m unittest test.srt.test_memory_leak_tracking
```

## Understanding Results

### Memory Growth Indicators

- **Stable Memory**: Growth < 100MB over 100 requests (good)
- **Moderate Growth**: Growth 100-500MB over 100 requests (concern)
- **High Growth**: Growth > 500MB over 100 requests (likely leak)

### Interpreting Plots

The generated plots show:
- **Blue Line**: Current GPU memory usage
- **Orange Dashed Line**: Peak memory usage
- **Text Box**: Summary statistics including growth rate

## Contributing

When adding new memory leak tests:

1. Follow the existing pattern in `test_memory_leak_tracking.py`
2. Use the `MemoryTracker` class for consistent monitoring
3. Include appropriate thresholds for pass/fail criteria
4. Document expected behavior and known issues

## Related Issues

- [#9365](https://github.com/sgl-project/sglang/issues/9365) - Main tracking issue for VLM/LLM OOM problems
- Memory leak reports in other SGLang issues

## Troubleshooting

### Common Issues

1. **Server startup timeout**: Increase timeout or use smaller models
2. **GPU memory errors**: Reduce `--mem-fraction` parameter
3. **Missing dependencies**: Install matplotlib, GPUtil, psutil
4. **Port conflicts**: Use different `--port` parameter

### Debugging

Enable verbose logging:
```bash
export SGLANG_LOG_LEVEL=DEBUG
python scripts/run_memory_tests.py --test text-model
```
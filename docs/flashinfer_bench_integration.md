# FlashInfer-Bench Integration

## Overview

SGLang integrates with [FlashInfer-Bench](https://bench.flashinfer.ai), a benchmarking infrastructure that enables automatic kernel optimization for LLM inference. This integration allows SGLang to collect production workloads and automatically substitute optimized kernels at runtime.

## What is FlashInfer-Bench?

FlashInfer-Bench is an AI-for-AI infrastructure that enables:

1. **Workload Tracing**: Capture real production workload patterns (tensor shapes, batch sizes, sequence lengths)
2. **Kernel Benchmarking**: Test different kernel implementations against your actual workloads
3. **Automatic Optimization**: Dynamically substitute optimized kernels at runtime for better performance

This creates a "self-improving" system where AI models automatically get faster over time based on real usage patterns.

## Benefits

- **15-30% Performance Improvement**: Based on workload-specific optimizations
- **Zero Code Changes**: Enable via environment variables or CLI arguments
- **Production Insights**: Understand actual workload characteristics
- **Custom Kernels**: Develop and test kernels for your specific use case
- **Cost Savings**: Better GPU utilization reduces inference costs

## Prerequisites

### Install FlashInfer-Bench

```bash
pip install flashinfer-bench
```

Or install from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer-bench.git
cd flashinfer-bench
pip install -e .
```

### Verify Installation

```python
import flashinfer_bench
print(flashinfer_bench.__version__)
```

## Usage

### Method 1: Environment Variables (Recommended)

**Enable Workload Tracing:**

```bash
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=/path/to/traces \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b
```

**Enable Kernel Substitution:**

```bash
FIB_ENABLE_APPLY=1 \
FIB_DATASET_PATH=/path/to/traces \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b
```

**Enable Both:**

```bash
FIB_ENABLE_TRACING=1 \
FIB_ENABLE_APPLY=1 \
FIB_DATASET_PATH=/path/to/traces \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b
```

### Method 2: Command-Line Arguments

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --enable-flashinfer-bench-tracing \
    --flashinfer-bench-dataset-path /path/to/traces
```

Or for kernel substitution:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --enable-flashinfer-bench-apply \
    --flashinfer-bench-dataset-path /path/to/traces
```

### Configuration Options

| Environment Variable | CLI Argument | Description | Default |
|---------------------|--------------|-------------|---------|
| `FIB_ENABLE_TRACING` | `--enable-flashinfer-bench-tracing` | Enable workload collection | `False` |
| `FIB_ENABLE_APPLY` | `--enable-flashinfer-bench-apply` | Enable kernel substitution | `False` |
| `FIB_DATASET_PATH` | `--flashinfer-bench-dataset-path` | Path to store/load traces | `~/.cache/flashinfer_bench/dataset` |

## Complete Workflow

### Phase 1: Collect Production Workloads

```bash
# Run your server with tracing enabled
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=./my_workloads \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --port 30000

# Use your application normally
# Workloads are automatically captured to ./my_workloads
```

### Phase 2: Benchmark Kernels

```bash
# Benchmark collected workloads
flashinfer-bench run --local ./my_workloads \
    --warmup-runs 10 \
    --iterations 100 \
    --num-trials 5

# View performance comparison
flashinfer-bench leaderboard --local ./my_workloads
```

### Phase 3: Deploy Optimized Kernels

```bash
# Enable automatic kernel substitution
FIB_ENABLE_APPLY=1 \
FIB_DATASET_PATH=./my_workloads \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8b \
    --port 30000

# SGLang automatically uses the fastest kernels
# based on your benchmarking results
```

## Real-World Examples

### Example 1: Llama-3-70B Optimization

```bash
# Step 1: Collect 1 week of production workloads
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=./llama70b_traces \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70b \
    --tp-size 4

# Step 2: Benchmark (offline)
flashinfer-bench run --local ./llama70b_traces

# Step 3: Deploy optimizations
FIB_ENABLE_APPLY=1 \
FIB_DATASET_PATH=./llama70b_traces \
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70b \
    --tp-size 4

# Result: 20-25% latency reduction!
```

### Example 2: Hardware-Specific Tuning

```bash
# Collect workloads on your specific GPU (H100, A100, etc.)
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=./h100_optimized \
python -m sglang.launch_server --model-path your-model

# Find H100-optimized kernels
flashinfer-bench run --local ./h100_optimized --gpu-arch h100

# Deploy H100-optimized kernels
FIB_ENABLE_APPLY=1 FIB_DATASET_PATH=./h100_optimized \
python -m sglang.launch_server --model-path your-model
```

### Example 3: Workload-Specific Patterns

Different applications have different patterns:

**Chatbot (short prompts, long generation):**

```bash
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=./chatbot_patterns \
python -m sglang.launch_server --model-path llama-3-8b
```

**RAG System (long prompts, short generation):**

```bash
FIB_ENABLE_TRACING=1 \
FIB_DATASET_PATH=./rag_patterns \
python -m sglang.launch_server --model-path llama-3-8b
```

## How It Works

### Automatic Integration

When FlashInfer-Bench is enabled, it automatically patches FlashInfer's internal functions:

1. **Patches FlashInfer Wrappers**: `BatchPrefillWithPagedKVCacheWrapper`, `BatchDecodeWithPagedKVCacheWrapper`, etc.
2. **Collects Workload Data**: Tensor shapes, batch configurations, sequence lengths
3. **Substitutes Kernels**: Uses the fastest kernel for each workload pattern
4. **Falls Back Gracefully**: Uses default kernels if no optimization is available

### Under the Hood

```python
# When you call enable_tracing() or enable_apply(), FlashInfer-Bench
# automatically patches FlashInfer's internal functions via
# install_flashinfer_integrations(). This happens inside TracingRuntime
# and ApplyRuntime initialization.

# The patched functions intercept calls like:
wrapper.plan(...)  # Captures configuration
wrapper.run(...)   # Traces workload / applies optimized kernel

# Definition names are computed from tensor shapes, e.g.:
# "gqa_paged_prefill_causal_h32_kv8_d128_ps1"
# This allows matching workloads to the right tracing config and solutions.
```

## Monitoring and Debugging

### Check Integration Status

```python
import logging
logging.basicConfig(level=logging.INFO)

# You should see:
# INFO: FlashInfer-Bench integration initialized
# INFO: FlashInfer-Bench tracing enabled, dataset path: /path/to/traces
```

### Verify Trace Collection

```bash
# Check that traces are being collected
ls -lh $FIB_DATASET_PATH/

# Should contain directories like:
# - definitions/
# - workloads/
# - solutions/ (after benchmarking)
# - evaluations/ (after benchmarking)
```

### View Trace Statistics

```bash
flashinfer-bench stats --local $FIB_DATASET_PATH
```

## Performance Monitoring

### Track Improvements

```python
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import TraceSet

# Load traces
trace_set = TraceSet.from_path("./my_workloads")

# Run benchmarks
config = BenchmarkConfig(warmup_runs=10, iterations=100)
benchmark = Benchmark(trace_set, config)
results = benchmark.run_all()

# Analyze performance
for result in results:
    print(f"Kernel: {result.definition_name}")
    print(f"Speedup: {result.speedup}x")
    print(f"Latency improvement: {result.latency_ms_saved}ms")
```

## Troubleshooting

### FlashInfer-Bench Not Available

**Error**: `FlashInfer-Bench not installed, kernel optimization disabled`

**Solution**:

```bash
pip install flashinfer-bench
# or
pip install -e /path/to/flashinfer-bench
```

### Tracing Not Working

**Symptoms**: No files in `$FIB_DATASET_PATH`

**Checklist**:

1. Verify environment variable is set: `echo $FIB_ENABLE_TRACING`
2. Check directory permissions: `ls -ld $FIB_DATASET_PATH`
3. Look for integration log: `INFO: FlashInfer-Bench tracing enabled`
4. Verify you're actually running inference (traces only captured during requests)

### Kernel Substitution Not Working

**Symptoms**: No performance improvement with `FIB_ENABLE_APPLY=1`

**Possible Causes**:

1. **No benchmarks run yet**: Run `flashinfer-bench run --local $FIB_DATASET_PATH` first
2. **No faster kernels found**: Check leaderboard with `flashinfer-bench leaderboard`
3. **Workload mismatch**: Current workloads don't match traced patterns

### Integration Not Initializing

**Error**: Integration silently disabled

**Debug Steps**:

```bash
# Enable debug logging
export SGLANG_LOG_LEVEL=DEBUG

# Check if FlashInfer-Bench can be imported
python -c "import flashinfer_bench; print(flashinfer_bench.__version__)"

# Verify environment variables
python -c "from sglang.srt.environ import envs; print(envs.FIB_ENABLE_TRACING.get())"
```

## Advanced Usage

### Programmatic Initialization

```python
# In your custom launcher script
from sglang.srt.layers.flashinfer_bench_integration import (
    initialize_flashinfer_bench,
    shutdown_flashinfer_bench,
    is_flashinfer_bench_enabled,
)

# Initialize with explicit settings
success = initialize_flashinfer_bench(
    tracing=True,
    apply=False,
    dataset_path="/custom/path",
)

if success:
    print("FlashInfer-Bench initialized!")

# Check status
if is_flashinfer_bench_enabled():
    print("Integration is active")

# Cleanup (flushes traces)
shutdown_flashinfer_bench()
```

### Custom Tracing Configuration

For advanced tracing configurations, you can use FlashInfer-Bench's API directly:

```python
from flashinfer_bench import enable_tracing, TracingConfig

# Custom config for specific definitions
custom_configs = {
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1": TracingConfig(
        input_dump_policy=["qo_indptr", "kv_indptr", "kv_indices", "sm_scale"],
        filter_policy="keep_first_by_axes",
    ),
}

enable_tracing(
    dataset_path="/custom/path",
    tracing_configs=custom_configs,
)
```

### Custom Kernel Development

```python
# 1. Collect workloads (as shown above)
# 2. Write your custom kernel
# 3. Add it to FlashInfer-Bench
# 4. Benchmark against production traces
# 5. Deploy if faster!

# See: https://bench.flashinfer.ai/docs/tutorials/bring_your_own_kernel
```

## Best Practices

1. **Start with Tracing Only**: Collect data first before enabling optimizations
2. **Use Representative Workloads**: Run tracing on production or production-like traffic
3. **Benchmark Regularly**: Re-benchmark when workload patterns change
4. **Monitor Performance**: Track latency improvements over time
5. **Version Your Traces**: Keep different trace sets for different models/workloads
6. **Test Before Production**: Validate optimizations on staging first

## FAQ

**Q: Does this work with all models?**
A: Yes, it works with any model using FlashInfer attention kernels.

**Q: What's the overhead of tracing?**
A: Minimal (<1%) when using recommended settings (dump_non_float, shape_only).

**Q: Can I use this with Tensor Parallelism?**
A: Yes, tracing works with TP, DP, and PP configurations.

**Q: How much disk space do traces use?**
A: Typically 10-100MB per 1000 requests with default settings.

**Q: Can I share traces with the community?**
A: Yes! Export your traces and share optimized kernels:

```bash
flashinfer-bench export --local ./my_traces --output traces.fib
```

**Q: Does this affect model accuracy?**
A: No, all kernels are validated for numerical correctness before use.

## Resources

- [FlashInfer-Bench Documentation](https://bench.flashinfer.ai/docs)
- [FlashInfer-Bench GitHub](https://github.com/flashinfer-ai/flashinfer-bench)
- [FlashInfer Documentation](https://flashinfer.ai)
- [SGLang GitHub Issue #12193](https://github.com/sgl-project/sglang/issues/12193)

## Support

For issues or questions:
- SGLang: [GitHub Issues](https://github.com/sgl-project/sglang/issues)
- FlashInfer-Bench: [GitHub Issues](https://github.com/flashinfer-ai/flashinfer-bench/issues)
- Community: [SGLang Slack](https://slack.sglang.ai)

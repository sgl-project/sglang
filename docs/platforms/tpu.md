# TPU

SGLang supports high-performance TPU inference through the SGLang-JAX backend, which is specifically optimized for Google Cloud TPUs. The JAX-based implementation delivers exceptional throughput and low latency for Large Language Model (LLM) serving workloads on TPU hardware.

For TPU-specific issues or feature requests, please visit the [sglang-jax GitHub issues page](https://github.com/sgl-project/sglang-jax/issues).

**NOTE:** SGLang TPU support is implemented via the SGLang-JAX backend, a dedicated JAX-based inference engine maintained as a separate repository at [https://github.com/sgl-project/sglang-jax](https://github.com/sgl-project/sglang-jax).

## System Requirements

### Supported TPU Hardware

| TPU Type | HBM Memory | Availability |
|----------|-----------|--------------|
| TPU v6e | 32 GB | Google Cloud |
| TPU v7 | 96 GB per core | Google Cloud |

### Software Requirements

- **Python:** 3.12 or higher
- **JAX:** Latest version with TPU support
- **Environment:** Google Cloud TPU VM or compatible TPU runtime
- **Optional:** SkyPilot for simplified cloud deployment

## Feature Support Matrix

SGLang-JAX provides comprehensive TPU-optimized features for production LLM serving:

| Feature | Support Status | Description |
|---------|---------------|-------------|
| High-Throughput Continuous Batching | ✅ | Dynamic request batching for maximum TPU utilization |
| Radix Tree KV Cache | ✅ | Memory-efficient prefix sharing between requests |
| FlashAttention Backend | ✅ | TPU-optimized attention kernel for long sequences |
| Tensor Parallelism | ✅ | Distribute models across multiple TPU cores |
| Paged Attention | ✅ | Flexible KV cache management with paging |
| Speculative Decoding (EAGLE/EAGLE3) | ✅ | 20-40% throughput improvement for compatible models |
| Chunked Prefill | ✅ | Mixed prefill-decode batching |
| OpenAI-Compatible API | ✅ | Drop-in replacement for OpenAI API |
| Data Parallel Attention | 🚧 | In development - Attention computation with data parallelism |
| Quantization | 🚧 | In development - Model quantization for reduced memory usage |
| Multi-LoRA | 🚧 | In development - Serve multiple LoRA adapters simultaneously |

### Attention Backend Comparison

| Backend | Paged Attention | Spec Decoding | MLA | Sliding Window |
|---------|----------------|---------------|-----|----------------|
| FlashAttention (fa) | ✅ | ✅ | ❌ | ✅ |
| Native | ❌ | ❌ | ❌ | ❌ |

**NOTE:** FlashAttention backend is recommended for production workloads due to superior memory efficiency and performance.

## Optimized Model List

The following models have been tested and optimized for TPU deployment:

| Model Family | Performance Status |
|--------------|-------------------|
| [Qwen 3](https://huggingface.co/Qwen) | ⭐ Recommended for production |
| [Qwen 3 MoE](https://huggingface.co/Qwen) | ⭐ Best performance |
| [Qwen 2](https://huggingface.co/Qwen) | Needs improvement |
| [Qwen 2 MoE](https://huggingface.co/Qwen) | Needs improvement |
| [Qwen 1.5](https://huggingface.co/Qwen) | Needs improvement |
| [Llama/LLaMA](https://huggingface.co/meta-llama) | Needs improvement |
| [Grok-2](https://huggingface.co/xai-org) | Needs improvement |
| [Gemma 2](https://huggingface.co/google) | Verified on TPU |
| Bailing MoE | Needs improvement |

## Installation

### Method 1: Using PyPI (Recommended)

```bash
pip install sglang-jax
```

### Method 2: From Source

```bash
git clone https://github.com/sgl-project/sglang-jax
cd sglang-jax
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e "python[all]"
```

### Method 3: Using Docker

**NOTE:** Docker support for TPU is currently under development. Please use PyPI or source installation methods.

### Method 4: Cloud TPU with SkyPilot

[SkyPilot](https://github.com/skypilot-org/skypilot) provides simplified deployment on Google Cloud TPU:

1. Install SkyPilot and configure GCP access (see [SkyPilot documentation](https://skypilot.readthedocs.io/))

2. Create a SkyPilot configuration file:

<details>
<summary>SkyPilot YAML: <code>sglang-jax.sky.yaml</code></summary>

```yaml
# sglang-jax.sky.yaml
resources:
   accelerators: tpu-v6e-4
   accelerator_args:
      tpu_vm: True
      runtime_version: v2-alpha-tpuv6e

run: |
  git clone https://github.com/sgl-project/sglang-jax.git
  cd sglang-jax
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -e "python[all]"
```

</details>

3. Launch your TPU cluster:

```bash
# Standard deployment
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp

# With spot instances for cost savings
sky launch -c sglang-jax sglang-jax.sky.yaml --infra=gcp --use-spot
```

## Launch of the Serving Engine

### Basic Example: Qwen-7B

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen-7B-Chat \
    --trust-remote-code \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1 \
    --tp-size=4 \
    --device=tpu \
    --random-seed=3 \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=8192 \
    --download-dir=/tmp \
    --dtype=bfloat16 \
    --skip-server-warmup \
    --host 0.0.0.0 \
    --port 30000
```

**Key Parameters Explained:**

1. `JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache` - Enables JIT compilation caching to accelerate server startup on subsequent runs
2. `--tp-size=4` - Tensor parallelism size; match this to your TPU core count (typically 1, 4, or 8)
3. `--device=tpu` - Specifies TPU device (this is the default for sglang-jax)
4. `--dtype=bfloat16` - Uses bfloat16 precision, which TPUs are optimized for
5. `--mem-fraction-static=0.8` - Allocates 80% of TPU HBM for static memory (adjustable from 0.2 to 0.9)
6. `--max-prefill-tokens=8192` - Maximum number of tokens processed in the prefill phase

### High-Performance Configuration: Qwen3-8B

For production workloads with optimal throughput:

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code \
    --tp-size=4 \
    --device=tpu \
    --mem-fraction-static=0.8 \
    --chunked-prefill-size=2048 \
    --dtype=bfloat16 \
    --max-running-requests=256 \
    --page-size=128 \
    --attention-backend=fa
```

### Advanced: Speculative Decoding (EAGLE3)

Speculative decoding can improve throughput by 20-40% for compatible models:

```bash
python3 -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-32B \
    --trust-remote-code \
    --device=tpu \
    --tp-size=4 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=4096 \
    --attention-backend=fa \
    --dtype=bfloat16 \
    --port=30000 \
    --host=0.0.0.0 \
    --disable-overlap-schedule \
    --speculative-algorithm=EAGLE3 \
    --speculative-draft-model-path=AngelSlim/Qwen3-32B_eagle3 \
    --page-size=64 \
    --speculative-eagle-topk=1 \
    --speculative-num-steps=3 \
    --speculative-num-draft-tokens=4
```

**NOTE:** Speculative decoding is currently supported for Qwen3 and LLaMA model families. See the [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) for detailed configuration guidance.


### Multi-Node Distributed Serving

For large models requiring multiple TPU VMs:

```bash
# Node 0 (coordinator)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=0 \
    --tp-size=8 \
    [other parameters...]

# Node 1 (worker)
python3 -m sgl_jax.launch_server \
    --model-path MODEL_PATH \
    --dist-init-addr=NODE0_IP:10011 \
    --nnodes=2 \
    --node-rank=1 \
    --tp-size=8 \
    [other parameters...]
```

## Benchmarking with Requests

### Throughput Testing

Basic throughput benchmark:

```bash
python3 -m sgl_jax.bench_serving \
    --backend sgl-jax \
    --dataset-name random \
    --num-prompts=100 \
    --random-input=512 \
    --random-output=128 \
    --max-concurrency=8 \
    --random-range-ratio=1 \
    --warmup-requests=0
```

### Latency Testing

Measure single-batch latency:

```bash
python3 -m sgl_jax.bench_one_batch_server \
    --base-url http://127.0.0.1:30000 \
    --model-path Qwen/Qwen-7B-Chat \
    --batch-size=32 \
    --input-len=256 \
    --output-len=32
```

### Comprehensive Benchmark Script

For systematic performance evaluation across different configurations:

```bash
#!/bin/bash
set -e

backend=${1:-sgl-jax}
num_prompts_per_concurrency=3
input_seq_lens=(1024 4096 8192)
output_seq_lens=(1 1024)
max_concurrencies=(8 16 32 64 128 256)

for input_seq_len in "${input_seq_lens[@]}"; do
    for output_seq_len in "${output_seq_lens[@]}"; do
        echo "======================================="
        echo "Testing ISL/OSL: $input_seq_len/$output_seq_len"
        echo "======================================="
        for max_concurrency in "${max_concurrencies[@]}"; do
            num_prompts=$((num_prompts_per_concurrency * max_concurrency))
            python3 -m sgl_jax.bench_serving \
                --backend ${backend} \
                --dataset-name random \
                --num-prompts ${num_prompts} \
                --random-input ${input_seq_len} \
                --random-output ${output_seq_len} \
                --max-concurrency ${max_concurrency} \
                --random-range-ratio 1 \
                --disable-ignore-eos \
                --warmup-requests 0
        done
    done
done
```

For detailed help on all benchmark parameters:

```bash
python3 -m sgl_jax.bench_serving --help
```

See the [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) for advanced benchmarking techniques and profiling with JAX Profiler.

## Performance Optimization

### Memory Optimization

**Reduce memory usage:**
- Lower `--mem-fraction-static` (from 0.8 → 0.5 → 0.3)
- Decrease `--max-prefill-tokens` (from 16384 → 8192 → 4096)
- Reduce `--max-running-requests`

**Handle OOM errors:**
- Start with conservative memory settings (`--mem-fraction-static=0.5`)
- Gradually increase until you find the optimal balance
- Increase `--page-size` for better memory locality (1 → 16 → 64 → 128)

### Throughput Optimization

To maximize tokens per second:

- Use FlashAttention backend: `--attention-backend=fa`
- Enable speculative decoding (EAGLE3) for Qwen3 models (20-40% improvement)
- Increase `--max-running-requests` to 256+
- Set `--mem-fraction-static` to 0.8+ (if memory allows)
- Use larger page sizes (64-128)
- Enable chunked prefill: `--chunked-prefill-size=2048`

### Latency Optimization

To minimize time-to-first-token (TTFT) and inter-token latency:

- Reduce `--page-size` to 1-4
- Lower `--max-running-requests` (16-32) for smaller batches
- Reduce `--chunked-prefill-size`
- Use conservative memory settings to avoid GC pauses

### TPU-Specific Optimizations

1. **JIT Compilation Cache:**
   ```bash
   export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
   ```
   Always set this environment variable to cache compiled kernels and accelerate server startup.

2. **Data Type Optimization:**
   Use `--dtype=bfloat16` for TPU native optimization. TPUs are specifically designed for bfloat16 computations.

3. **Tensor Parallelism:**
   Match `--tp-size` to your TPU core configuration (1, 4, or 8) for optimal model distribution.

4. **Attention Backend:**
   Always use `--attention-backend=fa` (FlashAttention) for production workloads.

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter out-of-memory errors:

1. Reduce `--mem-fraction-static` from 0.8 to 0.5 or lower
2. Decrease `--max-prefill-tokens` from 8192 to 4096 or 2048
3. Lower `--max-running-requests` to reduce concurrent batch size
4. Increase `--page-size` for better memory layout efficiency

### Compilation Long-Time

If the server takes too long to start:

1. Ensure `JAX_COMPILATION_CACHE_DIR` is properly set
2. Understand that the first run requires JIT compilation (this is normal)
3. Subsequent runs will be significantly faster with cached compilations
4. Consider using `--skip-server-warmup` to defer compilation until first request

### Low Throughput

If you're not achieving expected throughput:

1. Verify `--tp-size` matches your TPU core configuration
2. Check that `--attention-backend=fa` is enabled
3. Increase `--max-running-requests` to enable larger batch formation
4. Consider enabling speculative decoding for compatible models
5. Ensure memory settings allow for sufficient batch sizes

### Connection Issues

If clients cannot connect to the server:

1. Ensure `--host=0.0.0.0` for external access (not just `127.0.0.1`)
2. Verify firewall rules allow traffic on the specified port (default: 30000)
3. Check that the server process is running: `curl http://localhost:30000/health`

## Advanced Features

### Speculative Decoding

SGLang-JAX supports EAGLE and EAGLE3 speculative decoding algorithms for Qwen3 and LLaMA model families. Speculative decoding can improve throughput by 20-40% without affecting output quality.

See the [Speculative Decoding documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md) for detailed configuration and supported model combinations.

### Chunked Prefill

Enable mixed prefill-decode batching for better TPU utilization:

```bash
--chunked-prefill-size=2048 --enable-mixed-chunk
```

This allows the scheduler to mix prefill operations with decode operations in the same batch, improving overall throughput.

### Custom Attention Backends

SGLang-JAX supports a plugin-based attention backend system. You can implement custom attention kernels optimized for specific use cases.

See the [Attention Backend documentation](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/attention_backend.md) for implementation details.

### Environment Verification

Verify your TPU setup before deploying:

```bash
python -c "from sgl_jax import check_env; check_env.check_env()"
```

This command checks:
- Installed package versions
- TPU device availability and specifications
- System resources and configuration
- Compatibility of settings

## Contributing

We welcome contributions to improve TPU support in SGLang-JAX!

### Areas for Contribution

**Check the [Development Roadmap](https://github.com/sgl-project/sglang-jax/issues/190)** to see planned features and find opportunities to contribute new functionality.

Current contribution areas include:

- Performance optimizations for specific TPU generations
- Support for additional model architectures
- Documentation improvements and examples
- Bug reports and fixes
- Benchmark results and performance analysis

### How to Contribute

1. Visit the [sglang-jax repository](https://github.com/sgl-project/sglang-jax)
2. Read the [Contribution Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md)
3. Join the [SGL-JAX Slack community](https://sgl-fru7574.slack.com/archives/C09EBE5HT5X) for discussions
4. Report issues at [sglang-jax/issues](https://github.com/sgl-project/sglang-jax/issues)

### Testing on TPU

For contributors who need TPU access for testing:

- Refer to the [TPU Resources Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/tpu_resources_guide.md) for information on accessing TPU hardware
- Use SkyPilot with spot instances for cost-effective testing
- Follow the [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md) for performance validation

## References

### Documentation

- [SGLang-JAX Repository](https://github.com/sgl-project/sglang-jax)
- [SGLang-JAX Installation Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/get_started/install.md)
- [Qwen Models Quick Start](https://github.com/sgl-project/sglang-jax/blob/main/docs/basic_usage/qwen.md)
- [Benchmark and Profiling Guide](https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/benchmark_and_profiling.md)
- [Speculative Decoding](https://github.com/sgl-project/sglang-jax/blob/main/docs/features/speculative_decoding.md)

### External Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [SkyPilot Documentation](https://skypilot.readthedocs.io/)

# Quantized KV Cache

Quantized KV cache reduces the memory footprint of key-value cache storage by using lower-precision data types (FP8 or FP4) instead of the default model precision in BF16. During autoregressive generation, LLMs cache previously computed key-value pairs to avoid redundant calculations. The KV cache typically consumes a significant portion of GPU memory, especially for long sequences.

Quantized KV cache is a memory optimization technique that primarily benefits throughput by allowing more tokens to be cached, but may introduce minimal accuracy degradation depending on the quantization format used.

```{warning}
**Performance Warning**: When quantized KV cache must be dequantized before use in attention operations, performance can be extremely slow if dequantization is not fused with the attention kernel. Always verify that your chosen attention backend supports quantized KV cache. Backends without fused support may experience significant throughput degradation, potentially negating the memory benefits.

**Backend Support**: Not all attention backends support quantized KV cache. Refer to [Attention Backend](attention_backend.md) for which backends support it.
```

## Supported Formats

SGLang supports the following quantized KV cache formats:

### FP8 Format

[OCP (Open Compute Project)](https://www.opencompute.org) specifies two common 8-bit floating point formats:

- **E5M2** (5 exponent bits, 2 mantissa bits): Larger dynamic range (±57344.0), lower precision
- **E4M3** (4 exponent bits, 3 mantissa bits): Higher precision, smaller dynamic range (±240.0)

### FP4 Format

```{warning}
FP4 quantization is currently experimental.
```

[OCP (Open Compute Project)](https://www.opencompute.org) specifies MXFP4 (Microscaling FP4), a 4-bit floating-point format:

- **E2M1** (1 sign bit, 2 exponent bits, 1 mantissa bit): Uses block-based microscaling where tensors are divided into blocks of 32 consecutive elements, with each block sharing a single 8-bit exponential scaling factor

## Usage

### Enabling Quantized KV Cache

To enable quantized KV cache, use the `--kv-cache-dtype` argument when launching the server:

```bash
# Enable FP8 E5M2 KV cache
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e5m2 \

# Enable FP8 E4M3 KV cache
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-R1-0528 \
    --kv-cache-dtype fp8_e4m3 \

# Enable FP4 E2M1 KV cache
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --kv-cache-dtype fp4_e2m1 \
```

### Scaling Factors

FP8 quantization requires scaling factors to properly quantize and dequantize the KV cache.

```{note}
Currently, only per-tensor (scalar) scaling factors are supported.
```

Scaling factors can be:

- **Loaded from checkpoints**: Pre-quantized models (e.g., ModelOpt) may include `k_scale` and `v_scale` parameters that are automatically loaded
- **Provided via JSON**: Supply scaling factors via `--quantization-param-path`.

The JSON file should follow this format:

```json
{
  "kv_cache": {
    "dtype": "float8_e4m3fn",
    "scaling_factor": {
      "0": {
        "0": 1.0,
        "1": 1.0
      }
    }
  }
}
```

Where the outer keys in `scaling_factor` are tensor parallel ranks and inner keys are layer indices.

```{warning}
If scaling factors are not provided and not found in the checkpoint, it will default to 1.0, which may cause accuracy issues.
```

## Performance Considerations

### Memory Savings

Quantized KV cache provides significant memory savings:
- **BF16 → FP8**: 2x reduction (16 bits → 8 bits)
- **BF16 → FP4**: 4x reduction (16 bits → 4 bits)

This enables longer context lengths or more concurrent requests within the same memory budget.

### Accuracy Impact

FP8 E4M3 quantization typically introduces minimal accuracy degradation. The impact depends on model architecture, sequence length, and quantization format (generally, E4M3 has better accuracy than E5M2).

## Best Practices

- **Use pre-quantized models**: Prefer models quantized offline with scaling factors included in the checkpoint.
- **Choose the right format**: Use `fp8_e4m3` for better accuracy (recommended), `fp8_e5m2` for larger dynamic range, or `fp4_e2m1` for maximum memory savings (experimental)
- **Check backend compatibility**: Verify that your chosen attention backend supports quantized KV cache

```{seealso}
- [Quantization](quantization.md)
- [Attention Backend](attention_backend.md)
- [Server Arguments](server_arguments.md)
```

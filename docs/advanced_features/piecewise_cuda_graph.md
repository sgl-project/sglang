## Piecewise CUDA Graph

```{warning}
This is an experimental feature.
```

SGLang's regular CUDA graph optimization only captures the decoding step. Piecewise CUDA graph extends this by capturing prefill/extend. It can improve the TTFT of prefill-heavy workloads by 10-50%.

## What it does

- Captures a set of CUDA graphs for prefill/extend with different token counts.
- At runtime, for an extend step with `num_tokens`, it selects the smallest captured size `static_num_tokens` such that `static_num_tokens >= num_tokens`.

## When to use it

- Best for: workloads with large prefill/extend batches.
- Less useful for: decode-heavy workloads (regular CUDA graph already helps there).


## How to enable

You can enable piecewise CUDA graph by adding `--enable-piecewise-cuda-graph` when launching the server.

```bash
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3.1 \
  --enable-piecewise-cuda-graph
```

```{note}
Piecewise CUDA graph is a prefill-only optimization. Decode CUDA graph behavior is controlled separately by `--disable-cuda-graph`.
```

## Tuning knobs

All flags below are server launch arguments.

- **`--piecewise-cuda-graph-max-tokens`**: maximum token count to capture.
  - You can lower this to reduce startup time and memory overhead.

```{note}
The system automatically falls back to the normal (non-piecewise) path when requests exceed `--piecewise-cuda-graph-max-tokens`.
```

- **`--piecewise-cuda-graph-tokens`**: explicit JSON list of token sizes to capture.
  - If not set, SGLang auto-generates a token list up to `--piecewise-cuda-graph-max-tokens`.

- **`--piecewise-cuda-graph-compiler`**: compilation backend used by the piecewise pipeline.
  - Choices: `eager`, `inductor`.
  - Default: `eager`.

- **`--mem-fraction-static`**: leaves headroom for model weights/KV cache/graphs.

```{tip}
If capture fails with OOM errors, try lowering `--mem-fraction-static` (for example, `0.8` or `0.7`).
```

## Supported features

### Quantization
- FP8 (per-token and ModelOpt)
- NVFP4
- INT8
- W4A8
- GPTQ, AWQ

### Attention backends
- MLA backends
- SWA (GPT-OSS)
- Hybrid Linear Attention (Qwen3-Next, Kimi-Linear)

## Limitations / compatibility

SGLang disables piecewise CUDA graph automatically (or you can't use it) in these cases:

- Pipeline parallelism: not supported when `--pp > 1`.
- Expert Parallelism (EP): not supported.
- Data Parallelism Attention: not supported.
- torch.compile (global): not supported together with `--enable-torch-compile`.

## Troubleshooting

### Capture fails / OOM during capture

You can try lowering `--mem-fraction-static` (for example, `0.8`) or reducing `--piecewise-cuda-graph-max-tokens` (for example, `1024`).

## Known working models

Piecewise CUDA graph has been tested and verified to work with:

- GLM-4.5, GLM-4.6 (Glm4MoeForCausalLM)
- DeepSeek V3/R1
- Qwen3 series (including MoE variants)
- GPT-OSS
- Qwen2.5-VL

```{note}
For models not listed here, piecewise CUDA graph may still work but has not been explicitly tested. If you encounter issues with a specific model, please report them on GitHub.
```

## Related

- Regular CUDA graph arguments: `--disable-cuda-graph`, `--cuda-graph-max-bs`, `--cuda-graph-bs`.
- For development roadmap, known issues, and model compatibility status, see the [Piecewise CUDA Graph roadmap issue](https://github.com/sgl-project/sglang/issues/11490).

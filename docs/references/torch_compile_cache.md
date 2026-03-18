# Enabling cache for torch.compile

SGLang uses `max-autotune-no-cudagraphs` mode of torch.compile. The auto-tuning can be slow.
If you want to deploy a model on many different machines, you can ship the torch.compile cache to these machines and skip the compilation steps.

This is based on https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

## Unified cache root (recommended)

All JIT and precompilation caches (Triton, Inductor, torch\_compile, DeepGEMM) are stored under a single configurable root directory controlled by `SGLANG_JIT_CACHE_ROOT` (default: `~/.cache/sglang`).

```
SGLANG_JIT_CACHE_ROOT=/data/sglang_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
```

The layout under the root is:

```
$SGLANG_JIT_CACHE_ROOT/
  torch_compile/    # SGLang torch.compile hash-based cache
  deep_gemm/        # DeepGEMM JIT kernels
```

Triton and Inductor caches are managed inside the `torch_compile/<hash>/` subtree automatically.

## Legacy per-component overrides

You can still use `TORCHINDUCTOR_CACHE_DIR` or other per-component env vars. They take precedence over the unified root.

1. Generate the cache by setting TORCHINDUCTOR_CACHE_DIR and running the model once.
```
TORCHINDUCTOR_CACHE_DIR=/root/inductor_root_cache python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile
```
2. Copy the cache folder to other machines and launch the server with `TORCHINDUCTOR_CACHE_DIR`.

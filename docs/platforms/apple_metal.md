# Apple Silicon with Metal (MLX)

This document describes how run SGLang on Apple Silicon using [Metal (MLX)](https://opensource.apple.com/projects/mlx/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create a Python 3.11+ virtual environment (uv shown; works with venv too)
uv venv -p 3.11 .venv-mlx
source .venv-mlx/bin/activate

# Switch to the MPS-flavoured pyproject so optional MLX dependencies
# resolve correctly.
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install --upgrade pip
uv pip install -e "python[all_mps]"
```

Set `SGLANG_USE_MLX=1` when launching the server to route through the MLX
backend, e.g.:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --port 30000 \
    --host 127.0.0.1
```

## (Optional but recommended) Build the AOT Metal RoPE kernel

`sgl-kernel` ships a fused **NeoX RoPE + KV-pool scatter** Metal kernel
that is used by default when present. It is AOT-compiled via Apple's
`xcrun metal` / `xcrun metallib` toolchain into a `.metallib` that's
loaded by an `mlx::core::Primitive` subclass at server start.

When the kernel is **not** built, the MLX backend transparently falls
back to MLX's built-in `mx.fast.rope` — the server still runs, just
without the fused-pool optimisation.

### Prerequisites for building

1. **Full Xcode** (not just Command Line Tools — the Metal compiler
   ships with Xcode):

   ```bash
   # Download Xcode from the Mac App Store, then point xcode-select at it
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```

2. **Apple Metal Toolchain** (separate component on recent Xcode):

   ```bash
   sudo xcodebuild -downloadComponent MetalToolchain
   ```

   Verify that both `metal` and `metallib` are now reachable:

   ```bash
   TOOLCHAINS=metal xcrun -sdk macosx --find metal
   TOOLCHAINS=metal xcrun -sdk macosx --find metallib
   ```

### Build

From the repo root, with the same virtual environment active:

```bash
TOOLCHAINS=metal python sgl-kernel/setup_metal.py
```

The build:

* compiles `sgl-kernel/csrc/metal/rope_pool_fused.metal` →
  `sgl_metal_kernels.metallib` via `xcrun metal` + `xcrun metallib`,
* compiles `sgl-kernel/csrc/metal/rope_pool_fused.cpp` (an MLX
  `Primitive` + nanobind binding) into `sgl_kernel/_metal.so`,
* drops both artifacts inside `sgl-kernel/python/sgl_kernel/`.

If the toolchain is missing the build aborts with a clear error message
pointing at the install steps above.

### Disabling the AOT kernel

Set `SGLANG_DISABLE_CUSTOM_ROPE=1` to force the MLX `mx.fast.rope`
fallback even when the kernel is available. Useful for A/B testing.

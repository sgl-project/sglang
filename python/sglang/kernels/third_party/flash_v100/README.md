# Optional grouped Page4 QSA dependency

`scripts/setup_v100_grouped_qsa.py` builds only the grouped planner and attention
bindings from the Flash-V100 sources bundled in
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM/tree/d8f42b391af06410ab01bb71657ededf7dce7799/flash-attention-v100).
The installed binary is a local build artifact and is not committed. It is
written to SGLang's SM70 prebuilt directory
(`python/sglang/kernels/prebuilt/_sm70_grouped_qsa_v100.so`, resolved through
`sglang.kernels.sm70_paths` and overridable with `SGLANG_SM70_PREBUILT_DIR`).

The underlying Flash-V100 project is by D. Skryabin / ai-bond. The grouped Page4
work was contributed by yangzhuxinyzx in
[1Cat-vLLM #387](https://github.com/1CatAI/1Cat-vLLM/pull/387).
The BSD-3-Clause and upstream Apache-2.0 license texts are retained here.
SGLang's adapter uses the library through its planner/attention API and keeps
the existing native QSA path available.

The shared library must use the same PyTorch ABI as the serving environment.
Source and output hashes are written to the build directory's `metadata.json`.

Opt in with `SGLANG_SM70_QSA_GROUPED_PREFILL=1`. The adapter is limited to the
measured 2,048-row FP16/Page16 Qwen shape; a missing library, another shape, or
the flag being unset keeps the native QSA prefill.

## Optional vision attention

`--mm-attention-backend fa2_v100` (or `SGLANG_SM70_VISION_ATTENTION=1`, which
only replaces the platform default) uses the separate `flash_attn_v100` Python
package from [ai-bond/flash-attention-v100](https://github.com/ai-bond/flash-attention-v100).
The validated host build uses revision `d89800edf608d85744f3ab6188be5fd0736acf39`
with `vision_host_compat.patch`, which records the existing host's CUDA 12.8,
packaging, TorchBind optional-tensor and paged-layout compatibility changes.
These are separate from the grouped QSA patch above. The vision adapter uses
non-paged, bidirectional variable-length attention.

Qwen3.8 has 72-wide vision heads. The adapter zero-pads them to the supported
128-wide kernel, retains the original attention scale, and crops the output.
It requires SM70, FP16 and eager vision execution. Language CUDA graphs remain
enabled. Numerical tests cover ragged images, custom scaling, noncontiguous
inputs and bidirectional attention without mixing images.

Build in the serving environment, with its existing PyTorch installation:

```bash
conda activate sglang-v100
q38_repo="$PWD"  # Run from the SGLang checkout.
git clone https://github.com/ai-bond/flash-attention-v100.git flash-attention-v100-vision
git -C flash-attention-v100-vision checkout --detach d89800edf608d85744f3ab6188be5fd0736acf39
git -C flash-attention-v100-vision apply \
  "$q38_repo/python/sglang/kernels/third_party/flash_v100/vision_host_compat.patch"
MAX_JOBS=2 NVCC_THREADS=2 python -m pip install --no-deps --no-build-isolation \
  ./flash-attention-v100-vision
PYTHONPATH=python python -m pytest -q test/manual/layers/test_sm70_vision_attention.py
```

The performance evidence records the measured binary and interface hashes.
This round reused that installed library; it did not rebuild a Docker image.
The in-tree Triton vision path remains available with
`--mm-attention-backend triton_attn`, and it is what
`_determine_attention_backend` keeps when the package is absent.

# Vendored DeepGEMM wheel

`deep_gemm-2.5.0-cp312-cp312-linux_x86_64.whl` is a custom build of DeepGEMM
that ships with this `fp4-acts-pre-dispatch` branch. It contains:

- The FP4 acts + MXF4 kind code paths (kernel headers `sm100_fp8_fp4_mega_moe.cuh`
  with `kUseFp4Acts` / `kUseMxf4Kind` template flags).
- A new fused `mega_moe_pre_dispatch` kernel that replaces the unfused
  BF16 → quant + topk-copy + pad-fill chain in the deepseek_v4 forward path.
- A heuristic fix bumping `block_m=16 → 32` when `kUseMxf4Kind=true` to
  satisfy the 1024-byte smem alignment static_assert at small batches.
- Cherry-pick of upstream sgl-project/DeepGEMM `003ed71b` (relax NVLink
  barrier timeout 30s → 180s).

## Install

```bash
pip install /path/to/sglang/python/sglang/srt/_vendor/deep_gemm-2.5.0-cp312-cp312-linux_x86_64.whl \
    --force-reinstall --no-deps
```

The `--force-reinstall --no-deps` is important: another package's transitive
deps may have installed a different DeepGEMM. Without `--force-reinstall`, pip
will skip our wheel; without `--no-deps`, pip may pull a different version
of torch/cuda libs.

## Verify

```bash
python -c "import deep_gemm; print(hasattr(deep_gemm, 'mega_moe_pre_dispatch'))"
```

Should print `True`. If `False`, the wheel didn't install correctly — check
that no other Python install path shadows it.

## Compatibility

- Python: cp312 only (3.12)
- CUDA: 13.0+ (uses `nvidia-cuda-cccl` headers via the system cuda symlinks
  that the deepseek_v4 dockerfile sets up).
- GPU: `sm_100` (B100/B200/B300). Other archs untested.

## Source

The wheel is built from `pranjalssh/kernels@worktree-hazy-noodling-giraffe`,
specifically the `DeepGEMM/` subtree. To rebuild from source:

```bash
git clone -b worktree-hazy-noodling-giraffe https://github.com/pranjalssh/kernels.git
cd kernels/DeepGEMM
ln -sf $(pwd)/third-party/cutlass/include/cutlass $(pwd)/deep_gemm/include/cutlass
ln -sf $(pwd)/third-party/cutlass/include/cute    $(pwd)/deep_gemm/include/cute
DG_USE_LOCAL_VERSION=0 bash install.sh
```

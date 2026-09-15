# Temporary NPU kernels

Stage NPU Triton kernels here before migrating them to `sgl-kernel-npu`.
Qwen3.8-Flash-Next kernels belong in `qwen3_8_flash_next/`, split by operation.

- Keep kernel implementations, launch wrappers, and capability checks here;
  retain model orchestration, metadata, and Torch references in their existing
  modules.
- Import wrappers lazily from the owning layer's NPU branch. Use the Torch
  fallback for unsupported configurations; do not hide kernel execution errors.
- Validate correctness against Torch, graph capture/replay, and performance
  before enabling an optimized path. Use `NPU_ARCH` for generation checks.
- When migrating, switch to the external implementation and remove the local
  copy together. Keep one Triton source per operation, without local/external
  import probing.

`qwen3_8_flash_next/sparse_attention.py` implements sparse GQA over physical slots.

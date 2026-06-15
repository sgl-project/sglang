FlashInfer TRTLLM MoE Overlay
=============================

This directory contains only the editable overlay files used by SGLang's
`experimental_sgl_trtllm` SM100 TRTLLM fused MoE backend. Unmodified FlashInfer
and TRTLLM sources are compiled from the installed `flashinfer` package at JIT
time.

Local overlay source:

- `data/csrc/trtllm_fused_moe_kernel_launcher.cu`
- `data/csrc/trtllm_fused_moe_runner.cu`
- `data/csrc/fused_moe/trtllm_backend/trtllm_fused_moe_dev_kernel.cu`
- `data/include/flashinfer/trtllm/fused_moe/DevKernel.h`
- `data/include/flashinfer/trtllm/fused_moe/runner.h`

The backend still depends on the installed `flashinfer` and `flashinfer_cubin`
packages for the rest of the FlashInfer/TRTLLM JIT source tree and TRTLLM-Gen
BMM cubin artifacts. The local include directory is passed before FlashInfer's
installed include directory so these overlay headers shadow the originals.

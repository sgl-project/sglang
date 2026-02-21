from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDAExtension

root = Path(__file__).parent.resolve()

include_dirs = [
    root / "include",
    root / "include" / "impl",
    root / "csrc",
]

sources = [
    "csrc/allreduce/custom_all_reduce.hip",
    "csrc/allreduce/deterministic_all_reduce.hip",
    "csrc/allreduce/quick_all_reduce.cu",
    "csrc/common_extension_rocm.cc",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/elementwise/topk.cu",
    "csrc/grammar/apply_token_bitmask_inplace_cuda.cu",
    "csrc/kvcacheio/transfer.cu",
    "csrc/moe/moe_align_kernel.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/speculative/eagle_utils.cu",
]

libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]

ext_modules = [
    CUDAExtension(
        name="sgl_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        py_limited_api=False,
    ),
]

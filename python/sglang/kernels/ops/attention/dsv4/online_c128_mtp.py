from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.attention.dsv4.utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_online_c128_mtp_module(
    head_dim: int,
    seq_dtype: torch.dtype,
    req_dtype: torch.dtype,
    dtype_buffer: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, seq_dtype, req_dtype, dtype_buffer)
    return load_jit(
        make_name(f"online_c128_mtp_{head_dim}"),
        *args,
        cuda_files=["deepseek_v4/online_c128_mtp.cuh"],
        cuda_wrappers=[
            ("write_prefix_states", f"OnlineC128MTPWritePrefixKernel<{args}>::run"),
            ("mark_pending", f"OnlineC128MTPMarkPendingKernel<{args}>::run"),
            ("commit_pending", f"OnlineC128MTPCommitPendingKernel<{args}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )

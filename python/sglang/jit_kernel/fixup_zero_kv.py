from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fixup_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fixup_zero_kv",
        *args,
        cuda_files=["attention/fixup_zero_kv.cuh"],
        cuda_wrappers=[("fixup_zero_kv_rows", f"fixup_zero_kv_rows<{args}>")],
    )


def fixup_zero_kv_rows(
    out: torch.Tensor,
    lse: torch.Tensor,
    kv_lens: torch.Tensor,
    cum_seq_lens: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Fix output and LSE for zero-KV rows after TRT-LLM ragged attention.

    For sequences with kv_lens[i] == 0, sets out[tokens_i] = 0 and
    lse[tokens_i] = -inf.  Single CUDA kernel launch, no GPU-CPU sync.

    Args:
        out:          [total_tokens, num_heads, v_head_dim]  bf16/fp16
        lse:          [total_tokens, num_heads]               float32
        kv_lens:      [batch_size]                            int32
        cum_seq_lens: [batch_size + 1]                        int32
        max_seq_len:  max Q tokens in any single sequence     int
    """
    module = _jit_fixup_module(out.dtype)
    module.fixup_zero_kv_rows(out, lse, kv_lens, cum_seq_lens, max_seq_len)

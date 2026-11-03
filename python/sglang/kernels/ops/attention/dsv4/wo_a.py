from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, cuda_stubs_dir, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

GROUPS = (2, 4)
RANK = 1024
MAX_M = 32


def can_use_fused_wo_a(n_local_groups: int, o_lora_rank: int) -> bool:
    return n_local_groups in GROUPS and o_lora_rank == RANK


@cache_once
def _jit_module(max_tokens: int, groups: int) -> Module:
    return load_jit(
        f"fused_rope_wo_a_g{groups}_m{max_tokens}",
        cuda_files=["deepseek_v4/wo_a_fused.cuh"],
        cuda_wrappers=[("run", f"wo_a_fused_run<{max_tokens}, {groups}>")],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda"],
    )


def fused_rope_wo_a_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    *,
    out_mxfp8: bool = True,
) -> Sequence[torch.Tensor]:
    """Inverse RoPE + BF16 WO-A, returning BF16 or FlashInfer-swizzled MXFP8."""
    num_tokens = x.shape[0]
    groups = weight.shape[0]
    assert num_tokens <= MAX_M
    assert groups in GROUPS
    n_out = groups * RANK
    max_tokens = 16 if num_tokens <= 16 else 32
    module = _jit_module(max_tokens, groups)
    if out_mxfp8:
        q = torch.empty((num_tokens, n_out), dtype=torch.float8_e4m3fn, device=x.device)
        # One UE8M0 byte per 32-column block, for 128 padded token rows.
        scales = torch.empty(n_out // 32 * 128, dtype=torch.uint8, device=x.device)
        module.run(x, weight, freqs_cis, positions, q, scales, None)
        return q, scales
    else:
        y = x.new_empty((num_tokens, n_out))
        module.run(x, weight, freqs_cis, positions, None, None, y)
        return (y,)

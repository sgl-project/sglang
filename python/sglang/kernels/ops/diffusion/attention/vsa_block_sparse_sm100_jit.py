"""FastVideo's native Blackwell block-sparse VSA forward (64-token tiles).

A warp-specialized tcgen05 pipeline in which a CTA owns two adjacent query tiles
with their own key lists. It reaches ~1.06 PFLOPS on B300 against ~0.62 for the
Triton tile-64 kernel on the same inputs; outputs differ from it at bf16
rounding level (max one ulp, ~50% of elements bit-exact). Requires sm_100a or
sm_103a, bf16, head_dim 128, an even tile count and contiguous [B, H, S, D].
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
_HEAD_DIM = 128
_BLOCK = 64


@cache_once
def _module() -> Module:
    return load_jit(
        "vsa_block_sparse_sm100",
        cuda_files=["diffusion/vsa_block_sparse_sm100.cuh"],
        cuda_wrappers=[("vsa_block_sparse_sm100", "VsaBlockSparseSm100Kernel::run")],
        extra_ldflags=["-lcuda"],
    )


@cache_once
def can_use_vsa_block_sparse_sm100(
    device_index: int, dtype: torch.dtype, head_dim: int
) -> bool:
    if dtype != torch.bfloat16 or head_dim != _HEAD_DIM:
        return False
    if torch.cuda.get_device_capability(device_index) not in _SUPPORTED_CAPABILITIES:
        return False
    try:
        _module()
    except Exception as e:
        logger.warning("Failed to load the JIT VSA block-sparse sm100 kernel: %s", e)
        return False
    return True


@register_custom_op(mutates_args=["out"])
def vsa_block_sparse_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    block_sizes: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
) -> None:
    """``out[b, h, i*64:(i+1)*64]`` = attention of query tile ``i`` over its
    ``q2k_num`` listed key tiles, keys past each tile's ``block_sizes`` masked."""
    _module().vsa_block_sparse_sm100(
        q, k, v, q2k_idx, q2k_num, block_sizes, out, sm_scale
    )

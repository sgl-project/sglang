# SPDX-License-Identifier: Apache-2.0
"""aiter's Triton MXFP4 dense GEMM (gemm_afp4wfp4) on gfx950, with a tiling that keeps a masked K tail
out of an LDS stage no full K block filled.

Workaround for triton-lang/triton#11256: a masked K-tail scale load keeps stale LDS (NaN). Drop this
module, and call aiter's gemm_afp4wfp4 directly, once every ROCm image's Triton has the fix, which
3.7.0 and 3.8.0+git4cff872c lack.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

import torch

from sglang.kernels.arch import current_arch

logger = logging.getLogger(__name__)

# gemm_afp4wfp4 raises a Triton BLOCK_SIZE_K below 128 back to 128.
_MIN_BLOCK_K = 128


def k_tail_in_unfilled_stage(block_k: int, k: int, num_stages: int) -> bool:
    """Whether ``k`` FP4 elements leave a masked K-tail block in an LDS stage that no full K block filled."""
    # The pipeline cycles K blocks through num_stages LDS buffers (Triton 3.7.0 TTGIR). With more K blocks
    # than stages, the tail's stale lanes hold an earlier block's finite scales over zero-filled data;
    # otherwise they hold what the previous kernel left: every NaN of an MI355X sweep (N=7168, K <= 8192).
    return k % block_k != 0 and -(-k // block_k) <= num_stages


def k_tail_safe_tiling(
    block_k: int, k: int, num_stages: int
) -> Optional[tuple[int, int]]:
    """Widest BLOCK_SIZE_K, then most num_stages, that keep the K tail out of an unfilled LDS stage."""
    # gemm_afp4wfp4 shrinks a block wider than K to K's next power of two, then raises it to the minimum.
    block_k = max(min(block_k, 1 << (k - 1).bit_length()), _MIN_BLOCK_K)
    while k_tail_in_unfilled_stage(block_k, k, num_stages):
        if block_k > _MIN_BLOCK_K:
            block_k //= 2
            continue
        # Even the minimum block leaves too few K blocks, e.g. K=320 in three blocks of 128.
        num_stages = -(-k // block_k) - 1
        if num_stages < 1:
            return None
    return block_k, num_stages


def _aiter_get_config(m: int, n: int, k_packed: int) -> tuple[dict, bool]:
    # The private config getter gemm_afp4wfp4 itself calls when given no config; it takes packed K.
    from aiter.ops.triton.gemm_afp4wfp4 import _get_config

    return _get_config(m, n, k_packed)


@functools.cache
def _warn_no_k_tail_safe_tiling(k: int) -> None:
    logger.warning(
        "MXFP4 gemm_afp4wfp4: K=%d has no BLOCK_SIZE_K >= %d and num_stages that keep its K tail out "
        "of an unfilled LDS stage; it keeps the masked K-tail path of triton-lang/triton#11256.",
        k,
        _MIN_BLOCK_K,
    )


@functools.lru_cache(maxsize=1024)
def k_tail_safe_config(m: int, n: int, k_packed: int) -> Optional[dict]:
    """aiter's config for this shape with a K-tail-safe tiling; None keeps aiter's own selection."""
    # Gated on the gfx950 the bug and its fix were measured on (MI355X); another gfx95x needs its own sweep.
    hip = current_arch().hip
    if hip is None or hip.gcn_arch != "gfx950":
        return None
    try:
        config, _ = _aiter_get_config(m, n, k_packed)
    except (KeyError, AssertionError):
        # aiter's get_gemm_config: no M bucket (KeyError) or no DEFAULT.json (AssertionError).
        return None
    if config["NUM_KSPLIT"] != 1:
        # Split-K configs are left to AITER's get_splitk; its uneven one-split fallback (M <= 8 defaults)
        # loads scales via registers, not direct-to-LDS, so triton#11256 does not hit it (Triton 3.7.0).
        return None
    k = 2 * k_packed
    tiling = k_tail_safe_tiling(
        block_k=config["BLOCK_SIZE_K"], k=k, num_stages=config["num_stages"]
    )
    if tiling is None:
        _warn_no_k_tail_safe_tiling(k)
        return None
    block_k, num_stages = tiling
    if block_k == config["BLOCK_SIZE_K"] and num_stages == config["num_stages"]:
        return None
    return {**config, "BLOCK_SIZE_K": block_k, "num_stages": num_stages}


def gemm_afp4wfp4_k_tail_safe(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """y = x @ w.T for MXFP4 x [M, K/2], w [N, K/2] and their e8m0 [*, K/32] scales; writes y in place."""
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    config = k_tail_safe_config(m=x.shape[0], n=w.shape[0], k_packed=x.shape[1])
    gemm_afp4wfp4(x, w, x_scales, w_scales, y.dtype, y, config=config)

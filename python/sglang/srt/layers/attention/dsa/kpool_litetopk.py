# SPDX-License-Identifier: Apache-2.0
"""Guarded SM100 LiteTopK for prefill with the KPool4 extend plan."""

import logging
from subprocess import CalledProcessError

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.attention.dsa.kpool_topk_transform import map_kpool4_groups
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer import (
    DEFAULT_KV_STAGES,
    DEFAULT_Q_STAGES,
    _jit_fp8_indexer_module,
    fp8_indexer_candidates,
    is_fp8_indexer_supported,
)
from sglang.kernels.ops.attention.dsa.litetopk_prefill.topk_candidates import (
    _jit_guarded_fp32_topk_candidates_module,
    topk_candidates,
)

logger = logging.getLogger(__name__)

_GROUP_TOPK = 512
_CAP = 4096
_MIN_ROWS = 4096
_MAX_ROWS = 32768
_MIN_KEYS = 16384
# The widest sampling mode covers 32 tiles of 256 samples at stride 257.
_MAX_KEYS = 32 * 256 * 257


@cache_once
def _jit_available() -> bool:
    try:
        _jit_fp8_indexer_module(32, DEFAULT_KV_STAGES, DEFAULT_Q_STAGES, True)
        _jit_guarded_fp32_topk_candidates_module(_GROUP_TOPK)
    except (OSError, RuntimeError, CalledProcessError) as exc:
        logger.warning("KPool LiteTopK JIT unavailable; using dense prefill: %s", exc)
        return False
    return True


def try_kpool4_lite_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    scale: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    pool_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor | None,
    page_rows: torch.Tensor | None,
    offsets: torch.Tensor | None,
    *,
    num_sms: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    rows = q.shape[0]
    if (
        not _MIN_ROWS <= rows <= _MAX_ROWS
        or not _MIN_KEYS <= k.shape[0] <= _MAX_KEYS
        or q.shape[1:] != (32, 128)
        or q.dtype != torch.float8_e4m3fn
        or k.shape[1:] != (128,)
        or k.dtype != q.dtype
        or scale.dtype != torch.float32
        or scale.shape != (k.shape[0],)
        or weights.dtype != torch.float32
        or weights.shape != q.shape[:2]
        or (page_table is None and offsets is None)
        or (page_table is not None and (page_rows is None or page_table.stride(1) != 1))
        or not is_fp8_indexer_supported()
        or torch.cuda.is_current_stream_capturing()
        or not _jit_available()
    ):
        return None

    slab = fp8_indexer_candidates(
        q.contiguous(),
        weights.contiguous(),
        k,
        scale,
        starts.contiguous(),
        ends.contiguous(),
        _GROUP_TOPK,
        cap=_CAP,
        num_ctas=num_sms,
        fp32_reduction=True,
    )
    groups = torch.empty((rows, _GROUP_TOPK), dtype=torch.int32, device=q.device)
    rejected = slab.overflow
    topk_candidates(
        slab.scores, slab.indices, slab.counts, slab.thresholds, groups, reject=rejected
    )
    del slab
    fallback_rows = rejected.nonzero().flatten()
    if fallback_rows.numel() == rows:
        return None
    mapped = map_kpool4_groups(
        groups, rejected, pool_lens, seq_lens, page_table, page_rows, offsets
    )
    return mapped, fallback_rows

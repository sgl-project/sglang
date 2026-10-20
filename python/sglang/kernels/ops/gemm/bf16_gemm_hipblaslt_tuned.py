"""BF16 dense GEMMs on fixed hipBLASLt solutions for exact (N, K, M) shapes.

``y = x @ W.T`` through ``aiter.hipb_mm`` with a searched solution index: BF16 in, FP32 accumulate, BF16 out,
the same arithmetic as the heuristic hipBLASLt pick behind ``nn.Linear``; only the tiling / K-split order
differs. Shapes not in the table return None and the caller keeps its own path.

``configs/muse_spark_dense_hipblaslt_mi355x.json``: the Muse Spark TP8 MoE pooler (4096 x 8192) and unpooler
(8192 x 4096) at verify widths (M > 64), searched on MI355X (HIP-graph timed). These shapes cannot go through
aiter's tgemm CSV: aiter's shipped model configs already tune both (N, K) pairs at most M with other kernels,
and its padded-M lookup would move every untuned M of these projections off ``nn.Linear``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs",
    "muse_spark_dense_hipblaslt_mi355x.json",
)
_extension_ready = False


@lru_cache(maxsize=1)
def _solutions() -> dict:
    with open(_CONFIG) as f:
        return {
            tuple(int(v) for v in key.split(",")): int(s)
            for key, s in json.load(f).items()
        }


def hipblaslt_tuned_solution(x: torch.Tensor, weight: torch.Tensor) -> int | None:
    """The table's hipBLASLt solution for ``x @ weight.T``, or None."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    sol = _solutions().get((weight.shape[0], weight.shape[1], x.shape[0]))
    if sol is None or not x.is_contiguous() or not weight.is_contiguous():
        return None
    return sol


def bf16_gemm_hipblaslt(
    x: torch.Tensor, weight: torch.Tensor, solution: int
) -> torch.Tensor:
    global _extension_ready
    import aiter

    if not _extension_ready:
        aiter.hipb_create_extension()
        _extension_ready = True
    return aiter.hipb_mm(
        x, weight.t(), solution, None, torch.bfloat16, None, None, None, False
    )

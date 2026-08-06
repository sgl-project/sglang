"""Fail-closed loader for ASTER's optimized K3 IQ2 routed-MoE kernel."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import torch


logger = logging.getLogger(__name__)
_MODULE_NAME = "aster_iq2_moe_warpbench"
_KERNEL_PATH = Path(os.environ["SGLANG_ASTER_IQ2_KERNEL"])
if not _KERNEL_PATH.is_file():
    raise RuntimeError(f"ASTER IQ2 kernel is missing: {_KERNEL_PATH}")

_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _KERNEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load ASTER IQ2 kernel: {_KERNEL_PATH}")
_EXTENSION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXTENSION)
for _symbol in (
    "iq2_moe_vec_sglang",
    "iq2_moe_vec_situ",
    "iq2_moe_vec_weighted_reduce",
):
    if not hasattr(_EXTENSION, _symbol):
        raise RuntimeError(f"ASTER IQ2 extension lacks {_symbol}")

_REACHED = False
_FUSED_REACHED = False


def aster_iq2_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    qtype: int,
    rows: int,
    tokens: int,
) -> torch.Tensor:
    global _REACHED
    if not _REACHED:
        logger.warning(
            "ASTER_IQ2_MOE_KERNEL_REACHED qtype=%d rows=%d tokens=%d",
            qtype,
            rows,
            tokens,
        )
        _REACHED = True
    return _EXTENSION.iq2_moe_vec_sglang(
        x, weight, topk_ids, top_k, qtype, rows, tokens
    )


def aster_iq2_moe_fused_decode(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_qtype: int,
    w2_qtype: int,
    situ_beta: float | None,
    situ_linear_beta: float | None,
) -> torch.Tensor | None:
    """Bit-exact K3 SiTU path for decode/verify widths up to eight."""
    tokens = x.shape[0] if x.ndim == 2 else 0
    if (
        x.ndim != 2
        or not 1 <= tokens <= 8
        or topk_ids.shape != (tokens, 16)
        or topk_weights.shape != (tokens, 16)
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
        or w13_qtype not in (16, 17)
        or w2_qtype not in (16, 17)
        or situ_beta != 4.0
        or situ_linear_beta != 25.0
        or w13.shape[1] != 1_536
        or w2.shape[1] != 3_584
    ):
        return None

    global _FUSED_REACHED
    if not _FUSED_REACHED:
        logger.warning(
            "ASTER_IQ2_MOE_FUSED_DECODE_REACHED w13_qtype=%d w2_qtype=%d "
            "tokens=%d",
            w13_qtype,
            w2_qtype,
            tokens,
        )
        _FUSED_REACHED = True

    activated = _EXTENSION.iq2_moe_vec_situ(
        x,
        w13,
        topk_ids,
        16,
        w13_qtype,
        1_536,
        tokens,
        4,
        1,
        4.0,
        25.0,
    )
    return _EXTENSION.iq2_moe_vec_weighted_reduce(
        activated,
        w2,
        topk_ids,
        topk_weights,
        16,
        w2_qtype,
        3_584,
        tokens,
        1,
        1,
        True,
    )

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
if not hasattr(_EXTENSION, "iq2_moe_vec_sglang"):
    raise RuntimeError("ASTER IQ2 extension lacks iq2_moe_vec_sglang")

_REACHED = False


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

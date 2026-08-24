"""Retention ring for the EAGLE/MTP + capture-hidden candidate family.

Row-overwrite case §24.10.4: the clobber payload is a per-token,
natural-order, replica-invariant float stream -- a freed-tensor family
that [mf-fp] could never probe because the tensors are released before
the send-path check runs. The family (dflash FULL hidden capture on
prefill/verify, draft topk_p/topk_index/hidden, embedding output) is
hooked at its production points with ``retain(name, tensor)``; this ring
holds detached references so the fingerprint probe can sequence-match
them at [mf-raw] time. A FOUND line then names the payload tensor.

SGLANG_MF_EAGLE_RETAIN=1 enables retention (byte-budgeted ring,
SGLANG_MF_EAGLE_RETAIN_MB, default 512). SGLANG_MF_RETAIN_NOP=1 keeps
the same call overhead but holds nothing -- the sham control, same
semantics as quant_retention. torch-only imports: safe to call from
model/spec code on any backend; retain() no-ops on one cached env read
when both flags are off.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_recent: list = []
_MAX_ENTRIES = 24
_DEFAULT_BUDGET = 512 * 1024 * 1024
_flags: dict = {}


def _budget() -> int:
    mb = os.getenv("SGLANG_MF_EAGLE_RETAIN_MB", "")
    try:
        return int(float(mb) * 1024 * 1024) if mb else _DEFAULT_BUDGET
    except ValueError:
        return _DEFAULT_BUDGET


def _nbytes(t) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


def _ensure_flags() -> None:
    if not _flags:
        _flags["on"] = (
            os.getenv("SGLANG_MF_EAGLE_RETAIN", "0") == "1"
            or os.getenv("SGLANG_MF_RETAIN_NOP", "0") == "1"
        )
        _flags["nop"] = os.getenv("SGLANG_MF_RETAIN_NOP", "0") == "1"
        if _flags["on"] and not _flags["nop"]:
            logger.info("[mf-er] eagle retention ON (budget %d MB)", _budget() // (1024 * 1024))


def enabled() -> bool:
    """True when retention is active; forces the one-time env read.

    Callers use this to gate tensor SLICING (views would pin the full
    backing storage alive), so the off-flag path stays zero-cost.
    """
    _ensure_flags()
    return bool(_flags.get("on"))


def retain(name: str, tensor) -> None:
    """Hold one detached reference (or nothing, under the NOP sham)."""
    _ensure_flags()
    if not _flags.get("on") or not torch.is_tensor(tensor):
        return
    if _flags.get("nop"):
        return  # Same per-call overhead, holds nothing (sham control).
    try:
        _recent.append((name, tensor.detach()))
        budget = _budget()
        while len(_recent) > _MAX_ENTRIES or (
            len(_recent) > 1 and sum(_nbytes(t) for _, t in _recent) > budget
        ):
            _recent.pop(0)
    except Exception:
        pass


def payloads():
    """-> list[(name, tensor)] for the [mf-fp] candidate registration."""
    return list(_recent)

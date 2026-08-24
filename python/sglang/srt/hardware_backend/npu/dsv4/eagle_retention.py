"""Retention ring for the EAGLE/MTP + capture-hidden candidate family.

Row-overwrite case §24.10.4: the clobber payload is a per-token,
natural-order, replica-invariant float stream -- a freed-tensor family
that [mf-fp] could never probe because the tensors are released before
the send-path check runs. The family (dflash FULL hidden capture on
prefill/verify, draft topk_p/topk_index/hidden, embedding output) is
hooked at its production points with ``retain(name, tensor)``; this ring
holds detached references so the fingerprint probe can sequence-match
them at [mf-raw] time. A FOUND line then names the payload tensor.

SGLANG_MF_EAGLE_RETAIN=1 enables retention (byte-budgeted,
SGLANG_MF_EAGLE_RETAIN_MB, default 512). SGLANG_MF_RETAIN_NOP=1 keeps
the same call overhead but holds nothing -- the sham control, same
semantics as quant_retention. torch-only imports: safe to call from
model/spec code on any backend; retain() no-ops on one cached env read
when both flags are off.

Ring discipline (post-§24.18 review): ONE newest entry PER NAME, not a
FIFO -- a forward produces ~11 names x ~172MB, so a FIFO under the byte
budget evicts the strongest candidates (draft.embed/ctx_hidden) before
the [mf-raw] probe can audit them, which would make a "silent" run
unauditable. Same-name entries replace the previous hold (the newest
forward is the likely victim); eviction only when a single tensor alone
exceeds the budget. Held tensors are cloned OUTSIDE inference mode:
detach() alone leaves an inference tensor, and probing it from grad
context raises -- an exception the probe would silently miscount as
"probed, no hit".
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Per-name ring: name -> (tensor). Insertion-ordered; newest per name wins.
_recent: dict = {}
_MAX_PER_NAME = 1
_DEFAULT_BUDGET = 512 * 1024 * 1024
_flags: dict = {}
_warned: dict = {}


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
            logger.info(
                "[mf-er] eagle retention ON (budget %d MB, per-name keep-newest)",
                _budget() // (1024 * 1024),
            )


def enabled() -> bool:
    """True when retention is active; forces the one-time env read.

    Callers use this to gate tensor SLICING (views would pin the full
    backing storage alive), so the off-flag path stays zero-cost.
    """
    _ensure_flags()
    return bool(_flags.get("on"))


def _hold_clone(t: torch.Tensor) -> torch.Tensor:
    """Detached clone that is safe to probe outside inference mode.

    Retained tensors are produced under torch.inference_mode(); a bare
    detach() keeps that marker and grad-context probes raise on it.
    """
    with torch.inference_mode(False):
        return t.detach().clone()


def retain(name: str, tensor) -> None:
    """Hold the newest detached clone per name (nothing under the NOP sham)."""
    _ensure_flags()
    if not _flags.get("on") or not torch.is_tensor(tensor):
        return
    if _flags.get("nop"):
        return  # Same per-call overhead, holds nothing (sham control).
    try:
        if _nbytes(tensor) > _budget():
            # A single tensor beyond the budget cannot be held; say so
            # instead of silently dropping it.
            if not _warned.get("big:" + name):
                _warned["big:" + name] = True
                logger.warning(
                    "[mf-er] %s (%.1f MB) exceeds budget; not retained",
                    name,
                    _nbytes(tensor) / (1024 * 1024),
                )
            return
        _recent[name] = _hold_clone(tensor)
        # Keep-newest per name already bounds the total; drop oldest names
        # only if the caller widened the name set past the budget.
        while sum(_nbytes(t) for t in _recent.values()) > _budget():
            oldest = next(iter(_recent))
            _recent.pop(oldest)
    except Exception:
        if not _warned.get("v"):
            _warned["v"] = True
            try:
                logger.warning("[mf-er] retain failed once", exc_info=True)
            except Exception:
                pass


def payloads():
    """-> list[(name, tensor)] for the [mf-fp] candidate registration."""
    return list(_recent.items())


def names() -> list:
    """-> [name] currently held; print at probe time so every exclusion
    is auditable against the full site list."""
    return list(_recent.keys())

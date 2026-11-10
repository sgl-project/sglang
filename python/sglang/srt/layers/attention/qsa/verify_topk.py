"""Target-verify top-512 for the QSA indexer, opt-in by ``SGLANG_QSA_VERIFY_TOPK=sampled``.

``stock`` (default) keeps ``fast_topk``: one 1024-thread CTA per row, three or more passes
over the row. At verify batches of 44-64 requests (176-256 rows) that is about one CTA per
SM, so each row's own latency sets the launch time. ``sampled`` runs the prefill select's
top-k (``qsa_prefill_topk.cuh``) on the same rows: ``fast_topk``'s exact output (the same
indices in the same slots) from one pass over the row; rows it does not take run
fast_topk's own select. Up to one row per SM it uses 1024-thread blocks; past that, the
512-thread variant (``qsa_prefill_topk_half.cuh``, four blocks per SM, the same output).
"""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_MODES = ("stock", "sampled")
MODE = os.environ.get("SGLANG_QSA_VERIFY_TOPK", "").strip().lower() or "stock"
if MODE not in _MODES:
    raise ValueError(f"SGLANG_QSA_VERIFY_TOPK must be one of {_MODES}, got {MODE!r}")
# Debug only: the first N eager calls also run fast_topk and log whether the output matches.
_CHECK_CALLS = int(os.environ.get("SGLANG_QSA_VERIFY_TOPK_CHECK_CALLS", "0") or 0)
_TOPK = 512
_state = {"logged": set(), "checked": 0}


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _impl_for(rows: int, device: torch.device) -> str:
    # VR200 (212 SMs), [rows, 65536] logits, 8K-65K keys per row: 1024 threads are faster up to
    # 192 rows (15.9 vs 19.1-19.8 us), 512 threads from 228 rows (21.5-22.3 vs 25.1-25.4 us).
    index = device.index if device.index is not None else torch.cuda.current_device()
    return "sampled" if rows <= _num_sms(index) else "half"


def _log_once(rows: int, impl: str) -> None:
    if rows in _state["logged"]:
        return
    _state["logged"].add(rows)
    logger.info("SGLANG_QSA_VERIFY_TOPK engaged mode=%s rows=%d impl=%s", MODE, rows, impl)


def _check(logits, lengths, indices) -> None:
    if _state["checked"] >= _CHECK_CALLS or torch.cuda.is_current_stream_capturing():
        return
    from sglang.kernels.ops.attention.fast_topk import fast_topk

    _state["checked"] += 1
    ref = fast_topk(logits, lengths, topk=_TOPK, row_starts=None)
    logger.info(
        "SGLANG_QSA_VERIFY_TOPK check %d/%d: rows=%d equal=%s",
        _state["checked"],
        _CHECK_CALLS,
        indices.shape[0],
        bool(torch.equal(indices, ref)),
    )


def select(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``fast_topk(logits, lengths, 512, row_starts=None)`` for decode rows (they start at
    column 0); ``lengths`` int32."""
    from sglang.srt.layers.attention.qsa.prefill_select import _topk_launcher

    rows = logits.shape[0]
    impl = _impl_for(rows, logits.device)
    _log_once(rows, impl)
    indices = logits.new_empty((rows, _TOPK), dtype=torch.int32)
    starts = torch.zeros(rows, dtype=torch.int32, device=logits.device)
    _topk_launcher(impl)(logits, starts, indices, lengths)
    if _CHECK_CALLS:
        _check(logits, lengths, indices)
    return indices

"""Backend-agnostic assertions on `forward_metadata` after CG replay init.

Catches corruption that the output-equality assertion misses — e.g., negative
per-request lengths or non-monotonic indptr that happen to leave real-row
output correct while corrupting padded-row scratch state.

Usage from a CG runner kit:

    from .metadata_invariants import assert_cg_metadata_well_formed
    _init_cuda_graph_replay_metadata(backend, capture_batch_size, replay_batch)
    assert_cg_metadata_well_formed(backend, bs=capture_batch_size)
"""

from __future__ import annotations

from typing import Any

import torch

# Field names that, when present on `forward_metadata`, denote a CSR-style
# `indptr` array. Convention: a length-(bs+1) int tensor whose first element is
# 0 and that is non-decreasing.
_INDPTR_FIELDS = (
    "kv_indptr",
    "qo_indptr",
    "mask_indptr",
    "window_kv_indptr",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "encoder_cu_seqlens_k",
)

# Field names that, when present, denote a length-bs per-request length tensor.
# Each element must be >= 0.
_PER_REQ_LEN_FIELDS = (
    "cache_seqlens_int32",
    "encoder_lens_int32",
    "local_seqused_k",
    "max_seq_len_k",  # may be scalar; checked defensively
)


def _slice_bs_plus_one(t: torch.Tensor, bs: int) -> torch.Tensor:
    # An indptr conventionally has length bs+1; some backends may pre-size to
    # max_bs+1 and slice on demand. Take the first bs+1 either way.
    return t[: bs + 1] if t.numel() >= bs + 1 else t


def _slice_bs(t: torch.Tensor, bs: int) -> torch.Tensor:
    return t[:bs] if t.numel() >= bs else t


def assert_cg_metadata_well_formed(backend: Any, bs: int) -> None:
    """Inspect ``backend.forward_metadata`` and flag obviously-corrupt buffers.

    Best-effort: a field is checked only when it exists on the metadata object
    and is a non-None tensor. Backends without a ``forward_metadata`` attribute
    (or with one set to None) are skipped silently — the assertion only fires
    on tensors that are clearly malformed.
    """
    meta = getattr(backend, "forward_metadata", None)
    if meta is None:
        return

    errors: list[str] = []

    for field in _INDPTR_FIELDS:
        t = getattr(meta, field, None)
        if not isinstance(t, torch.Tensor):
            continue
        sliced = _slice_bs_plus_one(t, bs)
        if sliced.numel() < 2:
            continue
        diff = sliced[1:].to(torch.int64) - sliced[:-1].to(torch.int64)
        # Allow zero-length requests (diff == 0); reject only negative diffs.
        if (diff < 0).any().item():
            min_diff = diff.min().item()
            errors.append(
                f"{field} is not monotonic non-decreasing at bs={bs} "
                f"(min adjacent diff={min_diff}); slice={sliced[: min(bs + 1, 16)].tolist()}"
            )
        if sliced[0].item() != 0:
            errors.append(f"{field}[0] != 0 (got {sliced[0].item()}) at bs={bs}")

    for field in _PER_REQ_LEN_FIELDS:
        t = getattr(meta, field, None)
        if not isinstance(t, torch.Tensor):
            continue
        sliced = _slice_bs(t, bs)
        if sliced.numel() == 0:
            continue
        if (sliced < 0).any().item():
            min_v = sliced.min().item()
            errors.append(
                f"{field} has negative values at bs={bs} "
                f"(min={min_v}); slice={sliced[: min(bs, 16)].tolist()}"
            )

    if errors:
        raise AssertionError(
            "CG forward_metadata invariants violated after replay init:\n  - "
            + "\n  - ".join(errors)
        )

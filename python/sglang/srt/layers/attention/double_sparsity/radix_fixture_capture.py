"""Module-level capture log for the AC-10 M3-B radix-cache fixture.

The DS write hook (``DSABackend._write_token_labels`` in
``dsa_backend.py``) calls :func:`record_write` after every label
write. When the env var ``SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`` is set,
``record_write`` appends a per-write record carrying SHA256 fingerprints
of the slot indices and the projected K-noPE bytes. The capture-aware
manual fixture
(``test/manual/test_dsv32_radix_label_capture_fixture.py``) reads the
log between paired cold/warm requests and asserts the per-layer
fingerprints for the shared-prefix slots are bit-equal.

Why fingerprints, not full tensors:

* The fixture needs to fit inside an HTTP response or in-process
  capture buffer without dumping multi-MB tensors.
* SHA256 is a strict bit-equality test: identical inputs collide
  with cryptographic certainty; different inputs do not.
* Hash storage is constant-size per write regardless of slot count
  or label dim, so the capture overhead stays bounded.

Default-off semantics: when ``SGLANG_DS_RADIX_FIXTURE_CAPTURE`` is
unset (production), ``record_write`` and ``record_table_snapshot``
are zero-cost early-exits.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import torch

_ENV_FLAG = "SGLANG_DS_RADIX_FIXTURE_CAPTURE"

# Module-level capture state. A lock guards `_LOG` because the write
# hook fires from the model forward path, which may execute under
# CUDA streams on some backends; the producer-side cost is one
# uncontended `acquire`/`release` when capture is enabled, and a single
# early-exit branch otherwise.
_LOG: List[Dict[str, Any]] = []
_LOCK = threading.Lock()


def is_capture_enabled() -> bool:
    """Return True when ``SGLANG_DS_RADIX_FIXTURE_CAPTURE=1``."""
    return os.environ.get(_ENV_FLAG, "0") == "1"


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _tensor_bytes_sha(t: torch.Tensor) -> str:
    """SHA256 of a tensor's raw bytes. Tensor is moved to CPU + made
    contiguous so the hash is layout-stable across capture sites.
    """
    return _sha256_bytes(
        t.detach().to(torch.device("cpu")).contiguous().numpy().tobytes()
    )


def record_write(
    *,
    layer_id: int,
    cache_loc: torch.Tensor,
    k_nope: torch.Tensor,
    written_after: Optional[torch.Tensor] = None,
) -> None:
    """Append a per-write record to the capture log.

    No-op when capture is disabled — the production hot path pays
    exactly one ``os.environ.get`` lookup.

    Parameters
    ----------
    layer_id:
        DS layer index (model-local).
    cache_loc:
        ``[num_tokens]`` slot indices that the DS hook just wrote.
    k_nope:
        ``[num_tokens, num_heads_local, nope_dim]`` projected K-noPE
        (the input the labeling kernel consumed).
    written_after:
        Optional ``[num_tokens]`` slice of the table's ``written``
        flag taken AFTER the write. The fixture uses this to assert
        every shared-prefix slot is reachable after both cold and
        warm passes.
    """
    if not is_capture_enabled():
        return
    record: Dict[str, Any] = {
        "kind": "write",
        "layer_id": int(layer_id),
        "num_tokens": int(cache_loc.shape[0]) if cache_loc.dim() > 0 else 0,
        # Hash slot indices as int64 bytes so two int32 / int64 calls
        # with the same numeric values collide.
        "cache_loc_sha": _tensor_bytes_sha(cache_loc.long()),
        # Hash K-noPE as fp32 bytes for dtype-stability.
        "k_nope_sha": _tensor_bytes_sha(k_nope.float()),
    }
    if written_after is not None:
        record["written_after_sha"] = _tensor_bytes_sha(
            written_after.to(torch.bool)
        )
        record["written_after_all_true"] = bool(
            written_after.to(torch.bool).all().item()
        )
    with _LOCK:
        _LOG.append(record)


def record_table_snapshot(
    *,
    signatures: torch.Tensor,
    written: torch.Tensor,
    slots: torch.Tensor,
    label: str = "snapshot",
) -> None:
    """Append a post-forward snapshot of the table's label rows for
    ``slots``, broken down per local layer.

    The fixture calls this once after each paired request completes so
    the warm-vs-cold comparison can be by-slot, by-layer, with no
    sensitivity to ordering of intervening writes.
    """
    if not is_capture_enabled():
        return
    if signatures.dim() != 4 or written.dim() != 2:
        raise ValueError(
            "record_table_snapshot: expected signatures[L,T,H,D] + "
            f"written[L,T]; got {tuple(signatures.shape)} + "
            f"{tuple(written.shape)}."
        )
    slots_long = slots.long().to(signatures.device)
    L = signatures.shape[0]
    per_layer_label_sha: List[str] = []
    per_layer_written_sha: List[str] = []
    for layer_id in range(L):
        per_layer_label_sha.append(
            _tensor_bytes_sha(signatures[layer_id, slots_long])
        )
        per_layer_written_sha.append(
            _tensor_bytes_sha(written[layer_id, slots_long])
        )
    record = {
        "kind": "snapshot",
        "label": label,
        "num_layers": L,
        "num_slots": int(slots_long.shape[0]),
        "slots_sha": _tensor_bytes_sha(slots_long),
        "per_layer_label_sha": per_layer_label_sha,
        "per_layer_written_sha": per_layer_written_sha,
    }
    with _LOCK:
        _LOG.append(record)


def get_log() -> List[Dict[str, Any]]:
    """Return a shallow copy of the capture log."""
    with _LOCK:
        return list(_LOG)


def clear_log() -> None:
    """Reset the capture log. Called by the fixture between paired
    requests and at test setup."""
    with _LOCK:
        _LOG.clear()

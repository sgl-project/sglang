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


def _capturing_cuda_graph() -> bool:
    """True while a CUDA graph is being captured on the current stream.

    The capture records SHA fingerprints by copying tensors to CPU
    (``_tensor_bytes_sha``), which is illegal during CUDA-graph capture
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture").
    The fixture reads the capture log from EAGER paired requests, so recording
    during graph capture is both meaningless and unsafe — skip it. This matters
    because the production label-write hook now fires on the decode path (which
    is graph-captured); with capture enabled, an unguarded record would abort
    capture at model-runner init.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _sha256_bytes(buf: bytes) -> str:
    return hashlib.sha256(buf).hexdigest()


def _tensor_bytes_sha(t: torch.Tensor) -> str:
    """SHA256 of a tensor's raw bytes. Tensor is moved to CPU + made
    contiguous so the hash is layout-stable across capture sites.

    Hash through a flat uint8 byte view rather than ``.numpy()`` on the native
    dtype: the production token-label signatures are fp16 (and K-noPE may be
    bf16/fp8), and ``torch.Tensor.numpy()`` raises on dtypes NumPy cannot bridge
    (observed as ``UntypedStorage has no attribute 'dtype'`` / unsupported
    ScalarType). The uint8 reinterpret is byte-exact and dtype-agnostic, so cold
    and warm captures still compare identically.
    """
    cpu = t.detach().to(torch.device("cpu")).contiguous().reshape(-1)
    return _sha256_bytes(cpu.view(torch.uint8).numpy().tobytes())


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
    if not is_capture_enabled() or _capturing_cuda_graph():
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
    if not is_capture_enabled() or _capturing_cuda_graph():
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


def build_request_capture(
    *,
    signatures: torch.Tensor,
    written: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Build per-request post-forward capture records.

    For each request ``b`` in the batch, compute
    ``slots = req_to_token[req_pool_indices[b], :seq_lens[b]]`` and
    return per-layer SHA256 hashes of ``signatures[L, slots]`` and
    ``written[L, slots]``. The cold-pass and warm-pass records over
    the SAME prompt should produce identical ``slots_sha`` (radix
    cache reused the same physical slots) and identical
    ``per_layer_label_sha`` (the labels for those slots are bit-stable).

    Pure function: no IO, no globals, no env reads — safe to call
    from the production per-request finalization site. The env-gated
    branch lives at the call site so the production hot path stays
    zero-cost when capture is off.

    Parameters
    ----------
    signatures: ``[num_layers_local, max_tokens, num_heads_local, label_dim]``
        The DS token label table's ``signatures`` buffer.
    written: ``[num_layers_local, max_tokens]`` bool
        The DS token label table's ``written`` buffer.
    req_to_token: ``[max_requests, max_seq_len]`` int
        The request-pool's physical-slot map for each request row.
    req_pool_indices: ``[batch_size]`` int
        Per-batch request-pool row index.
    seq_lens: ``[batch_size]`` int
        Per-batch sequence length (the prompt + already-generated
        token count).

    Returns
    -------
    list of length ``batch_size``. Each record carries:
        ``prompt_len`` (int), ``slots_sha`` (str),
        ``per_layer_label_sha`` (list[str]),
        ``per_layer_written_sha`` (list[str]),
        ``per_layer_written_all_true`` (list[bool]).
    """
    bs = int(req_pool_indices.shape[0])
    if bs == 0:
        return []
    if signatures.dim() != 4 or written.dim() != 2:
        raise ValueError(
            "build_request_capture: expected signatures[L,T,H,D] + "
            f"written[L,T]; got {tuple(signatures.shape)} + "
            f"{tuple(written.shape)}."
        )
    if req_to_token.dim() != 2:
        raise ValueError(
            "build_request_capture: expected req_to_token[R,S]; got "
            f"{tuple(req_to_token.shape)}."
        )

    L = signatures.shape[0]
    rpi = req_pool_indices.long().to(req_to_token.device)
    sl = seq_lens.long().to(req_to_token.device)

    records: List[Dict[str, Any]] = []
    for b in range(bs):
        prompt_len = int(sl[b].item())
        slots = req_to_token[int(rpi[b].item()), :prompt_len]
        slots_long = slots.long().to(signatures.device)
        if prompt_len == 0 or slots_long.numel() == 0:
            records.append({
                "prompt_len": prompt_len,
                "slots_sha": _tensor_bytes_sha(slots_long),
                "per_token_slot_sha": [],
                "per_layer_label_sha": ["<empty>"] * L,
                "per_layer_written_sha": ["<empty>"] * L,
                "per_layer_written_all_true": [True] * L,
                "per_layer_per_token_label_sha": [[] for _ in range(L)],
            })
            continue
        per_layer_label_sha: List[str] = []
        per_layer_written_sha: List[str] = []
        per_layer_written_all_true: List[bool] = []
        per_layer_per_token_label_sha: List[List[str]] = []
        for layer_id in range(L):
            per_layer_label_sha.append(
                _tensor_bytes_sha(signatures[layer_id, slots_long])
            )
            written_slice = written[layer_id, slots_long].to(torch.bool)
            per_layer_written_sha.append(_tensor_bytes_sha(written_slice))
            per_layer_written_all_true.append(
                bool(written_slice.all().item())
            )
            # Per-token label SHA so the fixture can compare just the
            # first `cached_tokens` positions (the radix-cache-reused
            # prefix) without false-mismatching on later decode-allocated
            # slots that one side has and the other does not.
            layer_slice = signatures[layer_id, slots_long]
            per_layer_per_token_label_sha.append([
                _tensor_bytes_sha(layer_slice[t])
                for t in range(layer_slice.shape[0])
            ])
        per_token_slot_sha = [
            _tensor_bytes_sha(slots_long[t : t + 1])
            for t in range(slots_long.shape[0])
        ]
        records.append({
            "prompt_len": prompt_len,
            "slots_sha": _tensor_bytes_sha(slots_long),
            "per_token_slot_sha": per_token_slot_sha,
            "per_layer_label_sha": per_layer_label_sha,
            "per_layer_written_sha": per_layer_written_sha,
            "per_layer_written_all_true": per_layer_written_all_true,
            "per_layer_per_token_label_sha": per_layer_per_token_label_sha,
        })
    return records


def compare_cached_prefix(
    *,
    cold: Dict[str, Any],
    warm: Dict[str, Any],
    cached_tokens: int,
) -> Dict[str, Any]:
    """Compare just the first ``cached_tokens`` positions of two
    per-request capture records.

    The radix cache reuses the leading prefix slots when the warm
    request shares a prefix with a previously-served cold request.
    Beyond ``cached_tokens`` positions both passes may have allocated
    distinct slots (suffix tokens for the warm pass; the full prompt
    for cold). Comparing only the cached prefix avoids a false
    mismatch when one side's capture includes extra decode-allocated
    slots and the other does not.

    Returns ``{ok, first_diverging_position, divergence_kind}``.
    ``divergence_kind`` is one of:
      * ``"slot"`` — the physical slot at that position differs
        (radix cache did not reuse the slot);
      * ``"label"`` — the slot agrees but at least one layer's
        per-token label SHA differs at that position;
      * ``""`` — no divergence in the cached-prefix range.
    """
    if cached_tokens <= 0:
        return {
            "ok": False,
            "first_diverging_position": -1,
            "divergence_kind": "no_cached_prefix",
            "reason": "cached_tokens <= 0; nothing to compare",
        }
    c_slots = cold.get("per_token_slot_sha") or []
    w_slots = warm.get("per_token_slot_sha") or []
    n = min(cached_tokens, len(c_slots), len(w_slots))
    if n == 0:
        return {
            "ok": False,
            "first_diverging_position": -1,
            "divergence_kind": "empty_capture",
            "reason": (
                f"capture too short for cached_tokens={cached_tokens}: "
                f"cold={len(c_slots)} warm={len(w_slots)}"
            ),
        }
    for t in range(n):
        if c_slots[t] != w_slots[t]:
            return {
                "ok": False,
                "first_diverging_position": t,
                "divergence_kind": "slot",
                "reason": (
                    f"per_token_slot_sha differs at position {t}: "
                    f"cold={c_slots[t]!r} warm={w_slots[t]!r}"
                ),
            }
    c_pll = cold.get("per_layer_per_token_label_sha") or []
    w_pll = warm.get("per_layer_per_token_label_sha") or []
    if len(c_pll) != len(w_pll):
        return {
            "ok": False,
            "first_diverging_position": -1,
            "divergence_kind": "layer_count",
            "reason": (
                f"per_layer_per_token_label_sha layer count mismatch: "
                f"cold={len(c_pll)} warm={len(w_pll)}"
            ),
        }
    for layer_id in range(len(c_pll)):
        c_layer = c_pll[layer_id]
        w_layer = w_pll[layer_id]
        m = min(n, len(c_layer), len(w_layer))
        for t in range(m):
            if c_layer[t] != w_layer[t]:
                return {
                    "ok": False,
                    "first_diverging_position": t,
                    "divergence_kind": "label",
                    "reason": (
                        f"per_layer_per_token_label_sha differs at "
                        f"layer={layer_id} position={t}: "
                        f"cold={c_layer[t]!r} warm={w_layer[t]!r}"
                    ),
                }
    return {
        "ok": True,
        "first_diverging_position": -1,
        "divergence_kind": "",
        "reason": "",
    }

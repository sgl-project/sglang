"""Flatten ragged (nested, variable-length) structures into flat value buffers
plus per-position length vectors — the columnar wire layout used by the embedded
Rust server's egress path (see ``managers/rust_server.py``).
"""

from __future__ import annotations

from typing import List


def flatten_ragged(per_pos_val, per_pos_idx):
    """Flatten a per-position ``list[Optional[list]]`` (top-k / token-ids
    logprobs) into flat ``val``/``idx`` buffers plus a per-position ``lens``
    vector for the columnar egress wire. A falsy (None/empty) position
    contributes no values and a ``0`` length — the Rust side reshapes it back to
    a ``null`` position, matching ``detokenize_top_logprobs_tokens``.
    """
    flat_val: List[float] = []
    flat_idx: List[int] = []
    lens: List[int] = []
    if not per_pos_val:
        return flat_val, flat_idx, lens
    per_pos_idx = per_pos_idx or []
    for p, pv in enumerate(per_pos_val):
        if pv:
            pi = per_pos_idx[p] if p < len(per_pos_idx) else []
            # A truthy position holds only real logprobs; a `None`/empty position
            # is the falsy branch below (len 0), so no per-value None check here.
            flat_val.extend(pv)
            flat_idx.extend(pi or [])
            lens.append(len(pv))
        else:
            lens.append(0)
    return flat_val, flat_idx, lens


def flatten_hidden(hs):
    """Flatten one request's hidden states into a flat ``val`` buffer plus a
    per-row ``lens`` vector (one row per output position). Each top-level element
    becomes a single row; the Rust side reshapes back to ``list[list[float]]``,
    matching ``meta_info["hidden_states"]``'s common per-position-vector shape.
    """
    vals: List[float] = []
    lens: List[int] = []
    if not hs:
        return vals, lens
    for row in hs:
        flat = _flatten_floats(row)
        vals.extend(flat)
        lens.append(len(flat))
    return vals, lens


def _flatten_floats(x):
    """Recursively flatten a (possibly nested) float structure into a flat list
    of floats — handles the ``float | list[float]`` union inside a hidden-state
    chunk."""
    if isinstance(x, (int, float)):
        return [float(x)]
    out: List[float] = []
    for e in x:
        out.extend(_flatten_floats(e))
    return out

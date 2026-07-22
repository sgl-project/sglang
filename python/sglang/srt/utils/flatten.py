"""Flatten ragged (nested, variable-length) structures into flat value buffers
plus per-position length vectors — the columnar wire layout used by the embedded
Rust server's egress path (see ``managers/rust_server.py``).

The ``*Columns`` classes accumulate one batch column-family each: feed them one
request cell at a time (``accept``), then read the header contribution (length
vectors) and data contribution (raw ``array`` buffers).
"""

from __future__ import annotations

from array import array
from typing import List


def flatten_ragged(per_pos_val, per_pos_idx):
    """Flatten per-position ``list[Optional[list]]`` val/idx pairs into flat
    buffers + a shared ``lens`` vector (falsy position -> len 0 -> Rust ``null``).
    idx must mirror val exactly (asserted): the wire pairs both buffers by the
    one ``lens`` vector, so divergence shifts or drops token ids downstream."""

    flat_val: List[float] = []
    flat_idx: List[int] = []
    lens: List[int] = []
    if not per_pos_val:
        assert not per_pos_idx, (
            f"ragged idx column has {len(per_pos_idx)} positions but the val "
            "column is empty"
        )
        return flat_val, flat_idx, lens
    assert per_pos_idx is not None and len(per_pos_idx) == len(per_pos_val), (
        f"ragged idx column has {len(per_pos_idx) if per_pos_idx else 0} "
        f"positions, val column has {len(per_pos_val)}"
    )
    for p, pv in enumerate(per_pos_val):
        pi = per_pos_idx[p]
        if pv:
            # A truthy position holds only real logprobs; a `None`/empty position
            # is the falsy branch below (len 0), so no per-value None check here.
            assert pi is not None and len(pi) == len(pv), (
                f"position {p}: idx len "
                f"{len(pi) if pi is not None else None} != val len {len(pv)}"
            )
            flat_val.extend(pv)
            flat_idx.extend(pi)
            lens.append(len(pv))
        else:
            assert not pi, f"position {p}: idx has {len(pi)} entries but val is empty"
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


class FlatPairColumns:
    """A flat val/idx column pair (e.g. per-token logprob values + token ids):
    per-request element counts in the header, concatenated f32 + i32 buffers in
    the data. ``first_none_to_nan`` maps a leading ``None`` cell element to NaN
    (the input-logprob first-prompt-token sentinel)."""

    def __init__(self, name, vals, idxs, first_none_to_nan=False):
        self.name = name
        self.vals = vals
        self.idxs = idxs
        self.first_none_to_nan = first_none_to_nan
        self.v = array("f")
        self.i = array("i")
        self.lens = []

    def columns(self):
        return ((f"{self.name}_val", self.vals), (f"{self.name}_idx", self.idxs))

    def accept(self, j):
        vv = (self.vals[j] if self.vals else None) or []
        if self.first_none_to_nan and vv and vv[0] is None:
            self.v.append(float("nan"))
            self.v.extend(vv[1:])
        else:
            self.v.extend(vv)
        self.i.extend((self.idxs[j] if self.idxs else None) or [])
        self.lens.append(len(vv))

    def header_cols(self):
        return [self.lens]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class RaggedPairColumns:
    """A per-position ragged val/idx column pair (e.g. top-k / token-ids
    logprobs): per-request position counts + a flat per-position length stream
    in the header, concatenated f32/i32 buffers in the data."""

    def __init__(self, name, vals, idxs):
        self.name = name
        self.vals = vals
        self.idxs = idxs
        self.v = array("f")
        self.i = array("i")
        self.pos = []
        self.req = []

    def columns(self):
        return ((f"{self.name}_val", self.vals), (f"{self.name}_idx", self.idxs))

    def accept(self, j):
        fv, fi, lens = flatten_ragged(
            self.vals[j] if self.vals else None,
            self.idxs[j] if self.idxs else None,
        )
        self.v.extend(fv)
        self.i.extend(fi)
        self.pos.extend(lens)
        self.req.append(len(lens))

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class NestedRowColumns:
    """A nested-rows float column (e.g. hidden states): per-request row counts +
    per-row length stream in the header, one concatenated f32 buffer in the
    data."""

    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.v = array("f")
        self.pos = []
        self.req = []

    def columns(self):
        return ((self.name, self.rows),)

    def accept(self, j):
        hv, hlens = flatten_hidden(self.rows[j] if self.rows else None)
        self.v.extend(hv)
        self.pos.extend(hlens)
        self.req.append(len(hlens))

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes()]

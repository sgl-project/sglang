"""Flatten ragged (nested, variable-length) structures into flat value buffers
plus per-position length vectors — the columnar wire layout used by the embedded
Rust server's egress path (see ``rust_server/server.py``).

The ``*Columns`` classes flatten one batch column-family each: ``fill`` the
batch, then read the header contribution (length vectors) and data contribution
(raw ``array`` buffers). A family no request carries data for stays empty, the
wire's "absent".
"""

from __future__ import annotations

from array import array
from itertools import chain, compress
from typing import List

# How one row of a request's hidden states nests in ``meta_info["hidden_states"]``.
# ``HIDDEN_ROW_*`` in the Rust side's ``message/response.rs`` holds the same values.
HIDDEN_ROW_WHOLE = 0  # the value itself is this one vector (``"last"``)
HIDDEN_ROW_VECTOR = 1  # an item that is one vector (a decode step)
HIDDEN_ROW_MATRIX_FIRST = 2  # first row of an item that is a matrix (a prefill step)
HIDDEN_ROW_MATRIX_NEXT = 3  # a later row of that matrix


def flatten_hidden(hs):
    """Flatten one request's hidden states into a flat ``val`` buffer, a
    per-row ``lens`` vector and a per-row ``kinds`` vector, from which the Rust
    side rebuilds the value with the nesting it has here.

    ``return_hidden_states=True`` gives a list of items: a matrix (one row per
    token) for a prefill step, one vector for a decode step. A matrix sends
    each of its rows. ``return_hidden_states="last"`` gives one flat vector,
    sent as a single row.
    """
    vals: List[float] = []
    lens: List[int] = []
    kinds: List[int] = []
    if not hs:
        return vals, lens, kinds
    if isinstance(hs[0], (int, float)):
        vals.extend(float(x) for x in hs)
        return vals, [len(vals)], [HIDDEN_ROW_WHOLE]
    for item in hs:
        if item and isinstance(item[0], (list, tuple)):
            kind = HIDDEN_ROW_MATRIX_FIRST
            for row in item:
                flat = _flatten_floats(row)
                vals.extend(flat)
                lens.append(len(flat))
                kinds.append(kind)
                kind = HIDDEN_ROW_MATRIX_NEXT
        else:
            flat = _flatten_floats(item)
            vals.extend(flat)
            lens.append(len(flat))
            kinds.append(HIDDEN_ROW_VECTOR)
    return vals, lens, kinds


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


_NAN = float("nan")


def _rows_with_data(cells, batch_size):
    """The rows whose cell is non-empty. A falsy cell carries nothing: length 0,
    no elements."""
    if not any(cells):  # the common all-empty family, without a row index each
        return []
    return list(compress(range(batch_size), cells))


def _scatter(rows, values, batch_size):
    """``values[k]`` at position ``rows[k]`` of a ``batch_size`` list, 0 elsewhere."""
    if len(rows) == batch_size:
        return values
    out = [0] * batch_size
    for row, value in zip(rows, values):
        out[row] = value
    return out


def _select(cells, rows, batch_size):
    """The cells at ``rows`` (all of them when every row is selected)."""
    return cells if len(rows) == batch_size else [cells[row] for row in rows]


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

    def fill(self, batch_size):
        # Only the rows with data are visited; a family with none stays empty.
        rows = _rows_with_data(self.vals, batch_size)
        if not rows:
            return
        vals = _select(self.vals, rows, batch_size)
        idxs = _select(self.idxs, rows, batch_size) if self.idxs else []
        lens = list(map(len, vals))
        # Parity assert: only the val lengths are recorded in `lens`, but both
        # buffers are filled, so a longer idx cell silently pushes every LATER
        # column's offset out by the difference. The decoder cannot catch it —
        # the data buffer only grows, so the receiver's bounds check still
        # passes and it hands the client another column's bytes reinterpreted
        # as logprobs, with a 200.
        assert list(map(len, idxs)) == lens, (
            f"{self.name}: idx lengths {list(map(len, idxs))} != val lengths {lens}"
        )
        # `fromlist` sizes the buffer once; `extend` grows it per element.
        if len(rows) == 1:
            # One row, e.g. a prefill's input logprobs over a long prompt:
            # skip the flattening copy.
            vv = vals[0]
            if self.first_none_to_nan and vv[0] is None:
                self.v.append(_NAN)
                vv = vv[1:]
            self.v.fromlist(vv)
            self.i.fromlist(idxs[0])
        else:
            flat = list(chain.from_iterable(vals))
            if self.first_none_to_nan:
                start = 0
                for n in lens:
                    if flat[start] is None:
                        flat[start] = _NAN
                    start += n
            self.v.fromlist(flat)
            self.i.fromlist(list(chain.from_iterable(idxs)))
        self.lens = _scatter(rows, lens, batch_size)

    def header_cols(self):
        return [self.lens]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class RaggedPairColumns:
    """A per-position ragged val/idx column pair (e.g. top-k / token-ids
    logprobs): per-request position counts + a flat per-position length stream
    in the header, concatenated f32/i32 buffers in the data. A falsy position
    (``None`` / empty) is a null position: length 0, no elements."""

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

    def fill(self, batch_size):
        # Only the rows with data are visited; a family with none stays empty.
        rows = _rows_with_data(self.vals, batch_size)
        if not rows:
            return
        vals = _select(self.vals, rows, batch_size)
        idxs = _select(self.idxs, rows, batch_size) if self.idxs else []
        req = list(map(len, vals))
        # idx must mirror val exactly: the wire pairs both buffers by the one
        # per-position length stream, so divergence shifts or drops token ids
        # downstream.
        assert list(map(len, idxs)) == req, (
            f"{self.name}: idx positions {list(map(len, idxs))} != val positions {req}"
        )
        pos_v = list(chain.from_iterable(vals))
        pos_i = list(chain.from_iterable(idxs))
        pos = [len(p) if p else 0 for p in pos_v]
        assert [len(p) if p else 0 for p in pos_i] == pos, (
            f"{self.name}: idx lengths differ from val lengths by position"
        )
        self.v.fromlist(list(chain.from_iterable(filter(None, pos_v))))
        self.i.fromlist(list(chain.from_iterable(filter(None, pos_i))))
        self.pos = pos
        self.req = _scatter(rows, req, batch_size)

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes(), self.i.tobytes()]


class NestedRowColumns:
    """A nested-rows float column (e.g. hidden states): per-request row counts,
    a per-row length stream and a per-row kind stream in the header, one
    concatenated f32 buffer in the data."""

    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.v = array("f")
        self.pos = []
        self.kind = []
        self.req = []

    def columns(self):
        return ((self.name, self.rows),)

    def fill(self, batch_size):
        # Only the rows with data are visited; a family with none stays empty.
        rows = _rows_with_data(self.rows, batch_size)
        if not rows:
            return
        req = []
        for row in rows:
            hv, hlens, hkinds = flatten_hidden(self.rows[row])
            self.v.fromlist(hv)
            self.pos.extend(hlens)
            self.kind.extend(hkinds)
            req.append(len(hlens))
        self.req = _scatter(rows, req, batch_size)

    def header_cols(self):
        return [self.req, self.pos, self.kind]

    def data_cols(self):
        return [self.v.tobytes()]

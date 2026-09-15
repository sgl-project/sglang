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
        ii = (self.idxs[j] if self.idxs else None) or []
        # Parity assert, the flat twin of `flatten_ragged`'s: only `len(vv)` is
        # recorded in `lens`, but both buffers are extended, so a longer idx column
        # silently pushes every LATER column's offset out by the difference. The
        # decoder cannot catch it — the data buffer only grows, so the receiver's
        # bounds check still passes and it hands the client another column's bytes
        # reinterpreted as logprobs, with a 200.
        assert len(ii) == len(vv), (
            f"{self.name}: request {j} has {len(ii)} idx entries but {len(vv)} vals"
        )
        if self.first_none_to_nan and vv and vv[0] is None:
            self.v.append(float("nan"))
            self.v.extend(vv[1:])
        else:
            self.v.extend(vv)
        self.i.extend(ii)
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
        self.shapes = []

    def columns(self):
        return ((self.name, self.rows),)

    def accept(self, j):
        value = self.rows[j] if self.rows else None
        hv, hlens = flatten_hidden(value)
        self.v.extend(hv)
        self.pos.extend(hlens)
        self.req.append(len(hlens))
        self.shapes.append(_hidden_shape(value) if value is not None else None)

    def header_cols(self):
        return [self.req, self.pos]

    def data_cols(self):
        return [self.v.tobytes()]


def _hidden_shape(value):
    """Describe nested vectors without moving their floats into the header."""
    if not value or isinstance(value[0], (int, float)):
        return len(value)
    return [_hidden_shape(child) for child in value]


class SamplingMaskColumns:
    """Sparse per-token supports and selected-token logprobs, with explicit
    absent-request, null-token, and empty-support shapes."""

    def __init__(self, masks, logprobs):
        assert len(masks) == len(logprobs), "sampling mask/logprob batch sizes differ"
        self.shapes = []
        self.ids = array("i")
        self.logprobs = array("f")
        for request_masks, request_logprobs in zip(masks, logprobs):
            if request_masks is None:
                assert request_logprobs is None
                self.shapes.append(None)
                continue
            assert request_logprobs is not None and len(request_masks) == len(
                request_logprobs
            ), "sampling mask/logprob token counts differ"
            self.shapes.append(
                [None if mask is None else len(mask) for mask in request_masks]
            )
            for mask, logprob in zip(request_masks, request_logprobs):
                if mask is not None:
                    self.ids.extend(mask)
                self.logprobs.append(float("nan") if logprob is None else logprob)

    def data_cols(self):
        return [self.ids.tobytes(), self.logprobs.tobytes()]


class FlatTopLogprobColumns:
    """Keep scheduler-produced rectangular prompt arrays in their raw dtypes."""

    def __init__(self, vals, idxs, null_prefixes):
        assert len(vals) == len(idxs) == len(null_prefixes)
        self.shapes = []
        self.vals = []
        self.idxs = []
        for val, idx, null_prefix in zip(vals, idxs, null_prefixes):
            if val is None:
                assert idx is None and null_prefix is None
                self.shapes.append(None)
                continue
            assert val.ndim == 2 and val.shape == idx.shape
            assert val.dtype.str == "<f4" and idx.dtype.str == "<i4"
            assert null_prefix is not None and null_prefix >= 0
            self.shapes.append([*val.shape, null_prefix])
            self.vals.append(val.tobytes())
            self.idxs.append(idx.tobytes())

    def data_cols(self):
        return self.vals + self.idxs


class TensorBytesColumn:
    """Preserve CPU tensor bytes, including empty tensors and absent requests."""

    def __init__(self, tensors):
        self.lengths = []
        self.buffers = []
        for tensor in tensors:
            if tensor is None:
                self.lengths.append(None)
                continue
            data = tensor.numpy().tobytes()
            self.lengths.append(len(data))
            self.buffers.append(data)

    def data_cols(self):
        return self.buffers


class BeamSearchColumns:
    """Ranked beam headers with generated token IDs in one raw i32 column."""

    def __init__(self, outputs):
        self.headers = []
        self.tokens = array("i")
        for output in outputs:
            if output is None:
                self.headers.append(None)
                continue
            headers = []
            for sequence in output.sequences:
                headers.append(
                    (len(sequence.tokens), sequence.finish_reason, sequence.beam_score)
                )
                self.tokens.extend(sequence.tokens)
            self.headers.append(headers)

    def data_cols(self):
        return [self.tokens.tobytes()]

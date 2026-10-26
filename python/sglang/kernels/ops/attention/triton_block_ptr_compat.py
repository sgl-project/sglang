"""Restore ``tl.make_block_ptr`` / ``tl.advance`` on Triton builds that removed them.

The FLA kernels use block pointers, which triton-lang/triton#10833 (Triton 3.9)
removed. On such builds this rebuilds them
on plain pointer loads and stores, the same way Triton 3.8 lowered them. Older
Triton is left untouched.
"""

import triton.language as tl
from triton.language import core

try:
    from triton.language import advance  # noqa: F401

    _HAS_BLOCK_PTR = True
except ImportError:  # removed in triton-lang/triton#10833
    _HAS_BLOCK_PTR = False

# TODO: Drop this once the kernels are ported off block pointers, as upstream FLA
# did in fla-org/flash-linear-attention#1062.
if not _HAS_BLOCK_PTR:

    @core._aggregate
    class _BlockPtr:
        base: core.tensor
        shape: core.tuple
        strides: core.tuple
        offsets: core.tuple
        block_shape: core.tuple

    _BlockPtr.dtype = property(lambda self: self.base.dtype)

    def _int64(values, sem):
        return core.tuple([sem.cast(sem.to_tensor(v), tl.int64) for v in values])

    def _materialize(p, boundary_check, sem):
        tile = [b.value for b in p.block_shape]
        checked = [core._unwrap_if_constexpr(d) for d in boundary_check]
        ptrs, mask = p.base, None
        for dim, extent in enumerate(tile):
            coord = sem.add(p.offsets[dim], sem.arange(0, extent), True)
            for _ in range(dim):
                coord = sem.expand_dims(coord, 0)
            for axis in range(dim + 1, len(tile)):
                coord = sem.expand_dims(coord, axis)
            coord = sem.broadcast_impl_shape(coord, tile)
            ptrs = sem.add(ptrs, sem.mul(coord, p.strides[dim], True), True)
            if dim in checked:
                lo, hi = sem.greater_equal(coord, 0), sem.less_than(coord, p.shape[dim])
                valid = sem.and_(lo, hi)
                mask = valid if mask is None else sem.and_(mask, valid)
        return ptrs, mask

    @core.builtin
    def make_block_ptr(
        base, shape, strides, offsets, block_shape, order, _semantic=None
    ):
        shape, strides, offsets = (
            _int64(v, _semantic) for v in (shape, strides, offsets)
        )
        return _BlockPtr(base, shape, strides, offsets, core.tuple(block_shape))

    @core.builtin
    def advance(base, offsets, _semantic=None):
        deltas = zip(base.offsets, _int64(offsets, _semantic))
        offsets = core.tuple([_semantic.add(o, d, True) for o, d in deltas])
        return _BlockPtr(base.base, base.shape, base.strides, offsets, base.block_shape)

    _load, _store = tl.load, tl.store

    @core.builtin
    def load(
        pointer,
        mask=None,
        other=None,
        boundary_check=(),
        padding_option="",
        _semantic=None,
        **kwargs,
    ):
        if isinstance(pointer, _BlockPtr):
            pointer, mask = _materialize(pointer, boundary_check, _semantic)
            padding = core._unwrap_if_constexpr(padding_option)
            other = {"zero": 0.0, "nan": float("nan")}.get(padding)
        return _load(pointer, mask, other, _semantic=_semantic, **kwargs)

    @core.builtin
    def store(pointer, value, mask=None, boundary_check=(), _semantic=None, **kwargs):
        if isinstance(pointer, _BlockPtr):
            pointer, mask = _materialize(pointer, boundary_check, _semantic)
        return _store(pointer, value, mask, _semantic=_semantic, **kwargs)

    tl.make_block_ptr, tl.advance = make_block_ptr, advance
    tl.load, tl.store = load, store

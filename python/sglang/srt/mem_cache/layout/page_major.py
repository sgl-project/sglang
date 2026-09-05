"""Page-granularity envelope (page-major, token-major within a page) cache views.

A pool of this layout keeps all layers of all slots in one contiguous byte
buffer, split into pages of ``page_size`` slots. Within a page the slots follow
one another, and one slot's ENTRY holds every part of that token -- K and V of
every layer for MHA, the latent row of every layer for MLA -- at a fixed byte
offset:

    page bytes  = [ entry(slot 0) | entry(slot 1) | ... | entry(slot ps-1) ]
    entry bytes = [ K_0 | V_0 | K_1 | V_1 | ... ]  (MHA)
                  [ lat_0 | lat_1 | ... ]           (MLA)

Every per-layer view is therefore a flat ``(num_pages * page_size, *row_shape)``
tensor with slot stride ``entry_bytes`` and storage offset
``anchor + part offset``, indexed by the PHYSICAL token id
``page * page_size + slot``. Parts may differ in row width (K vs V); only their
offsets differ, never the stride.

These builders produce views into a raw ``uint8`` buffer; they hold no
allocator/ownership state. ``anchor_bytes`` is the byte offset of the pool's
region inside the raw buffer (0 for a standalone pool).
"""

from typing import List, Sequence, Tuple

import msgspec
import torch

ENTRY_ALIGN_BYTES = 32
ROW_ALIGN_BYTES = 16


def _prod(shape: Sequence[int]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= int(s)
    return tuple(reversed(strides))


def align_entry_bytes(num_bytes: int) -> int:
    """Round a slot's byte sum up to the entry alignment."""
    return -(-num_bytes // ENTRY_ALIGN_BYTES) * ENTRY_ALIGN_BYTES


def paged_view(flat: torch.Tensor, page_size: int) -> torch.Tensor:
    """``[slots, ...]`` -> ``[slots // page_size, page_size, ...]``, as a view.

    Splitting dim 0 never copies: the slot stride becomes dim 1's stride, so a
    strided per-layer view keeps addressing ``slot * stride(0)``.
    """
    num_slots = int(flat.shape[0])
    assert num_slots % page_size == 0, (num_slots, page_size)
    out = flat.view(num_slots // page_size, page_size, *flat.shape[1:])
    assert out.stride(1) == flat.stride(0)
    return out


def paged_row_view(flat: torch.Tensor, page_size: int) -> torch.Tensor:
    """``[slots, ...]`` -> ``[slots // page_size, page_size, row_elems]``, as a view."""
    return paged_view(flat.view(int(flat.shape[0]), -1), page_size)


class DensePart(msgspec.Struct, frozen=True, kw_only=True):
    """One row family inside the entry: ``layer_num`` rows of ``row_shape``,
    layer ``l`` at ``offset_bytes + l * layer_stride_bytes``."""

    name: str
    offset_bytes: int
    layer_stride_bytes: int
    layer_num: int
    row_shape: Tuple[int, ...]
    dtype: torch.dtype

    def row_bytes(self) -> int:
        return _prod(self.row_shape) * self.dtype.itemsize

    def layer_offset_bytes(self, layer: int) -> int:
        return self.offset_bytes + layer * self.layer_stride_bytes


class DenseEntryLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Byte layout of one slot's entry; ``entry_bytes`` is the slot stride of
    every view built over it."""

    entry_bytes: int
    parts: Tuple[DensePart, ...]

    def __post_init__(self):
        self.validate()

    def part(self, name: str) -> DensePart:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(f"no part {name!r} in {[p.name for p in self.parts]}")

    def validate(self) -> None:
        assert self.entry_bytes % ENTRY_ALIGN_BYTES == 0, (
            f"entry_bytes={self.entry_bytes} is not a multiple of {ENTRY_ALIGN_BYTES}"
        )
        spans = []
        for p in self.parts:
            row = p.row_bytes()
            assert (
                p.offset_bytes % ROW_ALIGN_BYTES == 0
                and p.layer_stride_bytes % ROW_ALIGN_BYTES == 0
                and row % ROW_ALIGN_BYTES == 0
            ), (
                f"part {p.name!r}: offset {p.offset_bytes}, layer stride "
                f"{p.layer_stride_bytes} and row {row} B must all be multiples of "
                f"{ROW_ALIGN_BYTES} (vector stores)"
            )
            for l in range(p.layer_num):
                lo = p.layer_offset_bytes(l)
                assert 0 <= lo and lo + row <= self.entry_bytes, (
                    f"part {p.name!r} layer {l} spans [{lo}, {lo + row}) outside "
                    f"the {self.entry_bytes}-byte entry"
                )
                spans.append((lo, lo + row, p.name, l))
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            assert a[1] <= b[0], (
                f"parts overlap: {a[2]}[{a[3]}] [{a[0]}, {a[1]}) and "
                f"{b[2]}[{b[3]}] [{b[0]}, {b[1]})"
            )


def build_dense_views(
    raw: torch.Tensor,
    *,
    layout: DenseEntryLayout,
    part: DensePart,
    page_size: int,
    num_pages: int,
    anchor_bytes: int = 0,
) -> List[torch.Tensor]:
    """Per-layer views of one part: ``(num_pages * page_size, *row_shape)`` with
    slot stride ``layout.entry_bytes``, indexed by the physical token id
    ``page * page_size + slot`` (``paged_view`` regroups them by page).
    """
    itemsize = part.dtype.itemsize
    assert layout.entry_bytes % itemsize == 0 and anchor_bytes % itemsize == 0
    n_rows = num_pages * page_size
    end = anchor_bytes + n_rows * layout.entry_bytes
    assert end <= raw.numel() * raw.itemsize, (
        f"build_dense_views: {n_rows} slots of {layout.entry_bytes} B end at byte "
        f"{end} but the raw buffer holds only {raw.numel() * raw.itemsize} bytes"
    )
    as_dtype_view = raw.view(part.dtype)
    stride = (layout.entry_bytes // itemsize, *_contiguous_strides(part.row_shape))
    views: List[torch.Tensor] = []
    for layer in range(part.layer_num):
        base_bytes = anchor_bytes + part.layer_offset_bytes(layer)
        assert base_bytes % itemsize == 0
        views.append(
            torch.as_strided(
                as_dtype_view,
                size=(n_rows, *part.row_shape),
                stride=stride,
                storage_offset=base_bytes // itemsize,
            )
        )
    return views


def mha_entry_bytes(
    *, layer_num: int, head_num: int, head_dim: int, v_head_dim: int, itemsize: int
) -> int:
    """Bytes occupied by one slot across all layers (K and V), aligned."""
    k_row_bytes = head_num * head_dim * itemsize
    v_row_bytes = head_num * v_head_dim * itemsize
    return align_entry_bytes(layer_num * (k_row_bytes + v_row_bytes))


def mla_entry_bytes(*, layer_num: int, kv_cache_dim: int, itemsize: int) -> int:
    """Bytes occupied by one MLA slot across all layers (latent rows), aligned."""
    return align_entry_bytes(layer_num * kv_cache_dim * itemsize)


def mamba_entry_bytes(
    *,
    layer_num: int,
    conv_state_shapes: Sequence[Sequence[int]],
    conv_dtype: torch.dtype,
    temporal_state_shape: Sequence[int],
    temporal_dtype: torch.dtype,
) -> int:
    """Bytes occupied by one Mamba slot across all layers (conv + temporal)."""
    total = 0
    for shape in conv_state_shapes:
        total += layer_num * _prod(shape) * conv_dtype.itemsize
    total += layer_num * _prod(temporal_state_shape) * temporal_dtype.itemsize
    return total


def build_page_major_mamba_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    conv_state_shapes: Sequence[Sequence[int]],
    conv_dtype: torch.dtype,
    temporal_state_shape: Sequence[int],
    temporal_dtype: torch.dtype,
    max_slots: int,
    anchor_bytes: int = 0,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Per-slot envelope views over ``raw`` for Mamba state.

    Layout per slot: ``[conv[0] rows × layers][conv[1] rows × layers]...
    [temporal rows × layers]``. Each returned view has shape
    ``(num_layers, max_slots, *inner_shape)`` matching ``MambaPool.State.conv[i]``
    / ``.temporal``. Mamba state is always token-granular (page_size == 1).
    """
    entry_bytes = mamba_entry_bytes(
        layer_num=layer_num,
        conv_state_shapes=conv_state_shapes,
        conv_dtype=conv_dtype,
        temporal_state_shape=temporal_state_shape,
        temporal_dtype=temporal_dtype,
    )

    def contiguous_strides(shape: Sequence[int]) -> Tuple[int, ...]:
        strides = []
        acc = 1
        for s in reversed(shape):
            strides.append(acc)
            acc *= int(s)
        return tuple(reversed(strides))

    conv_itemsize = conv_dtype.itemsize
    assert entry_bytes % conv_itemsize == 0, (
        f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
        f"multiple of the conv-state itemsize {conv_itemsize} B"
    )
    assert anchor_bytes % conv_itemsize == 0, (
        f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
        f"of the conv-state itemsize {conv_itemsize} B"
    )
    as_conv_dtype = raw.view(conv_dtype)
    conv_slot_stride_elems = entry_bytes // conv_itemsize

    offset_bytes_within_entry = 0
    conv_views: List[torch.Tensor] = []
    for shape in conv_state_shapes:
        inner_shape_bytes = _prod(shape) * conv_itemsize
        assert inner_shape_bytes % conv_itemsize == 0
        offset_elems = (anchor_bytes + offset_bytes_within_entry) // conv_itemsize
        stride = (
            inner_shape_bytes // conv_itemsize,
            conv_slot_stride_elems,
        ) + contiguous_strides(shape)
        conv_views.append(
            torch.as_strided(
                as_conv_dtype,
                size=(layer_num, max_slots) + tuple(shape),
                stride=stride,
                storage_offset=offset_elems,
            )
        )
        offset_bytes_within_entry += layer_num * inner_shape_bytes

    # The temporal view's storage_offset is computed in temporal-dtype elements
    # by integer-dividing a byte offset by itemsize, so every term of that byte
    # offset (entry stride, anchor, the conv region) must be a whole multiple of
    # itemsize or the offset truncates and mis-places the view.
    itemsize = temporal_dtype.itemsize
    assert entry_bytes % itemsize == 0, (
        f"misaligned mamba spec: per-slot entry_bytes={entry_bytes} is not a "
        f"multiple of the temporal-state itemsize {itemsize} B; the temporal "
        f"view's storage_offset would truncate and mis-place the state"
    )
    assert anchor_bytes % itemsize == 0, (
        f"misaligned mamba spec: anchor_bytes={anchor_bytes} is not a multiple "
        f"of the temporal-state itemsize {itemsize} B"
    )
    inner_shape_bytes = _prod(temporal_state_shape) * itemsize
    assert inner_shape_bytes % itemsize == 0, (
        f"misaligned mamba spec: temporal inner_shape_bytes={inner_shape_bytes} "
        f"is not a multiple of the temporal-state itemsize {itemsize} B"
    )
    assert (anchor_bytes + offset_bytes_within_entry) % itemsize == 0, (
        f"misaligned mamba spec: temporal region byte offset "
        f"{anchor_bytes + offset_bytes_within_entry} is not a multiple of the "
        f"temporal-state itemsize {itemsize} B"
    )
    offset_elems = (anchor_bytes + offset_bytes_within_entry) // itemsize
    as_temporal_dtype = raw.view(temporal_dtype)
    stride = (
        inner_shape_bytes // itemsize,
        entry_bytes // itemsize,
    ) + contiguous_strides(temporal_state_shape)
    temporal_view = torch.as_strided(
        as_temporal_dtype,
        size=(layer_num, max_slots) + tuple(temporal_state_shape),
        stride=stride,
        storage_offset=offset_elems,
    )
    return conv_views, temporal_view

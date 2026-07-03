"""Page-granularity envelope (page-major, layer-major within a page) cache views.

A pool of this layout keeps all layers of all slots in one contiguous byte
buffer. The buffer is split into pages of ``page_size`` slots; within a page,
each layer's K and V (or each Mamba conv/temporal tensor) are grouped together:

    page bytes = [L0_K * ps | L0_V * ps | L1_K * ps | L1_V * ps | ...]

Across pages the layout is envelope-major (one ``page_bytes`` block per page).
At ``page_size == 1`` a page is a single slot, so the within-page block is the
per-slot ``[L0_K | L0_V | L1_K | L1_V | ...]`` envelope (token-granularity).

These builders produce per-layer strided views into a raw ``uint8`` buffer; they
hold no allocator/ownership state. ``anchor_bytes`` is the byte offset of the
pool's region inside the raw buffer (0 for a standalone pool).
"""

from typing import List, Sequence, Tuple

import torch


def _prod(shape: Sequence[int]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def mha_entry_bytes(
    *, layer_num: int, head_num: int, head_dim: int, v_head_dim: int, itemsize: int
) -> int:
    """Bytes occupied by one slot across all layers (K and V)."""
    k_row_bytes = head_num * head_dim * itemsize
    v_row_bytes = head_num * v_head_dim * itemsize
    return layer_num * (k_row_bytes + v_row_bytes)


def build_page_major_mha_views(
    raw: torch.Tensor,
    *,
    layer_num: int,
    head_num: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype: torch.dtype,
    page_size: int,
    num_pages: int,
    anchor_bytes: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-layer K/V views over ``raw`` in the page-major layer-major layout.

    Each returned view is 4-D ``(num_pages, page_size, head_num, head_dim*)``
    with constant strides:

        stride[0] = page_bytes / itemsize     # next page
        stride[1] = k_row_bytes / itemsize    # next slot within layer L's K block
        stride[2] = head_dim                  # next head
        stride[3] = 1                         # next element

    V is analogous with ``v_row_bytes`` / ``v_head_dim``. A token id ``t`` reads
    page ``t // page_size``, slot ``t % page_size``.
    """
    itemsize = store_dtype.itemsize
    k_row_bytes = head_num * head_dim * itemsize
    v_row_bytes = head_num * v_head_dim * itemsize
    entry_bytes = layer_num * (k_row_bytes + v_row_bytes)
    page_bytes = page_size * entry_bytes
    assert anchor_bytes % itemsize == 0
    assert k_row_bytes % itemsize == 0
    assert v_row_bytes % itemsize == 0
    assert page_bytes % itemsize == 0

    as_dtype_view = raw.view(store_dtype)
    stride_page = page_bytes // itemsize
    stride_tok_k = k_row_bytes // itemsize
    stride_tok_v = v_row_bytes // itemsize

    k_shape = (num_pages, page_size, head_num, head_dim)
    v_shape = (num_pages, page_size, head_num, v_head_dim)
    k_stride = (stride_page, stride_tok_k, head_dim, 1)
    v_stride = (stride_page, stride_tok_v, v_head_dim, 1)

    k_buffer: List[torch.Tensor] = []
    v_buffer: List[torch.Tensor] = []
    for layer in range(layer_num):
        # Layer L's K block starts at L * page_size * (k_row + v_row); V follows.
        k_base_bytes = anchor_bytes + layer * page_size * (k_row_bytes + v_row_bytes)
        v_base_bytes = k_base_bytes + page_size * k_row_bytes
        assert k_base_bytes % itemsize == 0
        assert v_base_bytes % itemsize == 0
        k_buffer.append(
            torch.as_strided(
                as_dtype_view,
                size=k_shape,
                stride=k_stride,
                storage_offset=k_base_bytes // itemsize,
            )
        )
        v_buffer.append(
            torch.as_strided(
                as_dtype_view,
                size=v_shape,
                stride=v_stride,
                storage_offset=v_base_bytes // itemsize,
            )
        )
    return k_buffer, v_buffer


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

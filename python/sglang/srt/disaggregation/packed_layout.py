"""Transport-independent metadata proof for contiguous NHD KV page packing."""

PACKED_LAYOUT_VERSION = 2


def describe_packable_pool(pool, region=None):
    """Prove registerable storage is uniform contiguous NHD K/V byte rows."""
    if getattr(pool, "kv_cache_layout", None) != "nhd":
        return None
    if getattr(pool, "dtype", None) is None:
        return None
    if (
        getattr(pool, "is_quantized_kv_cache", False)
        or getattr(pool, "k_scale_buffer", None) is not None
        or getattr(pool, "v_scale_buffer", None) is not None
    ):
        return None
    k_buffers = getattr(pool, "k_buffer", None)
    v_buffers = getattr(pool, "v_buffer", None)
    if k_buffers is None or v_buffers is None:
        return None
    buffers = list(k_buffers) + list(v_buffers)
    if not buffers or len(k_buffers) != len(v_buffers):
        return None
    first = buffers[0]
    if first.ndim != 3 or first.shape[1] <= 0 or first.shape[2] <= 0:
        return None
    geometry = tuple(first.shape[1:])
    if any(
        tensor.ndim != 3
        or tuple(tensor.shape[1:]) != geometry
        or not tensor.is_contiguous()
        or tensor.dtype != first.dtype
        for tensor in buffers
    ):
        return None
    pointers, lengths, items = pool.get_contiguous_buf_infos()
    if region is not None:
        if (
            list(region.ptrs) != list(pointers)
            or list(region.lens) != list(lengths)
            or list(region.item_lens) != list(items)
        ):
            return None
        layer_ids = getattr(pool, "model_layer_ids", None)
        # Plain static MHA uses positional registration without wire layer IDs.
        # When IDs are advertised, they must still prove the actual pool order.
        if (
            region.layer_ids
            and layer_ids is not None
            and list(region.layer_ids) != list(layer_ids) * 2
        ):
            return None
    page_size = int(pool.page_size)
    page_bytes = page_size * geometry[0] * geometry[1] * first.element_size()
    if (
        page_size <= 0
        or pointers != [tensor.data_ptr() for tensor in buffers]
        or len(items) != len(buffers)
        or len(lengths) != len(buffers)
        or any(item != page_bytes for item in items)
        or any(length <= 0 or length % page_bytes for length in lengths)
    ):
        return None
    return {
        "page_size": page_size,
        "heads": geometry[0],
        "head_dim": geometry[1],
        "element_bytes": first.element_size(),
        "storage_dtype": str(first.dtype),
        "logical_dtype": str(pool.dtype),
        "page_bytes": page_bytes,
        "page_capacities": [length // page_bytes for length in lengths],
    }


def describe_packable_storage(storage):
    """Prove the current storage API's actual owner/state registration order."""
    entries = list(storage.wire_entries())
    # Only one optional ordinary SWA component; no draft, scale or state sidecars.
    if len(entries) > 1 or any(
        getattr(entry.state_type, "value", entry.state_type) != "swa"
        for entry in entries
    ):
        return None
    owner_view = storage.view(storage.owner)
    # Static hybrid storage exposes its wrapper as the owner view, while the
    # owner registration contains only the full-attention child's buffers.
    # The region comparison below still proves the exact registered byte order.
    if entries:
        owner_view = getattr(owner_view, "full_kv_pool", owner_view)
    full = describe_packable_pool(owner_view, storage.regions(storage.owner))
    swa = (
        describe_packable_pool(
            storage.view(entries[0].sub_pool), storage.regions(entries[0].sub_pool)
        )
        if entries
        else None
    )
    if full is None or (entries and swa is None):
        return None
    if swa is not None and (
        full["heads"] != swa["heads"] or full["page_size"] != swa["page_size"]
    ):
        return None
    return {"version": PACKED_LAYOUT_VERSION, "full": full, "swa": swa}

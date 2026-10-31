"""Transport-independent metadata proof for contiguous NHD KV page packing."""

import json
import operator
from dataclasses import dataclass

PACKED_LAYOUT_VERSION = 2


def decode_optional_pack_layout(frame):
    """A missing or malformed optional peer extension preserves the old path."""
    try:
        value = json.loads(frame) if frame else None
    except (ValueError, TypeError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def prove_packed_transfer(
    local,
    remote,
    component,
    *,
    src_tp,
    dst_tp,
    src_rank,
    dst_rank,
    total_heads,
    src_ptrs,
    dst_ptrs,
    src_item_lens,
    dst_item_lens,
    src_layer_ids,
    dst_layer_ids,
    src_pages,
    dst_pages,
):
    """Validate negotiated geometry before any gather or transport allocation."""
    try:
        layout = negotiate_packed_slice(
            local,
            remote,
            component,
            src_tp=src_tp,
            dst_tp=dst_tp,
            src_rank=src_rank,
            dst_rank=dst_rank,
            total_heads=total_heads,
        )
        if layout is None:
            return None
        source, destination = local[component], remote[component]
        count = len(src_ptrs)
        caps = [source["page_capacities"], destination["page_capacities"]]
        if (
            not count
            or count != len(dst_ptrs)
            or len(src_item_lens) != count
            or len(dst_item_lens) != count
            or list(src_layer_ids) != list(dst_layer_ids)
            or any(
                len(values) != count
                or any(type(n) is not int or n <= 0 for n in values)
                for values in caps
            )
            or any(n != layout.src_page_bytes for n in src_item_lens)
            or any(n != layout.dst_page_bytes for n in dst_item_lens)
            or any(operator.index(p) <= 0 for p in list(src_ptrs) + list(dst_ptrs))
        ):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if len(src_pages) != len(dst_pages):
        raise ValueError("Packed transfer needs paired pages")
    for pages, capacities in zip((src_pages, dst_pages), caps):
        kind = getattr(getattr(pages, "dtype", None), "kind", "i")
        if kind != "i":
            raise ValueError("Packed page indices must be signed integers")
        for page in pages:
            index = operator.index(page)
            if index < 0 or index >= min(capacities):
                raise ValueError("Packed page outside registered span")
    return layout


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


@dataclass(frozen=True)
class PackedSlice:
    page_size: int
    src_page_bytes: int
    dst_page_bytes: int
    head_offset_bytes: int

    def validate(self) -> None:
        if min(self.page_size, self.src_page_bytes, self.dst_page_bytes) <= 0:
            raise ValueError("Packing requires positive page sizes")
        if self.src_page_bytes % self.page_size or self.dst_page_bytes % self.page_size:
            raise ValueError("KV page byte lengths must contain whole token rows")
        src_row = self.src_page_bytes // self.page_size
        dst_row = self.dst_page_bytes // self.page_size
        if self.head_offset_bytes < 0 or self.head_offset_bytes + dst_row > src_row:
            raise ValueError("Packed head slice is outside the source token row")


def negotiate_packed_slice(
    local, remote, component, *, src_tp, dst_tp, src_rank, dst_rank, total_heads
):
    """Legacy, replicated-head, dtype/layout and unsupported TP cases fall back."""
    if not isinstance(local, dict) or not isinstance(remote, dict):
        return None
    if (
        local.get("version") != PACKED_LAYOUT_VERSION
        or remote.get("version") != PACKED_LAYOUT_VERSION
    ):
        return None
    if not (0 < src_tp < dst_tp <= total_heads) or dst_tp % src_tp:
        return None
    if total_heads % src_tp or total_heads % dst_tp:
        return None
    if not (0 <= src_rank < src_tp and 0 <= dst_rank < dst_tp):
        return None
    if dst_rank // (dst_tp // src_tp) != src_rank:
        return None
    source, destination = local.get(component), remote.get(component)
    if not isinstance(source, dict) or not isinstance(destination, dict):
        return None
    if (
        source.get("heads") != total_heads // src_tp
        or destination.get("heads") != total_heads // dst_tp
    ):
        return None
    keys = ("page_size", "head_dim", "element_bytes", "storage_dtype", "logical_dtype")
    if any(
        source.get(key) is None or source.get(key) != destination.get(key)
        for key in keys
    ):
        return None
    try:
        page_size = int(source["page_size"])
        head_bytes = int(source["head_dim"]) * int(source["element_bytes"])
        if head_bytes <= 0:
            return None
        if source["page_bytes"] != page_size * source["heads"] * head_bytes:
            return None
        if destination["page_bytes"] != page_size * destination["heads"] * head_bytes:
            return None
        head_start = (dst_rank % (dst_tp // src_tp)) * destination["heads"]
        layout = PackedSlice(
            page_size,
            source["page_bytes"],
            destination["page_bytes"],
            head_start * head_bytes,
        )
        layout.validate()
    except (KeyError, TypeError, ValueError):
        return None
    return layout

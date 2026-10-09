"""Build transfer-layer mappings and pack host target/draft buffer references.

Model layer IDs are translated to device-local transfer IDs. Packed draft pools
contribute one owned layer each, appended after the target's transfer layers.
This module allocates no host memory and copies no tensor data. Host assembly
packs compatible descriptors. Linker uses only the mapping and views each
source independently so heterogeneous draft formats retain their byte widths.
"""

from __future__ import annotations

from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, PagedLayerBufferInfo


def build_transfer_layer_mapping(
    *,
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    target_model_layer_ids: tuple[int, ...],
) -> dict[int, int]:
    """Map device-local callbacks to target buffers followed by draft buffers.

    target_model_layer_ids lists all main KV layers in device-buffer order.
    A sidecar can own only a subset of those model layers. Result keys are
    device-local transfer IDs and values index the target-then-draft buffers.
    Each draft descriptor must own exactly one layer, not share an owner.
    Multiple single-layer draft pools are supported. A multi-layer draft pool
    belongs on the separate-draft path and cannot use this packed mapping.
    """
    model_to_device_layer = {
        layer: position for position, layer in enumerate(target_model_layer_ids)
    }
    if len(model_to_device_layer) != len(target_model_layer_ids):
        raise ValueError(
            f"{target.pool_name}: duplicate target_model_layer_ids={target_model_layer_ids}"
        )
    missing_layers = tuple(
        layer for layer in target.layer_ids if layer not in model_to_device_layer
    )
    if missing_layers:
        raise ValueError(
            f"{target.pool_name}: model layers {missing_layers} are outside "
            f"target_model_layer_ids={target_model_layer_ids}"
        )
    if target.shared_layer_to_owner:
        raise ValueError(
            f"{target.pool_name}: shared_layer_to_owner is not supported, "
            f"got {target.shared_layer_to_owner}"
        )
    target_buffers: PagedLayerBufferInfo = target.buffer_info
    target_buffers.validate(layer_ids=target.layer_ids)
    layer_mapping = {
        model_to_device_layer[layer]: position
        for position, layer in enumerate(target.layer_ids)
    }
    offset = len(target_buffers.buffers)
    for depth, draft in enumerate(drafts):
        context = f"{target.pool_name}: draft[{depth}]"
        if draft.pool_name != target.pool_name:
            raise ValueError(
                f"{context}: expected pool_name={target.pool_name}, got {draft.pool_name}"
            )
        if draft.indices_from_pool != target.indices_from_pool:
            raise ValueError(
                f"{context}: expected indices_from_pool={target.indices_from_pool}, "
                f"got {draft.indices_from_pool}"
            )
        if type(draft.buffer_info) is not type(target.buffer_info):
            raise ValueError(
                f"{context}: expected buffer_info type={type(target.buffer_info).__name__}, "
                f"got {type(draft.buffer_info).__name__}"
            )
        if len(draft.layer_ids) != 1:
            raise ValueError(
                f"{context}: packed transfer requires one owned layer per draft pool, "
                f"got layer_ids={draft.layer_ids}"
            )
        if draft.shared_layer_to_owner:
            raise ValueError(
                f"{context}: shared_layer_to_owner is not supported, "
                f"got {draft.shared_layer_to_owner}"
            )
        draft.buffer_info.validate(layer_ids=draft.layer_ids)
        if draft.buffer_info.page_size != target.buffer_info.page_size:
            raise ValueError(
                f"{context}: packed draft page coverage differs, "
                f"expected page_size={target.buffer_info.page_size}, "
                f"got {draft.buffer_info.page_size}"
            )
        layer_mapping[len(target_model_layer_ids) + depth] = offset
        offset += len(draft.buffer_info.buffers)
    return layer_mapping


def pack_host_pool_buffers(
    *,
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    target_model_layer_ids: tuple[int, ...],
) -> tuple[PagedLayerBufferInfo, dict[int, int]]:
    """Combine one pool's target and packed drafts for host assembly.

    Return the same-format buffer descriptor and transfer-layer-to-buffer map.
    Tensor references are appended in draft order without allocating or copying
    tensor data. Empty drafts preserves the target-only mapping.
    """
    layer_mapping = build_transfer_layer_mapping(
        target=target, drafts=drafts, target_model_layer_ids=target_model_layer_ids
    )
    target_buffers: PagedLayerBufferInfo = target.buffer_info
    buffers = target_buffers.packed_with(tuple(draft.buffer_info for draft in drafts))
    return buffers, layer_mapping

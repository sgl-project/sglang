from __future__ import annotations

from sglang.srt.mem_cache.device_pool_info import DeviceBufferInfo, DevicePoolInfo


def can_use_dsa_buffer_infos(pool, drafts, *, dcp_enabled: bool) -> bool:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.utils import is_cuda

    return (
        is_cuda()
        and not dcp_enabled
        and all(
            type(item) is DSATokenToKVPool and not item.layer_shard_enabled
            for item in (pool, *drafts)
        )
    )


def bind_pool_layers(
    *,
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    target_model_layer_ids: tuple[int, ...],
) -> dict[int, int]:
    """Map device-local callbacks to target buffers followed by draft buffers.

    target_model_layer_ids lists all main KV layers in device-buffer order.
    A sidecar can own only a subset of those model layers.
    """
    model_to_device_layer = {
        layer: position for position, layer in enumerate(target_model_layer_ids)
    }
    if len(model_to_device_layer) != len(target_model_layer_ids) or any(
        layer not in model_to_device_layer for layer in target.layer_ids
    ):
        raise ValueError(
            f"{target.pool_name}: model layers are outside the target stage"
        )
    if target.shared_layer_to_owner:
        raise ValueError("this transfer binding does not support shared owner layers")
    target.buffer_info.validate(layer_ids=target.layer_ids)
    layer_mapping = {
        model_to_device_layer[layer]: position
        for position, layer in enumerate(target.layer_ids)
    }
    offset = len(target.buffer_info.buffers)
    for depth, draft in enumerate(drafts):
        if (
            draft.pool_name != target.pool_name
            or draft.indices_from_pool != target.indices_from_pool
            or type(draft.buffer_info) is not type(target.buffer_info)
            or len(draft.layer_ids) != 1
            or draft.shared_layer_to_owner
        ):
            raise ValueError(
                f"{target.pool_name}: packed draft must own one matching layer"
            )
        draft.buffer_info.validate(layer_ids=draft.layer_ids)
        if draft.buffer_info.page_size != target.buffer_info.page_size:
            raise ValueError(f"{target.pool_name}: packed draft page coverage differs")
        layer_mapping[len(target_model_layer_ids) + depth] = offset
        offset += len(draft.buffer_info.buffers)
    return layer_mapping


def bind_host_pool_buffers(
    *,
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    target_model_layer_ids: tuple[int, ...],
) -> tuple[DeviceBufferInfo, dict[int, int]]:
    layer_mapping = bind_pool_layers(
        target=target, drafts=drafts, target_model_layer_ids=target_model_layer_ids
    )
    buffers = target.buffer_info.packed_with(
        tuple(draft.buffer_info for draft in drafts)
    )
    return buffers, layer_mapping

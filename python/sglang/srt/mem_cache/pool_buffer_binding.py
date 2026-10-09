from __future__ import annotations

from sglang.srt.mem_cache.device_pool_info import (
    DevicePoolInfo,
    EncodedPageBuffers,
    IndexKeyBufferInfo,
    MLABufferInfo,
)


def bind_packed_pool_buffers(
    *,
    target: DevicePoolInfo,
    drafts: tuple[DevicePoolInfo, ...],
    model_to_transfer_layer: dict[int, int],
    target_layer_num: int,
) -> tuple[MLABufferInfo | IndexKeyBufferInfo, dict[int, int]]:
    info = target.buffer_info
    info.validate(layer_ids=target.layer_ids)
    layer_mapping = {
        model_to_transfer_layer[layer]: position
        for position, layer in enumerate(target.layer_ids)
    }
    if any(layer < 0 or layer >= target_layer_num for layer in layer_mapping):
        raise ValueError(
            f"{target.pool_name}: model layers are outside the target stage"
        )
    if target.shared_layer_to_owner:
        raise ValueError("this transfer binding does not support shared owner layers")
    buffers = list(
        info.buffers.buffers if isinstance(info, IndexKeyBufferInfo) else info.buffers
    )
    for depth, draft in enumerate(drafts):
        draft_info = draft.buffer_info
        if (
            draft.pool_name != target.pool_name
            or draft.indices_from_pool != target.indices_from_pool
            or len(draft.layer_ids) != 1
            or draft.shared_layer_to_owner
        ):
            raise ValueError(
                f"{target.pool_name}: packed draft must own one matching layer"
            )
        if (
            type(draft_info) is not type(info)
            or draft_info.page_size != info.page_size
            or draft_info.compress_ratio != info.compress_ratio
        ):
            raise ValueError(f"{target.pool_name}: packed draft page format differs")
        draft_info.validate(layer_ids=draft.layer_ids)
        if isinstance(info, IndexKeyBufferInfo):
            if draft_info.buffers.encoding != info.buffers.encoding:
                raise ValueError(f"{target.pool_name}: packed draft encoding differs")
            draft_buffers = draft_info.buffers.buffers
        else:
            draft_buffers = draft_info.buffers
        if (
            draft_buffers[0].shape[1:] != buffers[0].shape[1:]
            or draft_buffers[0].dtype != buffers[0].dtype
            or draft_buffers[0].device != buffers[0].device
        ):
            raise ValueError(f"{target.pool_name}: packed draft buffer format differs")
        layer_mapping[target_layer_num + depth] = len(buffers)
        buffers.extend(draft_buffers)
    if isinstance(info, IndexKeyBufferInfo):
        packed = IndexKeyBufferInfo(
            page_size=info.page_size,
            compress_ratio=info.compress_ratio,
            buffers=EncodedPageBuffers(
                buffers=tuple(buffers), encoding=info.buffers.encoding
            ),
        )
    else:
        packed = MLABufferInfo(
            page_size=info.page_size,
            compress_ratio=info.compress_ratio,
            buffers=tuple(buffers),
        )
    return packed, layer_mapping

"""Device buffer facts consumed by host assembly.

The device provider enumerates persistent tensors. Host assembly owns capacity,
layer routing and transfers. This first consumer describes Mamba checkpoints.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName


class MambaStateBufferInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Persistent checkpoint tensors. Layer and checkpoint axes come first.

    Extra buffers share checkpoint IDs but have no layer axis. Speculative
    intermediate tensors are work space and do not belong in host checkpoints.
    """

    temporal: torch.Tensor
    conv: tuple[torch.Tensor, ...]
    extra_checkpoint_buffers: tuple[torch.Tensor, ...] = ()

    def validate(self, *, layer_ids: tuple[int, ...] | None = None) -> None:
        shape = tuple(self.temporal.shape)
        if self.temporal.ndim < 2 or not self.conv:
            raise ValueError(
                f"Mamba temporal requires [layer, checkpoint, ...] and conv buffers, "
                f"got temporal.shape={shape}, conv count={len(self.conv)}"
            )
        if layer_ids is not None and len(layer_ids) != shape[0]:
            raise ValueError(
                f"Mamba layer_ids={layer_ids} must match temporal.shape[0]={shape[0]}"
            )
        for index, buffer in enumerate(self.conv):
            if buffer.shape[:2] != self.temporal.shape[:2]:
                raise ValueError(
                    f"Mamba conv[{index}].shape={tuple(buffer.shape)} must start "
                    f"with layer/checkpoint axes {shape[:2]}"
                )
            if buffer.dtype != self.conv[0].dtype:
                raise ValueError(
                    f"Mamba conv[{index}].dtype={buffer.dtype}, expected {self.conv[0].dtype}"
                )
        for index, buffer in enumerate(self.extra_checkpoint_buffers):
            if buffer.ndim < 1 or buffer.shape[0] != shape[1]:
                raise ValueError(
                    f"Mamba extra_checkpoint_buffers[{index}].shape={tuple(buffer.shape)} "
                    f"must start with checkpoint capacity {shape[1]}"
                )
        for buffer in (*self.conv, *self.extra_checkpoint_buffers):
            if buffer.device != self.temporal.device:
                raise ValueError(
                    f"Mamba buffer.device={buffer.device}, expected {self.temporal.device}"
                )


class DevicePoolInfo(msgspec.Struct, frozen=True, kw_only=True):
    pool_name: PoolName
    indices_from_pool: PoolName
    # Model layer IDs, including the PP stage offset, in buffer-axis order.
    layer_ids: tuple[int, ...]
    buffer_info: MambaStateBufferInfo

    def __post_init__(self) -> None:
        # A PP stage may own no checkpoint layers. Buffer validation decides
        # whether an empty layer axis is valid for the physical format.
        if len(set(self.layer_ids)) != len(self.layer_ids) or any(
            layer < 0 for layer in self.layer_ids
        ):
            raise ValueError(f"{self.pool_name}: invalid model layers {self.layer_ids}")

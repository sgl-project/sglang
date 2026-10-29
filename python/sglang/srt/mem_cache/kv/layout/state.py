"""The state layout: conv and temporal state rows of every layer in one slot
entry, and the typed view that reads, clears and copies them over a band of a
`KVArena`."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.kv.layout.base import SubPoolSpec
from sglang.srt.mem_cache.layout.fused_draft import DraftStateRegion
from sglang.srt.mem_cache.layout.token_major import (
    align_entry_bytes,
    align_part_offset,
    build_mamba_entry_views,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.platforms import current_platform
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo
    from sglang.srt.mem_cache.kv.memory.arena import KVArena

logger = logging.getLogger(__name__)


def _prod(iterable) -> int:
    out = 1
    for x in iterable:
        out *= int(x)
    return out


@dataclass(frozen=True, kw_only=True)
class MambaSubPoolSpec(SubPoolSpec):
    """Per-slot layout of one Mamba-shaped sub-pool.

    With `draft_region` (a `DraftStateRegion`) set, the draft's state block
    follows the host's inside every slot's entry:

        [ conv[0] x L | ... | temporal x L | draft conv[0] x Ld | ... | pad ]

    Unfused entries keep their raw, unaligned size.
    """

    conv_state_shapes: Tuple[Tuple[int, ...], ...]  # one shape per conv tensor
    conv_dtype: torch.dtype
    temporal_state_shape: Tuple[int, ...]
    temporal_dtype: torch.dtype
    conv_slice_axis: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert len(self.conv_state_shapes) > 0, "conv_state_shapes must be non-empty"
        if self.draft_region is not None:
            assert isinstance(self.draft_region, DraftStateRegion), (
                "mamba state entries carry a draft STATE region, never dense K/V rows"
            )
            self.draft_region.validate()

    def conv_row_bytes(self, idx: int) -> int:
        return _prod(self.conv_state_shapes[idx]) * self.conv_dtype.itemsize

    def temporal_row_bytes(self) -> int:
        return _prod(self.temporal_state_shape) * self.temporal_dtype.itemsize

    def host_entry_bytes(self) -> int:
        """Host (target-only) bytes for one slot, before any draft block."""
        total = 0
        for i in range(len(self.conv_state_shapes)):
            total += self.layer_num * self.conv_row_bytes(i)
        total += self.layer_num * self.temporal_row_bytes()
        return total

    def draft_offset_in_entry(self) -> int:
        """Byte offset of the fused draft state block inside one slot's entry."""
        assert self.draft_region is not None
        return align_part_offset(self.host_entry_bytes())

    def entry_bytes(self) -> int:
        if self.draft_region is None:
            return self.host_entry_bytes()
        return align_entry_bytes(
            self.draft_offset_in_entry() + self.draft_region.entry_bytes()
        )

    def get_dtype(self) -> torch.dtype:
        return self.conv_dtype  # representative state dtype; matches MambaPool.dtype

    def views(self, arena) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-layer conv and temporal views over this sub-pool's band of
        ``arena``, indexed by the physical slot id."""
        return build_mamba_entry_views(
            arena.raw,
            layer_num=self.layer_num,
            conv_state_shapes=self.conv_state_shapes,
            conv_dtype=self.conv_dtype,
            temporal_state_shape=self.temporal_state_shape,
            temporal_dtype=self.temporal_dtype,
            max_slots=arena.extent_entries(self.name),
            entry_bytes=self.entry_bytes(),
        )

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-layer conv and temporal views of the draft state block fused
        into this sub-pool's entries: the same slots, the same ids."""
        region = self.draft_region
        assert isinstance(region, DraftStateRegion), (
            f"sub-pool {self.name!r} carries no fused draft state region"
        )
        return build_mamba_entry_views(
            arena.raw,
            layer_num=region.lane_num,
            conv_state_shapes=region.state.conv_state_shapes,
            conv_dtype=region.state.conv_dtype,
            temporal_state_shape=region.state.temporal_state_shape,
            temporal_dtype=region.state.temporal_dtype,
            max_slots=arena.extent_entries(self.name),
            entry_bytes=self.entry_bytes(),
            offset_bytes=self.draft_offset_in_entry(),
        )


class StateView:
    """One state sub-pool's conv and temporal state over its band of a
    `KVArena`: per-layer ``[layer, slot, ...]`` tensors whose slot stride is the
    whole entry, indexed by the physical slot id. Clears and copies act on whole
    entries, so a fused draft's state block moves with the host's. Slots and
    their virtual-to-physical tables are the chain allocator's state band.

    The speculative-decoding work space (per draft token, per running request)
    is not per slot, so it lives beside the arena.
    """

    # The state containers the attention backends read (`mamba_cache`).
    State = MambaPool.State
    SpeculativeState = MambaPool.SpeculativeState
    # Each conv state is laid out (dim, K-1), its window last.
    conv_window_axis = -1
    host_capacity_tokens: Optional[int] = None
    host_capacity_bytes: Optional[int] = None
    enable_custom_mem_pool = False
    custom_mem_pool = None
    debug_memory_pool = False
    conv_shard_groups = None
    # GDN/KDA ReplaySSM is not supported in the arena: its rings are not per
    # slot entry. Readers test these for None.
    enable_linear_replayssm = False
    linear_replayssm_cache_len = 16
    replayssm_write_pos = None
    replayssm_is_kda = False
    enable_linear_replayssm_spec = False
    replayssm_spec_fold = False
    replayssm_spec_write_pos = None
    replayssm_cache_base = None
    replayssm_is_flush = None

    def __init__(
        self,
        *,
        arena: "KVArena",
        spec: MambaSubPoolSpec,
        spec_state_size: int,
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        assert spec.layer_num == len(mamba_layer_ids)
        self._arena = arena
        self._sub_pool_name = spec.name
        self._entry_bytes = spec.entry_bytes()
        # PD state transfer maps entries by global layer id.
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.num_mamba_layers = spec.layer_num
        self.conv_slice_axis = spec.conv_slice_axis
        self.device = arena.device
        max_slots = arena.extent_entries(spec.name)
        self.size = max_slots - 1  # slot 0 is reserved
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        conv_views, temporal_view = spec.views(arena)
        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv views hold {conv_views[0].shape[0]} layers, expected "
            f"{self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == max_slots, (
            f"conv views hold {conv_views[0].shape[1]} slots, expected {max_slots}"
        )
        if speculative_num_draft_tokens is None:
            self.mamba_cache = self.State(conv=list(conv_views), temporal=temporal_view)
        else:
            t_shape = spec.temporal_state_shape
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                intermediate_ssm = torch.zeros(
                    size=(
                        self.num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        t_shape[0],
                        t_shape[1],
                        t_shape[2],
                    ),
                    dtype=spec.temporal_dtype,
                    device=arena.device,
                )
                intermediate_conv_window = [
                    torch.zeros(
                        size=(
                            self.num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            c_shape[0],
                            c_shape[1],
                        ),
                        dtype=spec.conv_dtype,
                        device=arena.device,
                    )
                    for c_shape in spec.conv_state_shapes
                ]
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm,
                intermediate_conv_window=intermediate_conv_window,
            )
        self.mem_usage = arena.total_bytes / (1 << 30)
        logger.info(
            "State view %s over the KV arena: max_slots=%d, num_mamba_layers=%d",
            spec.name,
            max_slots,
            self.num_mamba_layers,
        )

    def register_slot_state(self, state) -> None:
        # The arena moves whole entries, so every per-slot state must live in
        # the entry: side states registered beside it would be left behind.
        raise ValueError(
            "--enable-unified-memory keeps a slot's whole state in its entry; "
            "side states registered beside the state pool are not supported"
        )

    # -- reads --

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def get_speculative_mamba2_params_all_layers(self):
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def get_kv_size_bytes(self) -> int:
        return self.mamba_cache.mem_usage_bytes()

    # -- slot operations, on physical slot ids and whole entries --

    def clear_slots(self, indices: torch.Tensor) -> None:
        # index_fill_, not `view[indices] = 0`: assigning a Python scalar
        # through an index tensor synchronizes the host.
        self._arena.entries(self._sub_pool_name).index_fill_(
            0, indices.to(torch.int64), 0
        )

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        entries = self._arena.entries(self._sub_pool_name)
        entries[dst_indices] = entries[src_indices]

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        temporal_cpu = self.mamba_cache.temporal[:, indices].to(
            "cpu", non_blocking=True
        )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices) -> None:
        if len(mamba_cache_cpu) == 3:
            conv_cpu, temporal_cpu, _ = mamba_cache_cpu
        else:
            conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        current_platform.synchronize()

    # -- host transfer and PD --

    def get_device_pool_infos(self) -> "tuple[DevicePoolInfo, ...]":
        """Checkpoint buffers for host transfer, without the speculative work
        space."""
        from sglang.srt.mem_cache.device_pool_info import (
            DevicePoolInfo,
            MambaStateBufferInfo,
        )
        from sglang.srt.mem_cache.hicache_storage import PoolName

        return (
            DevicePoolInfo(
                pool_name=PoolName.MAMBA,
                indices_from_pool=PoolName.MAMBA,
                layer_ids=tuple(self.mamba_layer_ids),
                buffer_info=MambaStateBufferInfo(
                    temporal=self.mamba_cache.temporal,
                    conv=tuple(self.mamba_cache.conv),
                    extra_checkpoint_buffers=(),
                ),
            ),
        )

    # The PD transfer item is a whole slot entry, addressed as
    # ``raw_ptr + physical_slot * entry_bytes``. An entry cannot be TP-resliced
    # or PP-subset, so the per-tensor metadata stays empty and both sides must
    # build identical state specs (equal attention TP, pp=1).

    def get_contiguous_buf_infos(self):
        raw = self._arena.raw
        return [raw.data_ptr()], [raw.numel()], [self._entry_bytes]

    def get_state_dim_per_tensor(self):
        return []

    def get_state_layer_ids(self):
        return []

    def get_state_slice_outer_counts(self):
        return []

    def get_state_conv_shard_groups(self):
        return []


class DraftStateView(StateView):
    """A fused draft runner's recurrent state: views of the draft block inside
    every slot entry of the host's state sub-pool, laid out as the host's
    (``mamba_cache.conv[stream][layer]``). Slot ids are the target's physical
    ones. The host's whole-entry operations carry the block, so the draft's own
    clear and copy touch the block alone, and it has no transfer or copy of its
    own."""

    def __init__(
        self,
        *,
        arena: "KVArena",
        spec: MambaSubPoolSpec,
        layer_lanes: "dict[int, int]",
    ):
        region = spec.draft_region
        assert isinstance(region, DraftStateRegion), (
            f"sub-pool {spec.name!r} carries no fused draft state region"
        )
        layer_ids = sorted(layer_lanes)
        assert layer_ids, "a fused draft state view binds at least one layer"
        lanes = [layer_lanes[layer_id] for layer_id in layer_ids]
        # A runner's lanes are one contiguous range, so its layers are one
        # strided slice of the region's views (a gather would copy).
        start = lanes[0]
        assert lanes == list(range(start, start + len(lanes))) and (
            0 <= start and start + len(lanes) <= region.lane_num
        ), (
            f"draft state lanes {lanes} must be a contiguous range within "
            f"{region.lane_num}"
        )
        stop = start + len(lanes)
        conv_views, temporal_view = spec.draft_views(arena)
        self._arena = arena
        self._sub_pool_name = spec.name
        self._entry_bytes = spec.entry_bytes()
        self.layer_lanes = dict(layer_lanes)
        self.mamba_layer_ids = layer_ids
        self.num_mamba_layers = len(layer_ids)
        self.conv_slice_axis = spec.conv_slice_axis
        self.device = arena.device
        self.size = arena.extent_entries(spec.name) - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
        self.mamba_cache = self.State(
            conv=[view[start:stop] for view in conv_views],
            temporal=temporal_view[start:stop],
        )
        self.mem_usage = 0.0  # fused into the host entries

    def clear_slots(self, indices: torch.Tensor) -> None:
        """Zero this runner's state block in the given slots, the block only:
        the host's streams share the entry and stay live while a new request
        takes over the draft's block. The host's whole-entry clear covers the
        block too, but only on a plain extend forward, not on the decode,
        target-verify and draft-extend forwards the draft needs it in."""
        for view in self.mamba_cache.conv:
            view[:, indices] = 0
        self.mamba_cache.temporal[:, indices] = 0

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """Copy-on-write this runner's state block, the block only."""
        for view in self.mamba_cache.conv:
            view[:, dst_indices] = view[:, src_indices]
        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
            :, src_indices
        ]

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "fused draft state has no region of its own; state transfer is unsupported"
        )

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offloading is unsupported for fused draft state")

    def load_cpu_copy(self, mamba_cache_cpu, indices) -> None:
        raise NotImplementedError("CPU offloading is unsupported for fused draft state")

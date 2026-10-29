"""The state layout: conv and temporal state rows of every layer in one slot
entry, and the typed view that reads, clears and copies them over a band of a
`KVArena`."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.kv.decl import StateReplay, TensorPartDecl
from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions, SubPoolSpec
from sglang.srt.mem_cache.kv.layout.parts import TensorEntryLayout
from sglang.srt.mem_cache.layout.fused_draft import DraftStateRegion
from sglang.srt.mem_cache.layout.token_major import (
    align_entry_bytes,
    align_part_offset,
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
    extra_parts: Tuple[TensorPartDecl, ...] = ()
    replay: Optional[StateReplay] = None

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

    def entry_layout(self) -> TensorEntryLayout:
        layers = tuple(range(self.layer_num))
        return TensorEntryLayout(
            (
                *(
                    TensorPartDecl(
                        name=f"conv.{i}",
                        shape=shape,
                        dtype=self.conv_dtype,
                        layer_ids=layers,
                        slice_axis=self.conv_slice_axis,
                    )
                    for i, shape in enumerate(self.conv_state_shapes)
                ),
                TensorPartDecl(
                    name="temporal",
                    shape=self.temporal_state_shape,
                    dtype=self.temporal_dtype,
                    layer_ids=layers,
                    slice_axis=0,
                ),
                *self.extra_parts,
            )
        )

    def host_entry_bytes(self) -> int:
        return self.entry_layout().entry_bytes

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

    def part_views(self, arena):
        return self.entry_layout().views(
            arena.raw, arena.extent_entries(self.name), entry_bytes=self.entry_bytes()
        )

    def views(self, arena) -> Tuple[List[torch.Tensor], torch.Tensor]:
        views = self.part_views(arena)
        return (
            [views[f"conv.{i}"] for i in range(len(self.conv_state_shapes))],
            views["temporal"],
        )

    def draft_views(self, arena) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Per-layer conv and temporal views of the draft state block fused
        into this sub-pool's entries: the same slots, the same ids."""
        region = self.draft_region
        assert isinstance(region, DraftStateRegion), (
            f"sub-pool {self.name!r} carries no fused draft state region"
        )
        views = region.entry_layout().views(
            arena.raw,
            arena.extent_entries(self.name),
            entry_bytes=self.entry_bytes(),
            offset_bytes=self.draft_offset_in_entry(),
        )
        return (
            [views[f"conv.{i}"] for i in range(len(region.state.conv_state_shapes))],
            views["temporal"],
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
        mamba_layer_ids: List[int],
        enable_memory_saver: bool = False,
        request_parts: Optional[dict[str, torch.Tensor]] = None,
        intermediate_ssm: Optional[torch.Tensor] = None,
        intermediate_conv_window: Optional[List[torch.Tensor]] = None,
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
        self.extra_parts = spec.extra_parts
        part_views = spec.part_views(arena)
        self.extra_views = {
            part.name: part_views[part.name] for part in spec.extra_parts
        }
        self._checkpoint_extra_buffers = tuple(
            self.extra_views[part.name].movedim(part.row_axis, 0)
            for part in spec.extra_parts
        )
        if spec.draft_region is not None:
            begin = spec.draft_offset_in_entry()
            end = begin + spec.draft_region.entry_bytes()
            self._checkpoint_extra_buffers += (arena.entries(spec.name)[:, begin:end],)
        for part in spec.extra_parts:
            self.extra_views[part.name].select(part.row_axis, 0).fill_(part.reset_value)

        assert conv_views[0].shape[0] == self.num_mamba_layers, (
            f"conv views hold {conv_views[0].shape[0]} layers, expected "
            f"{self.num_mamba_layers}"
        )
        assert conv_views[0].shape[1] == max_slots, (
            f"conv views hold {conv_views[0].shape[1]} slots, expected {max_slots}"
        )
        request_parts = {} if request_parts is None else request_parts
        payloads = self.extra_views | request_parts
        ring_fields = {
            name: payloads.get(name)
            for name in (
                "replayssm_d",
                "replayssm_k",
                "replayssm_g",
                "replayssm_rawv",
                "replayssm_rawk",
                "replayssm_beta",
            )
        }
        if spec.replay is not None:
            replay = spec.replay
            self.enable_linear_replayssm = replay.decode
            self.enable_linear_replayssm_spec = replay.speculative
            self.replayssm_spec_fold = replay.speculative and replay.is_kda
            self.replayssm_is_kda = replay.is_kda
            self.linear_replayssm_cache_len = replay.cache_len
            self.replayssm_write_pos = payloads.get("replayssm_write_pos")
            self.replayssm_spec_write_pos = payloads.get("replayssm_spec_write_pos")
            self.replayssm_cache_base = payloads.get("replayssm_cache_base")
            self.replayssm_is_flush = payloads.get("replayssm_is_flush")
        if intermediate_conv_window is None:
            self.mamba_cache = self.State(
                conv=list(conv_views), temporal=temporal_view, **ring_fields
            )
        else:
            self.mamba_cache = self.SpeculativeState(
                conv=list(conv_views),
                temporal=temporal_view,
                intermediate_ssm=intermediate_ssm,
                intermediate_conv_window=intermediate_conv_window,
                **ring_fields,
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
        for part in self.extra_parts:
            if part.reset_value != 0:
                self.extra_views[part.name].index_fill_(
                    part.row_axis, indices.to(torch.int64), part.reset_value
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
        extra_cpu = [
            buffer[indices].to("cpu", non_blocking=True)
            for buffer in self._checkpoint_extra_buffers
        ]
        current_platform.synchronize()
        return conv_cpu, temporal_cpu, extra_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices) -> None:
        if len(mamba_cache_cpu) == 3:
            conv_cpu, temporal_cpu, extra_cpu = mamba_cache_cpu
        else:
            conv_cpu, temporal_cpu = mamba_cache_cpu
            extra_cpu = []
        if len(extra_cpu) != len(self._checkpoint_extra_buffers):
            raise ValueError("checkpoint copy omits declared state parts")
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
            self.mamba_cache.temporal.device, non_blocking=True
        )
        for buffer, saved in zip(self._checkpoint_extra_buffers, extra_cpu):
            buffer[indices] = saved.to(buffer.device, non_blocking=True)

        current_platform.synchronize()

    # -- host transfer and PD --

    def host_layout(self):
        (info,) = self.get_device_pool_infos()
        return StateHostLayout(
            buffer_info=info.buffer_info,
            capacity=self.size
            if self.host_capacity_tokens is None
            else self.host_capacity_tokens,
            layer_ids=info.layer_ids,
        )

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
                    extra_checkpoint_buffers=self._checkpoint_extra_buffers,
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
        self._checkpoint_extra_buffers = ()
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


class StateHostLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Checkpoint host rows; speculative scratch never enters a checkpoint."""

    buffer_info: MambaStateBufferInfo
    capacity: int
    layer_ids: tuple[int, ...]

    @property
    def device_layer_num(self) -> int:
        return len(self.layer_ids)

    def build(self, options: HostPoolOptions, *, draft_buffer_infos=()):
        from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost

        self.buffer_info.validate(layer_ids=self.layer_ids)
        for info in draft_buffer_infos:
            info.validate()
        layout = options.layout
        if layout == "page_first_kv_split":
            layout = "page_first_direct"
        return MambaPoolHost(
            self.buffer_info,
            options.ratio,
            options.size,
            device_capacity=self.capacity,
            pin_memory=options.pin_memory,
            allocator_type=options.allocator_type,
            layout=layout,
            mtp_draft_buffer_infos=draft_buffer_infos,
        )


class ShortConvView:
    """Layer access to declared short-convolution state and verify workspace."""

    def __init__(self, parts, tensors, scratch):
        part = next((p for p in parts if p.name == "short_conv"), None)
        self.layer_map = (
            {} if part is None else {layer: i for i, layer in enumerate(part.layer_ids)}
        )
        self.conv_state = tensors.get("short_conv")
        self.intermediate_conv_state = scratch.get("intermediate_short_conv")

    @property
    def enabled(self):
        return self.conv_state is not None

    def layer_cache(self, layer_id):
        return self.conv_state[self.layer_map[layer_id]]

    def layer_intermediate_cache(self, layer_id):
        return (
            None
            if self.intermediate_conv_state is None
            else self.intermediate_conv_state[self.layer_map[layer_id]]
        )

    def clear(self):
        # The arena spans other bands too; fresh state slots are reset by StateView.
        pass


class NGramView:
    """Request-wide history in a declared state entry."""

    def __init__(self, parts, tensors, scratch):
        part = next((p for p in parts if p.name == "ngram"), None)
        self.context = tensors.get("ngram")
        self.intermediate_context = scratch.get("intermediate_ngram")
        self.context_len = 0 if part is None else part.shape[0]
        self.eos_token_id = 0 if part is None else part.reset_value

    @property
    def enabled(self):
        return self.context is not None

    def get_context(self, indices):
        return self.context.index_select(0, indices.to(dtype=torch.long))

    def set_context(self, indices, context):
        if self.context is not None and indices.numel():
            self.context[indices.to(dtype=torch.long)] = context.to(self.context)

    def set_intermediate_context(self, context, indices=None):
        if self.intermediate_context is not None and context.numel():
            context = context.to(self.intermediate_context)
            if indices is None:
                self.intermediate_context[: context.shape[0], : context.shape[1]].copy_(
                    context
                )
            else:
                self.intermediate_context[
                    indices.to(device=context.device, dtype=torch.long),
                    : context.shape[1],
                ] = context

    def clear(self):
        # StateView clears newly acquired slots, including the EOS fill.
        pass

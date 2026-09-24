"""NVSHMEM bootstrap ownership and deterministic symmetric-arena layout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from sglang.srt.afd.comm.base import AFDCommBackend, AFDCommConfig

if TYPE_CHECKING:
    from sglang.srt.afd.afd_handshake import (
        AFDHandshakeInfo,
        AFDHandshakeResult,
    )

logger = logging.getLogger(__name__)

ARENA_ALIGNMENT = 256
_CONTROL_VALUE_BYTES = 8


def _align_up(value: int, alignment: int = ARENA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _parse_wire_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"Unsupported AFD wire dtype {name!r}")
    return dtype


@dataclass(frozen=True)
class NVSHMEMLayerSchema:
    layer_id: int
    hidden_size: int
    top_k: int


@dataclass(frozen=True)
class NVSHMEMWorkLayout:
    hidden_size: int
    top_k: int
    hidden_offset: int
    hidden_bytes: int
    topk_ids_offset: int
    topk_ids_bytes: int
    topk_weights_offset: int
    topk_weights_bytes: int
    request_bytes: int
    output_offset: int
    output_bytes: int


@dataclass(frozen=True)
class NVSHMEMDecodeViews:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor


def _request_layout(
    schema: NVSHMEMLayerSchema,
    capacity: int,
    wire_dtype: torch.dtype,
) -> tuple[int, int, int, int, int, int, int]:
    hidden_offset = 0
    hidden_bytes = capacity * schema.hidden_size * wire_dtype.itemsize
    topk_ids_offset = _align_up(hidden_offset + hidden_bytes)
    topk_ids_bytes = capacity * schema.top_k * torch.int32.itemsize
    topk_weights_offset = _align_up(topk_ids_offset + topk_ids_bytes)
    topk_weights_bytes = capacity * schema.top_k * torch.float32.itemsize
    request_bytes = topk_weights_offset + topk_weights_bytes
    return (
        hidden_offset,
        hidden_bytes,
        topk_ids_offset,
        topk_ids_bytes,
        topk_weights_offset,
        topk_weights_bytes,
        request_bytes,
    )


@dataclass(frozen=True)
class NVSHMEMArenaLayout:
    schemas: tuple[NVSHMEMLayerSchema, ...]
    token_buckets: tuple[int, ...]
    wire_dtype: torch.dtype
    num_slots: int
    num_channels: int
    scattered: bool
    slot_stride: int
    channel_stride: int
    request_offset: int
    response_offset: int
    request_signal_offset: int
    response_signal_offset: int
    request_expected_offset: int
    response_expected_offset: int
    total_bytes: int

    @classmethod
    def build(
        cls,
        moe_layers: Sequence[dict],
        token_buckets: Sequence[int],
        wire_dtype: str,
        num_slots: int,
        num_channels: int = 1,
        scattered: bool = False,
    ) -> NVSHMEMArenaLayout:
        schemas = tuple(
            sorted(
                (
                    NVSHMEMLayerSchema(
                        layer_id=int(layer["layer_id"]),
                        hidden_size=int(layer["hidden_size"]),
                        top_k=int(layer["top_k"]),
                    )
                    for layer in moe_layers
                ),
                key=lambda schema: schema.layer_id,
            )
        )
        if not schemas:
            raise ValueError("AFD NVSHMEM arena requires at least one MoE layer")
        if len({schema.layer_id for schema in schemas}) != len(schemas):
            raise ValueError("AFD NVSHMEM arena found duplicate MoE layer ids")
        if any(schema.hidden_size <= 0 or schema.top_k <= 0 for schema in schemas):
            raise ValueError("AFD NVSHMEM layer dimensions must be positive")

        buckets = tuple(sorted(set(token_buckets)))
        if not buckets or buckets[0] <= 0:
            raise ValueError("AFD NVSHMEM arena requires positive token buckets")
        if num_slots <= 0:
            raise ValueError("AFD NVSHMEM arena requires at least one slot")
        if num_channels <= 0:
            raise ValueError("AFD NVSHMEM arena requires at least one channel")

        dtype = _parse_wire_dtype(wire_dtype)
        max_capacity = buckets[-1]
        max_request_bytes = max(
            _request_layout(schema, max_capacity, dtype)[-1] for schema in schemas
        )
        max_output_bytes = max(
            max_capacity * schema.hidden_size * dtype.itemsize for schema in schemas
        )

        request_offset = 0
        response_offset = _align_up(max_request_bytes)
        channel_stride = _align_up(response_offset + max_output_bytes)
        # Payloads have per-edge channels; each slot shares arrival counters.
        request_signal_offset = num_channels * channel_stride
        response_signal_offset = request_signal_offset + _CONTROL_VALUE_BYTES
        request_expected_offset = response_signal_offset + _CONTROL_VALUE_BYTES
        response_expected_offset = request_expected_offset + _CONTROL_VALUE_BYTES
        slot_stride = _align_up(response_expected_offset + _CONTROL_VALUE_BYTES)
        total_bytes = num_slots * slot_stride

        return cls(
            schemas=schemas,
            token_buckets=buckets,
            wire_dtype=dtype,
            num_slots=num_slots,
            num_channels=num_channels,
            scattered=scattered,
            slot_stride=slot_stride,
            channel_stride=channel_stride,
            request_offset=request_offset,
            response_offset=response_offset,
            request_signal_offset=request_signal_offset,
            response_signal_offset=response_signal_offset,
            request_expected_offset=request_expected_offset,
            response_expected_offset=response_expected_offset,
            total_bytes=total_bytes,
        )

    @property
    def max_capacity(self) -> int:
        return self.token_buckets[-1]

    def slot_offset(self, slot_id: int) -> int:
        if not 0 <= slot_id < self.num_slots:
            raise ValueError(
                f"AFD NVSHMEM slot {slot_id} is outside [0, {self.num_slots})"
            )
        return slot_id * self.slot_stride

    def channel_offset(self, slot_id: int, channel_id: int) -> int:
        if not 0 <= channel_id < self.num_channels:
            raise ValueError(
                f"AFD NVSHMEM channel {channel_id} is outside [0, {self.num_channels})"
            )
        return self.slot_offset(slot_id) + channel_id * self.channel_stride

    def work_layout(
        self,
        layer_id: int,
        capacity: int,
    ) -> NVSHMEMWorkLayout:
        if capacity not in self.token_buckets:
            raise ValueError(
                f"AFD NVSHMEM capacity {capacity} is not a configured token bucket"
            )
        schema = next(
            (schema for schema in self.schemas if schema.layer_id == layer_id),
            None,
        )
        if schema is None:
            raise ValueError(f"AFD NVSHMEM arena has no layer {layer_id}")

        (
            hidden_offset,
            hidden_bytes,
            topk_ids_offset,
            topk_ids_bytes,
            topk_weights_offset,
            topk_weights_bytes,
            request_bytes,
        ) = _request_layout(schema, capacity, self.wire_dtype)
        output_bytes = capacity * schema.hidden_size * self.wire_dtype.itemsize
        if request_bytes > self.response_offset:
            raise RuntimeError("AFD NVSHMEM request layout exceeds its arena region")
        if self.response_offset + output_bytes > self.channel_stride:
            raise RuntimeError("AFD NVSHMEM response layout exceeds its arena region")

        return NVSHMEMWorkLayout(
            hidden_size=schema.hidden_size,
            top_k=schema.top_k,
            hidden_offset=hidden_offset,
            hidden_bytes=hidden_bytes,
            topk_ids_offset=topk_ids_offset,
            topk_ids_bytes=topk_ids_bytes,
            topk_weights_offset=topk_weights_offset,
            topk_weights_bytes=topk_weights_bytes,
            request_bytes=request_bytes,
            output_offset=self.response_offset,
            output_bytes=output_bytes,
        )


def _build_session_arena_layout(
    attn_info: AFDHandshakeInfo, ffn_info: AFDHandshakeInfo
) -> NVSHMEMArenaLayout:
    if attn_info.role != "attn":
        raise ValueError("AFD handshake has no attn-side arena authority")
    if ffn_info.role != "ffn":
        raise ValueError("AFD handshake has no ffn-side arena authority")
    scattered = attn_info.afd_ffn_moe_backend == "deepep"
    if scattered:
        num_channels = (
            attn_info.num_ranks + ffn_info.num_ranks - 1
        ) // ffn_info.num_ranks
    else:
        num_channels = (
            ffn_info.num_ranks + attn_info.num_ranks - 1
        ) // attn_info.num_ranks
    return NVSHMEMArenaLayout.build(
        moe_layers=attn_info.moe_layers,
        token_buckets=attn_info.decode_token_buckets,
        wire_dtype=attn_info.wire_dtype,
        num_slots=attn_info.decode_num_microbatches,
        num_channels=num_channels,
        scattered=scattered,
    )


def _build_joint_pe_map(
    ffn_info: AFDHandshakeInfo,
    attn_members: Sequence[AFDHandshakeInfo],
) -> tuple[int, ...]:
    """Map the joint logical rank space (F ranks first, then every attn
    instance's ranks in session order) to node-major NVSHMEM PEs."""
    for info in (ffn_info, *attn_members):
        if len(info.rank_node_ids) != info.num_ranks or any(
            not node_id for node_id in info.rank_node_ids
        ):
            raise RuntimeError(
                "AFD NVSHMEM topology requires one node ID per "
                f"{info.role} rank: ranks={info.num_ranks}, "
                f"node_ids={info.rank_node_ids}"
            )

    logical_rank_nodes = tuple(ffn_info.rank_node_ids) + tuple(
        node_id for attn_info in attn_members for node_id in attn_info.rank_node_ids
    )
    logical_ranks_by_node: dict[str, list[int]] = {}
    for logical_rank, node_id in enumerate(logical_rank_nodes):
        logical_ranks_by_node.setdefault(node_id, []).append(logical_rank)

    rank_counts = {
        node_id: len(logical_ranks)
        for node_id, logical_ranks in logical_ranks_by_node.items()
    }
    if len(set(rank_counts.values())) != 1:
        raise RuntimeError(
            "AFD NVSHMEM cross-node topology requires the same total number "
            f"of A+F ranks on every node, got {rank_counts}"
        )

    # First-seen node and logical-rank order keep the UID owner (F rank 0)
    # as provider PE 0 while making every node's PEs contiguous.
    provider_order = tuple(
        logical_rank
        for logical_ranks in logical_ranks_by_node.values()
        for logical_rank in logical_ranks
    )
    provider_pe_by_global_rank = [0] * len(provider_order)
    for provider_pe, logical_rank in enumerate(provider_order):
        provider_pe_by_global_rank[logical_rank] = provider_pe
    return tuple(provider_pe_by_global_rank)


@dataclass(frozen=True)
class NVSHMEMWorldPlan:
    """Joint-world description computed identically by every member from the
    handshake roster: F ranks first, then each attn instance's ranks in
    session order. Every member allocates the same joint arena (symmetric
    allocation is collective) and touches only its own session's region."""

    unique_id: bytes
    remote_transport: str
    ffn_info: AFDHandshakeInfo
    attn_members: tuple[AFDHandshakeInfo, ...]
    session_layouts: tuple[NVSHMEMArenaLayout, ...]
    session_arena_offsets: tuple[int, ...]
    total_arena_bytes: int
    pe_by_joint_rank: tuple[int, ...]
    attn_rank_offsets: tuple[int, ...]

    @classmethod
    def build(cls, handshake: AFDHandshakeResult) -> NVSHMEMWorldPlan:
        if not isinstance(handshake.nvshmem_uid, bytes) or not handshake.nvshmem_uid:
            raise ValueError("AFD NVSHMEM world requires a unique ID")
        ffn_info = handshake.local if handshake.local.role == "ffn" else handshake.peer
        remote_transport = ffn_info.nvshmem_remote_transport
        if remote_transport not in {"none", "ibgda"}:
            raise ValueError("AFD NVSHMEM world requires a valid remote transport")
        attn_members = handshake.attn_members
        if attn_members is None:
            attn_info = (
                handshake.local if handshake.local.role == "attn" else handshake.peer
            )
            attn_members = (attn_info,)
        session_layouts = tuple(
            _build_session_arena_layout(attn_info, ffn_info)
            for attn_info in attn_members
        )
        session_arena_offsets = []
        attn_rank_offsets = []
        arena_offset = 0
        rank_offset = 0
        for layout, attn_info in zip(session_layouts, attn_members):
            session_arena_offsets.append(arena_offset)
            attn_rank_offsets.append(rank_offset)
            # Layout sizes are ARENA_ALIGNMENT multiples, so region bases
            # inherit the symmetric heap's alignment.
            arena_offset += layout.total_bytes
            rank_offset += attn_info.num_ranks
        return cls(
            unique_id=handshake.nvshmem_uid,
            remote_transport=remote_transport,
            ffn_info=ffn_info,
            attn_members=attn_members,
            session_layouts=session_layouts,
            session_arena_offsets=tuple(session_arena_offsets),
            total_arena_bytes=arena_offset,
            pe_by_joint_rank=_build_joint_pe_map(ffn_info, attn_members),
            attn_rank_offsets=tuple(attn_rank_offsets),
        )

    @property
    def world_size(self) -> int:
        return len(self.pe_by_joint_rank)

    def session_pe_map(self, session_id: int) -> tuple[int, ...]:
        """Session-local F-first global rank -> joint provider PE."""
        num_ffn = self.ffn_info.num_ranks
        attn_base = num_ffn + self.attn_rank_offsets[session_id]
        num_attn = self.attn_members[session_id].num_ranks
        return (
            self.pe_by_joint_rank[:num_ffn]
            + self.pe_by_joint_rank[attn_base : attn_base + num_attn]
        )

    def local_pe(self, config: AFDCommConfig, session_id: int) -> int:
        return self.session_pe_map(session_id)[config.global_rank]


class NVSHMEMWorldRuntime:
    """Process-wide provider and joint symmetric arena, shared by every
    session's runtime owner (a process can init NVSHMEM only once)."""

    def __init__(self, plan: NVSHMEMWorldPlan, local_pe: int, device_index: int):
        from afd_nvshmem import Provider

        self.plan = plan
        self.local_pe = local_pe
        self._refs = 0
        self._closed = False
        self._provider = Provider.init(
            unique_id=plan.unique_id,
            rank=local_pe,
            world_size=plan.world_size,
            device=device_index,
            remote_transport=plan.remote_transport,
        )
        self._arena: torch.Tensor = self._provider.allocate_arena(
            plan.total_arena_bytes
        )
        if (
            self._arena.dtype != torch.uint8
            or self._arena.shape != (plan.total_arena_bytes,)
            or self._arena.data_ptr() % ARENA_ALIGNMENT != 0
        ):
            raise RuntimeError("afd-nvshmem returned an invalid symmetric arena")

    @property
    def provider(self):
        return self._provider

    @property
    def closed(self) -> bool:
        return self._closed

    def session_arena(self, session_id: int) -> torch.Tensor:
        layout = self.plan.session_layouts[session_id]
        return self._arena.narrow(
            0, self.plan.session_arena_offsets[session_id], layout.total_bytes
        )

    def acquire(self) -> None:
        self._refs += 1

    def release(self) -> None:
        self._refs -= 1
        if self._refs <= 0 and not self._closed:
            self._provider.close()
            self._closed = True


_world_runtime: NVSHMEMWorldRuntime | None = None


def _get_or_create_world_runtime(
    plan: NVSHMEMWorldPlan, local_pe: int, device_index: int
) -> NVSHMEMWorldRuntime:
    global _world_runtime
    if _world_runtime is None or _world_runtime.closed:
        _world_runtime = NVSHMEMWorldRuntime(plan, local_pe, device_index)
    elif _world_runtime.plan.unique_id != plan.unique_id:
        raise RuntimeError(
            "AFD NVSHMEM world already initialized with a different unique ID"
        )
    return _world_runtime


class NVSHMEMRuntimeOwner:
    """Per-session view of the shared provider: session arena region,
    session-local PE mapping, signals and channels."""

    def __init__(
        self,
        config: AFDCommConfig,
        world: NVSHMEMWorldRuntime,
        session_id: int,
    ):
        device = torch.device(config.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"AFD NVSHMEM runtime requires an indexed CUDA device, got {device}"
            )

        layout = world.plan.session_layouts[session_id]
        self._config = config
        self._world = world
        self._layout = layout
        self._provider = world.provider
        self._provider_pe_by_global_rank = world.plan.session_pe_map(session_id)
        self._provider_pe = self._provider_pe_by_global_rank[config.global_rank]
        if self._provider_pe != world.local_pe:
            raise RuntimeError(
                "AFD NVSHMEM session PE mismatch: "
                f"session={session_id} pe={self._provider_pe} "
                f"world_local_pe={world.local_pe}"
            )
        self._closed = False
        if config.role.is_attn():
            self._peer_global_ranks = tuple(
                config.ffn_global_rank(ffn_rank)
                for ffn_rank in config.ffn_ranks_for_attn(
                    config.instance_rank,
                    scattered=layout.scattered,
                )
            )
        else:
            self._peer_global_ranks = tuple(
                config.attn_global_rank(attn_rank)
                for attn_rank in config.attn_ranks_for_ffn(
                    config.instance_rank,
                    scattered=layout.scattered,
                )
            )
        if self._peer_global_ranks:
            (
                self._default_request_channel,
                self._default_response_channel,
            ) = self._channels_for_peer(self._peer_global_ranks[0])
        else:
            self._default_request_channel = 0
            self._default_response_channel = 0
        self._signal_set_peers = frozenset(
            peer for peer in self._peer_global_ranks if self._remote_arrivals(peer) == 1
        )

        torch.cuda.set_device(device)
        world.acquire()
        self._arena: torch.Tensor | None = world.session_arena(session_id)
        self._request_signals = tuple(
            self._uint64_view(
                layout.slot_offset(slot_id) + layout.request_signal_offset
            )
            for slot_id in range(layout.num_slots)
        )
        self._response_signals = tuple(
            self._uint64_view(
                layout.slot_offset(slot_id) + layout.response_signal_offset
            )
            for slot_id in range(layout.num_slots)
        )
        self._request_expected = tuple(
            self._uint64_view(
                layout.slot_offset(slot_id) + layout.request_expected_offset
            )
            for slot_id in range(layout.num_slots)
        )
        self._response_expected = tuple(
            self._uint64_view(
                layout.slot_offset(slot_id) + layout.response_expected_offset
            )
            for slot_id in range(layout.num_slots)
        )
        self._wait_helper: torch.cuda.CUDAGraph | None = None

        logger.info(
            "AFD NVSHMEM runtime ready: role=%s session=%d logical_rank=%d/%d "
            "provider_pe=%d world_size=%d peers=%s peer_pes=%s device=%s "
            "remote_transport=%s slots=%d channels=%d slot_stride=%d "
            "arena_bytes=%d/%d version=%s",
            config.role.value,
            session_id,
            config.global_rank,
            config.world_size,
            self._provider_pe,
            world.plan.world_size,
            self._peer_global_ranks,
            tuple(
                self._provider_pe_by_global_rank[peer]
                for peer in self._peer_global_ranks
            ),
            device,
            world.plan.remote_transport,
            layout.num_slots,
            layout.num_channels,
            layout.slot_stride,
            layout.total_bytes,
            world.plan.total_arena_bytes,
            self._provider.version_info.runtime,
        )

    @property
    def layout(self) -> NVSHMEMArenaLayout:
        return self._layout

    @property
    def arena(self) -> torch.Tensor:
        self._check_open()
        assert self._arena is not None
        return self._arena

    def decode_views(
        self,
        layer_id: int,
        capacity: int,
        slot_id: int = 0,
        *,
        peer_global_rank: int | None = None,
    ) -> NVSHMEMDecodeViews:
        self._check_open()
        work = self._layout.work_layout(layer_id, capacity)
        if peer_global_rank is None:
            request_channel = self._default_request_channel
            response_channel = self._default_response_channel
        else:
            request_channel, response_channel = self._channels_for_peer(
                peer_global_rank
            )
        request_offset = (
            self._layout.channel_offset(slot_id, request_channel)
            + self._layout.request_offset
        )
        return NVSHMEMDecodeViews(
            hidden_states=self._typed_view(
                request_offset + work.hidden_offset,
                work.hidden_bytes,
                self._layout.wire_dtype,
                (capacity, work.hidden_size),
            ),
            topk_ids=self._typed_view(
                request_offset + work.topk_ids_offset,
                work.topk_ids_bytes,
                torch.int32,
                (capacity, work.top_k),
            ),
            topk_weights=self._typed_view(
                request_offset + work.topk_weights_offset,
                work.topk_weights_bytes,
                torch.float32,
                (capacity, work.top_k),
            ),
            output=self._typed_view(
                self._layout.channel_offset(slot_id, response_channel)
                + work.output_offset,
                work.output_bytes,
                self._layout.wire_dtype,
                (capacity, work.hidden_size),
            ),
        )

    def put_request(
        self,
        layer_id: int,
        capacity: int,
        slot_id: int = 0,
        *,
        peer_global_rank: int | None = None,
    ) -> None:
        if not self._config.role.is_attn():
            raise RuntimeError("Only an AFD attn rank can send a decode request")
        peer_global_rank = self._resolve_peer(peer_global_rank)
        request_channel, _ = self._channels_for_peer(peer_global_rank)
        work = self._layout.work_layout(layer_id, capacity)
        self._put(
            self._layout.channel_offset(slot_id, request_channel)
            + self._layout.request_offset,
            work.request_bytes,
            self._request_signals[slot_id],
            peer_global_rank,
        )

    def wait_request(self) -> None:
        if self._wait_helper is None:
            self.launch_wait_request()
            return
        if not self._config.role.is_ffn():
            raise RuntimeError("Only an AFD ffn rank can wait for a decode request")
        self._check_open()
        self._wait_helper.replay()

    def launch_wait_request(self, slot_id: int = 0) -> None:
        if not self._config.role.is_ffn():
            raise RuntimeError("Only an AFD ffn rank can wait for a decode request")
        self._wait(
            self._request_signals[slot_id],
            self._request_expected[slot_id],
            len(self._peer_global_ranks),
        )

    def put_response(
        self,
        layer_id: int,
        capacity: int,
        slot_id: int = 0,
        *,
        peer_global_rank: int | None = None,
    ) -> None:
        if not self._config.role.is_ffn():
            raise RuntimeError("Only an AFD ffn rank can send a decode response")
        peer_global_rank = self._resolve_peer(peer_global_rank)
        _, response_channel = self._channels_for_peer(peer_global_rank)
        work = self._layout.work_layout(layer_id, capacity)
        self._put(
            self._layout.channel_offset(slot_id, response_channel) + work.output_offset,
            work.output_bytes,
            self._response_signals[slot_id],
            peer_global_rank,
        )

    def wait_response(self) -> None:
        if self._wait_helper is None:
            self.launch_wait_response()
            return
        if not self._config.role.is_attn():
            raise RuntimeError("Only an AFD attn rank can wait for a decode response")
        self._check_open()
        self._wait_helper.replay()

    def launch_wait_response(self, slot_id: int = 0) -> None:
        if not self._config.role.is_attn():
            raise RuntimeError("Only an AFD attn rank can wait for a decode response")
        self._wait(
            self._response_signals[slot_id],
            self._response_expected[slot_id],
            len(self._peer_global_ranks),
        )

    def capture_wait_graph(
        self,
        num_waits: int,
        stream: torch.cuda.Stream,
    ) -> torch.cuda.CUDAGraph:
        if num_waits <= 0:
            raise ValueError("AFD wait graph requires at least one wait")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(num_waits):
                if self._config.role.is_attn():
                    self.launch_wait_response()
                else:
                    self.launch_wait_request()
        return graph

    def install_wait_helper(self, graph: torch.cuda.CUDAGraph) -> None:
        if self._wait_helper is not None:
            raise RuntimeError("AFD NVSHMEM wait helper is already installed")
        self._wait_helper = graph

    def close(self) -> None:
        if self._closed:
            return
        self._world.release()
        self._arena = None
        self._closed = True

    def _typed_view(
        self,
        offset: int,
        num_bytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        assert self._arena is not None
        return self._arena.narrow(0, offset, num_bytes).view(dtype).view(shape)

    def _uint64_view(self, offset: int) -> torch.Tensor:
        assert self._arena is not None
        return self._arena.narrow(0, offset, _CONTROL_VALUE_BYTES).view(torch.uint64)

    def _put(
        self,
        offset: int,
        num_bytes: int,
        signal: torch.Tensor,
        peer_global_rank: int,
    ) -> None:
        self._check_open()
        assert self._arena is not None
        payload = self._arena.narrow(0, offset, num_bytes)
        self._provider.put_signal_on_stream(
            payload,
            signal,
            self._provider_pe_by_global_rank[peer_global_rank],
            use_signal_set=peer_global_rank in self._signal_set_peers,
        )

    def _wait(
        self,
        signal: torch.Tensor,
        expected: torch.Tensor,
        arrivals: int,
    ) -> None:
        self._check_open()
        self._provider.wait_next_on_stream(
            signal,
            expected,
            arrivals,
            use_signal_set=arrivals == 1,
        )

    def _remote_arrivals(self, peer_global_rank: int) -> int:
        if self._config.role.is_attn():
            peer_ranks = self._config.attn_ranks_for_ffn(
                peer_global_rank,
                scattered=self._layout.scattered,
            )
        else:
            attn_rank = peer_global_rank - self._config.num_ffn_ranks
            peer_ranks = self._config.ffn_ranks_for_attn(
                attn_rank,
                scattered=self._layout.scattered,
            )
        return len(peer_ranks)

    def _resolve_peer(self, peer_global_rank: int | None) -> int:
        if peer_global_rank is not None:
            return peer_global_rank
        if not self._peer_global_ranks:
            raise RuntimeError("AFD NVSHMEM rank has no data peer")
        return self._peer_global_ranks[0]

    def _channels_for_peer(self, peer_global_rank: int) -> tuple[int, int]:
        if self._config.role.is_attn():
            attn_rank = self._config.instance_rank
            ffn_rank = peer_global_rank
        else:
            attn_rank = peer_global_rank - self._config.num_ffn_ranks
            ffn_rank = self._config.instance_rank
        if self._layout.scattered:
            channel = attn_rank // self._config.num_ffn_ranks
            return channel, channel
        return 0, ffn_rank // self._config.num_attn_ranks

    def _check_open(self) -> None:
        if self._closed or self._arena is None:
            raise RuntimeError("AFD NVSHMEM runtime is closed")


class NVSHMEMCommBackend(AFDCommBackend):
    """NVSHMEM decode transport with an exact-size NCCL prefill path."""

    def __init__(
        self,
        config: AFDCommConfig,
        handshake: AFDHandshakeResult,
    ):
        super().__init__(config)
        device = torch.device(config.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError(
                f"AFD NVSHMEM backend requires an indexed CUDA device, got {device}"
            )
        plan = NVSHMEMWorldPlan.build(handshake)
        session_id = handshake.session_id
        torch.cuda.set_device(device)
        world = _get_or_create_world_runtime(
            plan, plan.local_pe(config, session_id), device.index
        )
        self._runtime = NVSHMEMRuntimeOwner(config, world, session_id)

        from sglang.srt.afd.comm.nccl import NCCLCommBackend

        self._nccl = NCCLCommBackend(config)

    @property
    def decode_transport(self) -> NVSHMEMRuntimeOwner:
        return self._runtime

    def send_to(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        self._nccl.send_to(tensors, peer_global_rank)

    def recv_from(
        self,
        tensors: Sequence[torch.Tensor],
        peer_global_rank: int,
    ) -> None:
        self._nccl.recv_from(tensors, peer_global_rank)

    def startup_barrier(self) -> None:
        self._nccl.startup_barrier()

    def close(self) -> None:
        self._nccl.close()
        self._runtime.close()

"""ROCm DWDP backend using HIP IPC and rank-ordered multi-B tensors."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.moe.dwdp.common import restore_storage_rank
from sglang.srt.layers.moe.dwdp.hsa_copy import HsaSdmaCopyEngine
from sglang.srt.layers.moe.dwdp.layout import DwdpExpertLayout
from sglang.srt.layers.moe.dwdp.tensor_schema import DwdpTensorSchema
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

TensorKey = Tuple[int, str]
PeerTensorKey = Tuple[int, int, str]

_COPIED_TENSOR_ATTRS = (
    "is_shuffled",
    "weight_padded",
    "is_transposed",
)


@dataclass(frozen=True)
class DwdpPartitionView:
    tensors: Tuple[torch.Tensor, ...]
    partition_sizes: Tuple[int, ...]


class RocmIpcDwdpManager:
    """Materialize remote expert partitions in two HIP-allocated staging slots."""

    weight_backend = "ipc"

    def __init__(self, server_args):
        self.dwdp_size = int(server_args.dwdp_size)
        self.dwdp_rank = get_parallel().tp_rank
        self.device_id = torch.cuda.current_device()
        self.layout: Optional[DwdpExpertLayout] = None

        self._moe_layers: List[Tuple[int, FusedMoE]] = []
        self._moe_layer_indices: List[int] = []
        self._schemas: Dict[int, DwdpTensorSchema] = {}
        self._num_fused_shared = 0

        self._local_tensors: Dict[TensorKey, torch.Tensor] = {}
        self._local_routed: Dict[TensorKey, torch.Tensor] = {}
        self._local_shared: Dict[TensorKey, torch.Tensor] = {}
        self._original_replicated_tensors: Dict[TensorKey, torch.Tensor] = {}
        self._peer_tensors: Dict[PeerTensorKey, torch.Tensor] = {}
        self._peer_device_ids: Dict[int, int] = {}
        self._peer_storages: List[torch.UntypedStorage] = []

        self._prefetch_buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        self._copy_streams: Dict[int, torch.cuda.Stream] = {}
        self._prefetch_events: List[Dict[int, torch.cuda.Event]] = []
        self._consume_events: List[torch.cuda.Event] = []
        self._copy_engine = None
        self._copy_tickets: List[List[int]] = [[], []]
        self._initialized = False

    @staticmethod
    def _collect_moe_layers(model: nn.Module) -> List[Tuple[int, FusedMoE]]:
        decoder = model.model if hasattr(model, "model") else model
        result: List[Tuple[int, FusedMoE]] = []
        for layer_idx, layer in enumerate(decoder.layers):
            experts = next(
                (module for module in layer.modules() if isinstance(module, FusedMoE)),
                None,
            )
            if experts is not None:
                result.append((layer_idx, experts))
        return result

    def setup(self, model: nn.Module) -> None:
        if self._initialized:
            return
        arch = torch.cuda.get_device_properties(self.device_id).gcnArchName
        if "gfx950" not in arch:
            raise NotImplementedError(
                f"ROCm IPC DWDP requires gfx950, got {arch!r}"
            )
        if torch.cuda.device_count() < self.dwdp_size:
            raise RuntimeError(
                f"ROCm IPC DWDP needs {self.dwdp_size} visible GPUs on one node, "
                f"got {torch.cuda.device_count()}"
            )
        inaccessible = [
            peer
            for peer in range(torch.cuda.device_count())
            if peer != self.device_id
            and not torch.cuda.can_device_access_peer(self.device_id, peer)
        ]
        if inaccessible:
            raise RuntimeError(
                f"ROCm IPC DWDP requires full XGMI P2P; device {self.device_id} "
                f"cannot access peers {inaccessible}"
            )
        try:
            from aiter.fused_moe import fused_moe_multi_b  # noqa: F401
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ROCm IPC DWDP requires an Aiter build with fused_moe_multi_b"
            ) from error

        self._moe_layers = self._collect_moe_layers(model)
        if not self._moe_layers:
            raise RuntimeError(
                f"DWDP is enabled but no FusedMoE layers were found in "
                f"{type(model).__name__}"
            )
        self._moe_layer_indices = [layer_idx for layer_idx, _ in self._moe_layers]

        expert_counts = {
            experts.num_global_routed_experts for _, experts in self._moe_layers
        }
        if len(expert_counts) != 1:
            raise RuntimeError(
                "ROCm IPC DWDP requires a uniform routed expert count, got "
                f"{sorted(expert_counts)}"
            )
        num_routed = expert_counts.pop()
        if num_routed % self.dwdp_size != 0:
            raise ValueError(
                f"num_routed_experts ({num_routed}) must be divisible by "
                f"dwdp_size ({self.dwdp_size})"
            )

        shared_counts = {
            int(getattr(experts, "num_fused_shared_experts", 0))
            for _, experts in self._moe_layers
        }
        if len(shared_counts) != 1:
            raise RuntimeError(
                "ROCm IPC DWDP requires a uniform fused shared expert count, got "
                f"{sorted(shared_counts)}"
            )
        self._num_fused_shared = shared_counts.pop()
        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed,
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_fused_shared_experts=self._num_fused_shared,
        )

        try:
            self._register_tensors()
            self._exchange_ipc_handles()
            self._allocate_prefetch_buffers()
            self._replicate_small_tensors()
            for _, experts in self._moe_layers:
                experts.bind_dwdp_partitioned_weights()
            self._initialized = True
        except Exception:
            self.cleanup(synchronize_ranks=False)
            raise

        logger.info(
            "ROCm IPC DWDP setup complete: rank=%d/%d, routed=%d, shared=%d",
            self.dwdp_rank,
            self.dwdp_size,
            num_routed,
            self._num_fused_shared,
        )

    def _register_tensors(self) -> None:
        assert self.layout is not None
        local_routed = self.layout.num_experts_per_worker

        reference_specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        reference_names: Optional[Tuple[str, ...]] = None
        for layer_idx, experts in self._moe_layers:
            if experts.quant_method is None:
                raise RuntimeError(f"Layer {layer_idx} has no MoE quantization method")
            schema = experts.quant_method.get_dwdp_tensor_schema(experts)
            schema.validate(experts)
            if reference_names is None:
                reference_names = schema.partitioned
            elif schema.partitioned != reference_names:
                raise RuntimeError(
                    "ROCm IPC DWDP requires identical partitioned tensor names "
                    f"across layers; got {reference_names} and {schema.partitioned}"
                )
            self._schemas[layer_idx] = schema

            for name in schema.partitioned:
                tensor = experts.quant_method.get_dwdp_tensor(experts, name)
                required = local_routed + self._num_fused_shared
                if tensor.ndim == 0 or tensor.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {layer_idx} tensor {name} has shape "
                        f"{tuple(tensor.shape)}, expected at least {required} experts"
                    )
                if not tensor.is_contiguous():
                    raise RuntimeError(
                        f"Layer {layer_idx} tensor {name} must be contiguous for HIP IPC"
                    )

                key = (layer_idx, name)
                self._local_tensors[key] = tensor
                self._local_routed[key] = tensor.narrow(0, 0, local_routed)
                if self._num_fused_shared:
                    self._local_shared[key] = tensor.narrow(
                        0, local_routed, self._num_fused_shared
                    )

                spec = (tuple(tensor.shape[1:]), tensor.dtype)
                previous = reference_specs.setdefault(name, spec)
                if previous != spec:
                    raise RuntimeError(
                        f"ROCm IPC DWDP requires uniform {name} specs across layers; "
                        f"got {previous} and {spec}"
                    )

    def _exchange_ipc_handles(self) -> None:
        group = get_parallel().tp_group
        local_metadata = {}
        for key, tensor in sorted(self._local_tensors.items()):
            handle = tensor.untyped_storage()._share_cuda_()
            local_metadata[key] = {
                "handle": handle,
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "dtype": tensor.dtype,
                "source_device": tensor.device.index,
            }

        all_metadata = [None] * self.dwdp_size
        dist.all_gather_object(
            all_metadata,
            local_metadata,
            group=group.cpu_group,
        )
        local_keys = set(local_metadata)
        for peer_rank, metadata in enumerate(all_metadata):
            if set(metadata) != local_keys:
                raise RuntimeError(
                    "Mismatched ROCm DWDP IPC tensor keys: "
                    f"rank {self.dwdp_rank} has {sorted(local_keys)}, "
                    f"rank {peer_rank} has {sorted(metadata)}"
                )
            if peer_rank == self.dwdp_rank:
                continue

            for (layer_idx, name), item in metadata.items():
                self._peer_device_ids[peer_rank] = int(item["source_device"])
                original_handle = item["handle"]
                redirected_handle = (self.device_id,) + tuple(original_handle)[1:]
                target_device = torch.device("cuda", self.device_id)
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
                    tensor = torch.empty(
                        0,
                        dtype=item["dtype"],
                        device=target_device,
                    ).set_(
                        storage,
                        storage_offset=item["storage_offset"],
                        size=item["shape"],
                        stride=item["stride"],
                    )
                self._peer_storages.append(storage)
                self._peer_tensors[(peer_rank, layer_idx, name)] = tensor

        group.barrier()

    def _allocate_prefetch_buffers(self) -> None:
        assert self.layout is not None
        local_routed = self.layout.num_experts_per_worker
        last_rank = self.dwdp_size - 1
        first_layer = self._moe_layer_indices[0]
        names = self._schemas[first_layer].partitioned
        device = torch.device("cuda", self.device_id)

        for _slot in range(2):
            slot: Dict[str, List[Optional[torch.Tensor]]] = {}
            for name in names:
                reference = self._local_tensors[(first_layer, name)]
                per_expert_shape = tuple(reference.shape[1:])
                peer_buffers: List[Optional[torch.Tensor]] = [None] * self.dwdp_size
                for peer_rank in range(self.dwdp_size):
                    if peer_rank == self.dwdp_rank:
                        continue
                    count = local_routed
                    if peer_rank == last_rank:
                        count += self._num_fused_shared
                    peer_buffers[peer_rank] = torch.empty(
                        (count,) + per_expert_shape,
                        dtype=reference.dtype,
                        device=device,
                    )
                slot[name] = peer_buffers
            self._prefetch_buffers.append(slot)

        self._copy_streams = {
            peer_rank: torch.cuda.Stream(device=device)
            for peer_rank in range(self.dwdp_size)
            if peer_rank != self.dwdp_rank
        }
        self._prefetch_events = [
            {peer_rank: torch.cuda.Event() for peer_rank in self._copy_streams}
            for _ in range(2)
        ]
        self._consume_events = [torch.cuda.Event(), torch.cuda.Event()]
        current_stream = torch.cuda.current_stream(device)
        for event in self._consume_events:
            event.record(current_stream)
        self._copy_engine = HsaSdmaCopyEngine.create_or_none()

    def _replicate_small_tensors(self) -> None:
        assert self.layout is not None
        group = get_parallel().tp_group
        local_routed = self.layout.num_experts_per_worker

        for layer_idx, experts in self._moe_layers:
            schema = self._schemas[layer_idx]
            for name in schema.replicated:
                key = (layer_idx, name)
                storage_tensor = getattr(experts, name)
                if isinstance(storage_tensor, torch.nn.Parameter):
                    storage_tensor = storage_tensor.data
                self._original_replicated_tensors.setdefault(key, storage_tensor)
                tensor = experts.quant_method.get_dwdp_tensor(experts, name)
                required = local_routed + self._num_fused_shared
                if tensor.ndim == 0 or tensor.shape[0] < required:
                    raise RuntimeError(
                        f"Layer {layer_idx} replicated tensor {name} has shape "
                        f"{tuple(tensor.shape)}, expected at least {required} experts"
                    )
                routed = tensor.narrow(0, 0, local_routed)
                shards = [torch.empty_like(routed) for _ in range(self.dwdp_size)]
                dist.all_gather(shards, routed, group=group.device_group)
                full = torch.cat(shards, dim=0)
                if self._num_fused_shared:
                    full = torch.cat(
                        [
                            full,
                            tensor.narrow(0, local_routed, self._num_fused_shared),
                        ],
                        dim=0,
                    )
                experts.replace_expert_tensor(
                    name,
                    restore_storage_rank(storage_tensor, full),
                )

    def _slot_for_layer(self, layer_idx: int) -> int:
        return self._moe_layer_indices.index(layer_idx) % 2

    def _next_moe_layer(self, layer_idx: int, distance: int = 1) -> Optional[int]:
        position = self._moe_layer_indices.index(layer_idx) + distance
        if position >= len(self._moe_layer_indices):
            return None
        return self._moe_layer_indices[position]

    def _disable_hsa_copy_engine(self, completed_slot: int) -> None:
        engine = self._copy_engine
        if engine is None:
            return
        # wait_all() always destroys every submitted ticket, including when one
        # signal reports an error. Do not wait the completed slot twice.
        self._copy_tickets[completed_slot].clear()
        for slot_idx, tickets in enumerate(self._copy_tickets):
            if slot_idx == completed_slot or not tickets:
                continue
            try:
                engine.wait_all(tickets)
            except Exception:
                logger.exception(
                    "Failed while draining HSA DWDP IPC slot %s during fallback",
                    slot_idx,
                )
            finally:
                tickets.clear()
        self._copy_engine = None

    def prefetch_layer(self, layer_idx: int) -> None:
        if not self._prefetch_buffers:
            return
        assert self.layout is not None
        slot_idx = self._slot_for_layer(layer_idx)
        local_routed = self.layout.num_experts_per_worker
        last_rank = self.dwdp_size - 1
        schema = self._schemas[layer_idx]

        if self._copy_engine is not None:
            self._consume_events[slot_idx].synchronize()
            if self._copy_tickets[slot_idx]:
                raise RuntimeError(
                    f"DWDP IPC slot {slot_idx} has outstanding HSA tickets"
                )

        for peer_rank, stream in self._copy_streams.items():
            with torch.cuda.stream(stream):
                if self._copy_engine is None:
                    stream.wait_event(self._consume_events[slot_idx])
                for name in schema.partitioned:
                    source = self._peer_tensors[(peer_rank, layer_idx, name)]
                    destination = self._prefetch_buffers[slot_idx][name][peer_rank]
                    assert destination is not None
                    routed_destination = destination.narrow(0, 0, local_routed)
                    routed_source = source.narrow(0, 0, local_routed)
                    if self._copy_engine is None:
                        routed_destination.copy_(
                            routed_source,
                            non_blocking=True,
                        )
                    else:
                        self._copy_tickets[slot_idx].append(
                            self._copy_engine.submit(
                                routed_destination,
                                routed_source,
                                destination_device=self.device_id,
                                source_device=self._peer_device_ids[peer_rank],
                            )
                        )
                    if peer_rank == last_rank and self._num_fused_shared:
                        destination.narrow(
                            0,
                            local_routed,
                            self._num_fused_shared,
                        ).copy_(
                            self._local_shared[(layer_idx, name)],
                            non_blocking=True,
                        )
                self._prefetch_events[slot_idx][peer_rank].record(stream)

    def prefetch_first_layers(self) -> None:
        if not self._initialized:
            return
        for layer_idx in self._moe_layer_indices[:2]:
            self.prefetch_layer(layer_idx)

    def wait_prefetch(self, layer_idx: int) -> None:
        if not self._initialized:
            return
        slot_idx = self._slot_for_layer(layer_idx)
        if self._copy_engine is not None:
            try:
                self._copy_engine.wait_all(self._copy_tickets[slot_idx])
            except Exception:
                logger.exception(
                    "HSA DWDP IPC prefetch failed for layer %s; retrying with "
                    "HIP copy_",
                    layer_idx,
                )
                self._disable_hsa_copy_engine(slot_idx)
                self.prefetch_layer(layer_idx)
            else:
                self._copy_tickets[slot_idx].clear()
        compute_stream = torch.cuda.current_stream(self.device_id)
        for event in self._prefetch_events[slot_idx].values():
            compute_stream.wait_event(event)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        if not self._initialized:
            return
        slot_idx = self._slot_for_layer(layer_idx)
        self._consume_events[slot_idx].record(torch.cuda.current_stream(self.device_id))
        next_next = self._next_moe_layer(layer_idx, distance=2)
        if next_next is not None:
            self.prefetch_layer(next_next)

    def get_partition_view(
        self,
        layer_idx: int,
        name: str,
        reference: Optional[torch.Tensor] = None,
    ) -> DwdpPartitionView:
        if not self._initialized:
            raise RuntimeError("ROCm IPC DWDP manager is not initialized")
        assert self.layout is not None
        if name not in self._schemas[layer_idx].partitioned:
            raise KeyError(f"Tensor {name!r} is not partitioned for layer {layer_idx}")

        local_routed = self.layout.num_experts_per_worker
        last_rank = self.dwdp_size - 1
        slot_idx = self._slot_for_layer(layer_idx)
        parts: List[torch.Tensor] = []
        sizes: List[int] = []
        for rank in range(self.dwdp_size):
            count = local_routed + (self._num_fused_shared if rank == last_rank else 0)
            if rank == self.dwdp_rank:
                local = self._local_tensors[(layer_idx, name)]
                part = local.narrow(0, 0, count)
            else:
                part = self._prefetch_buffers[slot_idx][name][rank]
                assert part is not None
            parts.append(self._match_reference(part, reference))
            sizes.append(count)
        return DwdpPartitionView(tuple(parts), tuple(sizes))

    def find_partitioned_name(
        self,
        layer_idx: int,
        reference: Optional[torch.Tensor],
    ) -> Optional[str]:
        if reference is None:
            return None
        reference_storage = reference.untyped_storage().data_ptr()
        for name in self._schemas[layer_idx].partitioned:
            tensor = self._local_tensors[(layer_idx, name)]
            if tensor.untyped_storage().data_ptr() == reference_storage:
                return name
        raise KeyError(
            f"No DWDP partitioned tensor in layer {layer_idx} shares storage "
            f"with shape={tuple(reference.shape)} dtype={reference.dtype}"
        )

    @staticmethod
    def _match_reference(
        tensor: torch.Tensor,
        reference: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if reference is None:
            return tensor
        result = tensor
        if result.dtype != reference.dtype:
            result = result.view(reference.dtype)
        target_tail = tuple(reference.shape[1:])
        if tuple(result.shape[1:]) != target_tail:
            target_numel = math.prod(target_tail)
            if result[0].numel() != target_numel:
                raise RuntimeError(
                    f"Cannot view DWDP partition shape {tuple(result.shape)} like "
                    f"{tuple(reference.shape)}"
                )
            result = result.reshape((result.shape[0],) + target_tail)
        for attr in _COPIED_TENSOR_ATTRS:
            if hasattr(reference, attr):
                setattr(result, attr, getattr(reference, attr))
        return result

    def cleanup(self, synchronize_ranks: bool = True) -> None:
        if synchronize_ranks and dist.is_initialized():
            get_parallel().tp_group.barrier()
        if self._copy_engine is not None:
            for tickets in self._copy_tickets:
                try:
                    self._copy_engine.wait_all(tickets)
                except Exception:
                    logger.exception("Failed while draining HSA tickets during cleanup")
                finally:
                    tickets.clear()
        if self._copy_streams and torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)
        layers = dict(self._moe_layers)
        for (layer_idx, name), original in self._original_replicated_tensors.items():
            layers[layer_idx].replace_expert_tensor(name, original)
        for _, experts in self._moe_layers:
            experts.unbind_dwdp_weights()
        self._prefetch_buffers.clear()
        self._prefetch_events.clear()
        self._consume_events.clear()
        self._copy_streams.clear()
        self._peer_tensors.clear()
        self._peer_device_ids.clear()
        self._peer_storages.clear()
        self._local_shared.clear()
        self._local_routed.clear()
        self._local_tensors.clear()
        self._original_replicated_tensors.clear()
        self._schemas.clear()
        self._copy_tickets = [[], []]
        self._copy_engine = None
        self._initialized = False

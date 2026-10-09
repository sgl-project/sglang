"""Canonical logical KV ownership for ordinary dense TP/PP layouts."""

from __future__ import annotations

import msgspec
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Identifier,
    KVSpec,
    Nonnegative,
    Positive,
    StrictStruct,
    Topology,
    canonical_bytes,
    digest_bytes,
    validate_kv_spec,
)


class KVHeadRange(StrictStruct):
    layer_id: Nonnegative
    start: Nonnegative
    end: Positive


class CapturePartition(StrictStruct):
    owner_id: Identifier
    kv_sha256: Digest
    heads: tuple[KVHeadRange, ...]
    include_aux: bool

    @property
    def active(self):
        return self.include_aux or bool(self.heads)

    def local_layers(self, kv: KVSpec):
        if self.kv_sha256 != digest_bytes(canonical_bytes(kv)):
            raise ContractError("capture partition belongs to a different KV contract")
        ranges = {part.layer_id: part for part in self.heads}
        if len(ranges) != len(self.heads):
            raise ContractError("duplicate layer in capture partition")
        layers = []
        for layer in kv.layers:
            if (part := ranges.pop(layer.layer_id, None)) is None:
                continue
            if not 0 <= part.start < part.end <= layer.num_kv_heads:
                raise ContractError("capture partition head range is out of bounds")
            layers.append(
                msgspec.structs.replace(layer, num_kv_heads=part.end - part.start)
            )
        if ranges:
            raise ContractError("capture partition contains an unselected layer")
        return layers


class CaptureLayout(StrictStruct):
    topology: Topology
    partitions: tuple[CapturePartition, ...]

    def partition(self, owner_id: str) -> CapturePartition:
        for partition in self.partitions:
            if partition.owner_id == owner_id:
                return partition
        raise ContractError("rank is outside the capture layout")


def plan_capture_layout(
    kv: KVSpec,
    *,
    tp_size: int,
    pp_layer_ranges: list[tuple[int, int]],
    aux_tp_rank: int = 0,
    dp_rank: int = 0,
) -> CaptureLayout:
    """Use the first TP rank of each replicated-head group as canonical owner.

    Layer ranges are the serving PP stages' half-open global attention-layer
    intervals. Ranks without selected KV remain explicit inactive partitions;
    passing one to a Host pool must not fall back to full-model allocation.
    """
    if (
        type(tp_size) is not int
        or tp_size < 1
        or type(aux_tp_rank) is not int
        or not 0 <= aux_tp_rank < tp_size
        or type(dp_rank) is not int
        or dp_rank < 0
        or not pp_layer_ranges
    ):
        raise ContractError("invalid capture parallel layout")
    try:
        kv = msgspec.convert(msgspec.to_builtins(kv), type=KVSpec)
    except (msgspec.ValidationError, TypeError) as error:
        raise ContractError("invalid KV geometry in capture layout") from error
    validate_kv_spec(kv)
    cursor = 0
    for start, end in pp_layer_ranges:
        if (
            type(start) is not int
            or type(end) is not int
            or start != cursor
            or end <= start
        ):
            raise ContractError("PP layer ranges must be contiguous from layer zero")
        cursor = end
    if any(layer.layer_id >= cursor for layer in kv.layers):
        raise ContractError("selected layer is outside the PP layout")
    for layer in kv.layers:
        heads = layer.num_kv_heads
        if (
            heads < 1
            or (heads >= tp_size and heads % tp_size)
            or (heads < tp_size and tp_size % heads)
        ):
            raise ContractError("KV heads do not divide or replicate evenly across TP")
    pp_size = len(pp_layer_ranges)
    aux_owner = f"dp{dp_rank}-pp{pp_size - 1}-tp{aux_tp_rank}"
    fingerprint = digest_bytes(canonical_bytes(kv))
    partitions = []
    for pp_rank, (start, end) in enumerate(pp_layer_ranges):
        for tp_rank in range(tp_size):
            ranges = []
            for layer in kv.layers:
                if not start <= layer.layer_id < end:
                    continue
                heads = layer.num_kv_heads
                if heads >= tp_size:
                    count = heads // tp_size
                    first = tp_rank * count
                else:
                    replicas = tp_size // heads
                    if tp_rank % replicas:
                        continue
                    first, count = tp_rank // replicas, 1
                ranges.append(
                    KVHeadRange(layer_id=layer.layer_id, start=first, end=first + count)
                )
            owner = f"dp{dp_rank}-pp{pp_rank}-tp{tp_rank}"
            partitions.append(
                CapturePartition(
                    owner_id=owner,
                    kv_sha256=fingerprint,
                    heads=tuple(ranges),
                    include_aux=owner == aux_owner,
                )
            )
    return CaptureLayout(
        topology=Topology(
            tp_size=tp_size,
            pp_size=pp_size,
            aux_owner=aux_owner,
            owners=[part.owner_id for part in partitions if part.active],
        ),
        partitions=tuple(partitions),
    )

"""Build an immutable manifest after request-owned D2H buffers are complete."""

from __future__ import annotations

from datetime import datetime, timezone

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    OWNER,
    ContractError,
    Digest,
    Identifier,
    KVSpec,
    Manifest,
    Provenance,
    SequenceInfo,
    TeacherIdentity,
    TensorDescriptor,
    Topology,
    aux_specs,
    canonical_bytes,
    digest_bytes,
    tensor_bytes,
    validate_kv_spec,
    validate_manifest,
    validate_tensors,
)
from sglang.srt.training_capture.topology import (
    CaptureLayout,
    CapturePartition,
    plan_capture_layout,
)


class SnapshotMetadata(msgspec.Struct, frozen=True, kw_only=True):
    dataset_id: str
    sample_id: str
    generation_id: str
    teacher: TeacherIdentity
    sequence: SequenceInfo
    kv: KVSpec
    provenance: Provenance
    contract_id: str = "maas-target-kv-top128-v1"
    topology: Topology = msgspec.field(default_factory=Topology)


class PreparedSnapshotPartition(msgspec.Struct, frozen=True, kw_only=True):
    owner_id: Identifier
    metadata_sha256: Digest
    valid_kv_tokens: int
    objects: tuple[TensorDescriptor, ...]


def build_snapshot(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
) -> tuple[Manifest, dict[str, torch.Tensor]]:
    """Slicing preserves registered arena storage; nothing here extends its lease."""
    validate_kv_spec(metadata.kv)
    layout = plan_capture_layout(
        metadata.kv,
        tp_size=1,
        pp_layer_ranges=[(0, max(metadata.kv.selected_layer_ids) + 1)],
    )
    part, tensors = prepare_snapshot_partition(
        metadata,
        buffers,
        valid_kv_tokens=valid_kv_tokens,
        partition=layout.partition(OWNER),
    )
    manifest = assemble_snapshot(metadata, [part], layout=layout)
    validate_tensors(manifest, tensors)
    return manifest, tensors


def prepare_snapshot_partition(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
    partition: CapturePartition,
) -> tuple[PreparedSnapshotPartition, dict[str, torch.Tensor]]:
    """Describe only owner-local completed Host views, without copying payloads."""
    sequence, kv = metadata.sequence, metadata.kv
    layers = partition.local_layers(kv)
    if (
        not partition.active
        or partition.owner_id not in metadata.topology.owners
        or partition.include_aux != (partition.owner_id == metadata.topology.aux_owner)
    ):
        raise ContractError("partition does not match snapshot ownership")
    n, r = sequence.total_length, sequence.response_length
    if valid_kv_tokens not in (n - 1, n):
        raise ContractError("only the final token may lack KV")
    prefix = f"draft-data/{metadata.dataset_id}/{metadata.sample_id}/{metadata.generation_id}/"
    objects = []
    tensors = {}

    def describe(name, tensor, suffix, kind, **extra):
        key = prefix + suffix
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ContractError(
                "snapshot construction requires completed contiguous Host buffers"
            )
        obj = TensorDescriptor(
            object_id=suffix.replace("/", "-"),
            name=name,
            kind=kind,
            key=key,
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=list(tensor.shape),
            nbytes=tensor.numel() * tensor.element_size(),
            sha256=digest_bytes(tensor_bytes(tensor)),
            owner_id=partition.owner_id,
            byte_order="little",
            contiguous=True,
            **extra,
        )
        objects.append(obj)
        tensors[key] = tensor

    if partition.include_aux:
        for name, (_, shape) in aux_specs(n, r).items():
            describe(name, buffers[name][: shape[0]], "aux/" + name, "aux")
    heads = {part.layer_id: (part.start, part.end) for part in partition.heads}
    for layer in layers:
        for chunk, start in enumerate(
            range(0, valid_kv_tokens, kv.storage_chunk_tokens)
        ):
            end = min(start + kv.storage_chunk_tokens, valid_kv_tokens)
            for component in ("k", "v"):
                name = f"target_{component}.{layer.layer_id}"
                describe(
                    name,
                    buffers[name][start:end],
                    f"kv/{layer.layer_id}/{partition.owner_id}/{chunk}/{component}",
                    "kv",
                    layer_id=layer.layer_id,
                    component=component,
                    token_range=(start, end),
                    head_range=heads[layer.layer_id],
                )
    return (
        PreparedSnapshotPartition(
            owner_id=partition.owner_id,
            metadata_sha256=digest_bytes(canonical_bytes(metadata)),
            valid_kv_tokens=valid_kv_tokens,
            objects=tuple(objects),
        ),
        tensors,
    )


def assemble_snapshot(
    metadata: SnapshotMetadata,
    partitions: list[PreparedSnapshotPartition],
    *,
    layout: CaptureLayout,
) -> Manifest:
    """Assemble immutable descriptors only; every owner retains its own payloads."""
    topology = metadata.topology
    if (
        topology.tp_size != layout.topology.tp_size
        or topology.pp_size != layout.topology.pp_size
        or topology.aux_owner != layout.topology.aux_owner
        or set(topology.owners) != set(layout.topology.owners)
    ):
        raise ContractError("snapshot topology differs from the ownership plan")
    fingerprint = digest_bytes(canonical_bytes(metadata))
    seen, validity, objects = set(), set(), []
    for prepared in partitions:
        if (
            prepared.owner_id in seen
            or prepared.owner_id not in topology.owners
            or prepared.metadata_sha256 != fingerprint
        ):
            raise ContractError("duplicate, foreign or mismatched snapshot partition")
        partition = layout.partition(prepared.owner_id)
        partition.local_layers(metadata.kv)
        heads = {part.layer_id: (part.start, part.end) for part in partition.heads}
        for obj in prepared.objects:
            if (
                obj.owner_id != prepared.owner_id
                or (obj.kind == "aux" and not partition.include_aux)
                or (obj.kind == "kv" and obj.head_range != heads.get(obj.layer_id))
            ):
                raise ContractError("object differs from canonical owner head ranges")
        seen.add(prepared.owner_id)
        validity.add(prepared.valid_kv_tokens)
        objects.extend(prepared.objects)
    if seen != set(topology.owners) or len(validity) != 1:
        raise ContractError("incomplete owners or inconsistent KV validity")
    manifest = Manifest(
        dataset_id=metadata.dataset_id,
        sample_id=metadata.sample_id,
        generation_id=metadata.generation_id,
        contract_id=metadata.contract_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        teacher=metadata.teacher,
        sequence=metadata.sequence,
        kv=metadata.kv,
        provenance=metadata.provenance,
        objects=objects,
        total_tensor_bytes=sum(obj.nbytes for obj in objects),
        topology=topology,
    )
    if validate_manifest(manifest) != validity.pop():
        raise ContractError("partition KV validity differs from object coverage")
    return manifest

"""Build an immutable manifest after request-owned D2H buffers are complete."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

import msgspec
import torch
from sglang.srt.training_capture.catalog import (
    MAX_CATALOG_REQUEST_BYTES,
    manifest_descriptor,
    seal_request_nbytes,
)
from sglang.srt.training_capture.protocol import (
    ELEMENT_BYTES,
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
    token_ids_sha256: Digest
    valid_kv_tokens: int
    objects: tuple[TensorDescriptor, ...]


def manifest_size_bound(metadata: SnapshotMetadata, *, layout: CaptureLayout) -> int:
    """Bound the producer's JSON bytes at the request's maximum sequence length.

    Use the already validated global layout, including all canonical owners.
    Only one sizing descriptor per layer/component/owner is encoded, independent
    of token chunk count. Payload tensors are neither allocated nor inspected.
    """
    return _manifest_size_bound(metadata, layout=layout)[1]


def manifest_fits_budget(metadata, *, layout, lease, manifest_buffer_bytes):
    envelope, nbytes = _manifest_size_bound(metadata, layout=layout)
    descriptor = manifest_descriptor(envelope, nbytes=nbytes, sha256="0" * 64)
    return nbytes <= manifest_buffer_bytes and (
        seal_request_nbytes(envelope, descriptor, lease) <= MAX_CATALOG_REQUEST_BYTES
    )


def _manifest_size_bound(metadata, *, layout):
    n, r = metadata.sequence.total_length, metadata.sequence.response_length
    kv = metadata.kv
    prefix = f"draft-data/{metadata.dataset_id}/{metadata.sample_id}/{metadata.generation_id}/"
    descriptor_bytes = object_count = total_tensor_bytes = 0

    def count(name, dtype, shape, suffix, kind, owner, copies=1, **extra):
        nonlocal descriptor_bytes, object_count
        descriptor = TensorDescriptor(
            object_id=suffix.replace("/", "-"),
            name=name,
            kind=kind,
            key=prefix + suffix,
            dtype=dtype,
            shape=shape,
            nbytes=math.prod(shape) * ELEMENT_BYTES[dtype],
            sha256="0" * 64,
            owner_id=owner,
            byte_order="little",
            contiguous=True,
            **extra,
        )
        descriptor_bytes += copies * len(canonical_bytes(descriptor))
        object_count += copies
        return descriptor.nbytes

    for name, (dtype, shape) in aux_specs(n, r).items():
        total_tensor_bytes += count(
            name, dtype, shape, "aux/" + name, "aux", layout.topology.aux_owner
        )
    chunks = (n + kv.storage_chunk_tokens - 1) // kv.storage_chunk_tokens
    chunk_tokens = min(n, kv.storage_chunk_tokens)
    geometry = {layer.layer_id: layer for layer in kv.layers}
    for partition in layout.partitions:
        for heads in partition.heads:
            layer = geometry[heads.layer_id]
            width = heads.end - heads.start
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                # Independently maximize numeric field widths. This sizing-only
                # descriptor need not describe a single valid token rectangle.
                count(
                    f"target_{component}.{layer.layer_id}",
                    kv.dtype,
                    [chunk_tokens, width, dim],
                    f"kv/{layer.layer_id}/{partition.owner_id}/{chunks - 1}/{component}",
                    "kv",
                    partition.owner_id,
                    copies=chunks,
                    layer_id=layer.layer_id,
                    component=component,
                    token_range=(n - 1, n),
                    head_range=(heads.start, heads.end),
                )
                total_tensor_bytes += n * width * dim * ELEMENT_BYTES[kv.dtype]
    envelope = Manifest(
        **{
            name: getattr(metadata, name)
            for name in metadata.__struct_fields__
            if name != "sequence"
        },
        sequence=msgspec.structs.replace(metadata.sequence, stop_reason="stop_string"),
        created_at="2000-01-01T00:00:00.000000+00:00",
        objects=[],
        total_tensor_bytes=total_tensor_bytes,
    )
    return envelope, len(
        canonical_bytes(envelope)
    ) + descriptor_bytes + object_count - 1


def build_snapshot(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
) -> tuple[Manifest, dict[str, torch.Tensor]]:
    """Prepare and fully validate a snapshot for callers that consume it directly."""
    manifest, tensors = prepare_snapshot(
        metadata, buffers, valid_kv_tokens=valid_kv_tokens
    )
    validate_tensors(manifest, tensors)
    return manifest, tensors


def prepare_snapshot(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
) -> tuple[Manifest, dict[str, torch.Tensor]]:
    """Describe completed Host views; the writer must validate their contents.

    Slicing preserves registered arena storage; nothing here extends its lease.
    Descriptor digests identify bytes but do not certify their training semantics.
    """
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
    return manifest, tensors


def prepare_snapshot_partition(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
    partition: CapturePartition,
    token_ids: Sequence[int] | None = None,
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
    if token_ids is None:
        if not partition.include_aux:
            raise ContractError(
                "KV-only owners must provide their committed token ledger"
            )
        tokens = buffers["token_ids"][:n]
    else:
        if len(token_ids) != n or any(
            type(token) is not int or not 0 <= token < metadata.teacher.vocab_size
            for token in token_ids
        ):
            raise ContractError("committed tokens differ from the sequence contract")
        tokens = torch.tensor(token_ids, dtype=torch.int32)
    if tokens.dtype != torch.int32 or tuple(tokens.shape) != (n,):
        raise ContractError(
            "token ledger requires exactly one int32 value per position"
        )
    token_digest = digest_bytes(tensor_bytes(tokens))
    if partition.include_aux and token_digest != digest_bytes(
        tensor_bytes(buffers["token_ids"][:n])
    ):
        raise ContractError("aux token payload differs from the committed token ledger")
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
            token_ids_sha256=token_digest,
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
    seen, validity, token_digests, objects = set(), set(), set(), []
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
        token_digests.add(prepared.token_ids_sha256)
        objects.extend(prepared.objects)
    if seen != set(topology.owners) or len(validity) != 1:
        raise ContractError("incomplete owners or inconsistent KV validity")
    aux_token_digests = {
        obj.sha256 for obj in objects if obj.kind == "aux" and obj.name == "token_ids"
    }
    if len(token_digests) != 1 or token_digests != aux_token_digests:
        raise ContractError("owners disagree on the committed token sequence")
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

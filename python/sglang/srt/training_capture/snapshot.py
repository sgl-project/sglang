"""Build an immutable manifest after request-owned D2H buffers are complete."""

from __future__ import annotations

from datetime import datetime, timezone

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    OWNER,
    ContractError,
    KVSpec,
    Manifest,
    Provenance,
    SequenceInfo,
    TeacherIdentity,
    TensorDescriptor,
    aux_specs,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
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


def build_snapshot(
    metadata: SnapshotMetadata,
    buffers: dict[str, torch.Tensor],
    *,
    valid_kv_tokens: int,
) -> tuple[Manifest, dict[str, torch.Tensor]]:
    """Slicing preserves registered arena storage; nothing here extends its lease."""
    sequence, kv = metadata.sequence, metadata.kv
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
            owner_id=OWNER,
            byte_order="little",
            contiguous=True,
            **extra,
        )
        objects.append(obj)
        tensors[key] = tensor

    for name, (_, shape) in aux_specs(n, r).items():
        describe(name, buffers[name][: shape[0]], "aux/" + name, "aux")
    for layer in kv.layers:
        for chunk, start in enumerate(
            range(0, valid_kv_tokens, kv.storage_chunk_tokens)
        ):
            end = min(start + kv.storage_chunk_tokens, valid_kv_tokens)
            for component in ("k", "v"):
                name = f"target_{component}.{layer.layer_id}"
                describe(
                    name,
                    buffers[name][start:end],
                    f"kv/{layer.layer_id}/{OWNER}/{chunk}/{component}",
                    "kv",
                    layer_id=layer.layer_id,
                    component=component,
                    token_range=(start, end),
                    head_range=(0, layer.num_kv_heads),
                )
    manifest = Manifest(
        dataset_id=metadata.dataset_id,
        sample_id=metadata.sample_id,
        generation_id=metadata.generation_id,
        contract_id=metadata.contract_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        teacher=metadata.teacher,
        sequence=sequence,
        kv=kv,
        provenance=metadata.provenance,
        objects=objects,
        total_tensor_bytes=sum(obj.nbytes for obj in objects),
    )
    validate_tensors(manifest, tensors)
    return manifest, tensors

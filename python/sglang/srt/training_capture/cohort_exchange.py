"""Typed owner metadata exchanged after D2H completion, before publication."""

from datetime import datetime, timezone
from typing import Literal

import msgspec
from sglang.srt.training_capture.protocol import (
    ContractError,
    Digest,
    Identifier,
    Positive,
    StrictStruct,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    validate_manifest,
)
from sglang.srt.training_capture.snapshot import (
    PreparedSnapshotPartition,
    SnapshotMetadata,
    assemble_snapshot,
)
from sglang.srt.training_capture.snapshot_writer import OwnerWriteReceipt


class SnapshotOffer(StrictStruct):
    capture_id: Identifier
    fencing_token: Positive
    owner_id: Identifier
    request_sha256: Digest
    execution_sha256: Digest
    partition: PreparedSnapshotPartition | None
    metadata: SnapshotMetadata | None
    created_at: str | None
    version: Literal[1] = 1


def snapshot_offer(cohort, ticket, partition, execution_sha256, prepared, metadata):
    offer = SnapshotOffer(
        capture_id=cohort.lease.capture_id,
        fencing_token=cohort.lease.fencing_token,
        owner_id=partition.owner_id,
        request_sha256=ticket.request_sha256,
        execution_sha256=execution_sha256,
        partition=prepared,
        metadata=metadata,
        created_at=datetime.now(timezone.utc).isoformat()
        if partition.include_aux
        else None,
    )
    # Constructors do not validate msgspec annotations. Freeze and decode now so
    # caller mutation cannot alter the later collective's input.
    payload = canonical_bytes(offer)
    checked = msgspec.json.decode(payload, type=SnapshotOffer)
    _check_owner(checked, cohort, ticket.request_sha256, partition)
    return payload


def _check_owner(offer, cohort, request_sha256, partition):
    if (
        offer.capture_id != cohort.lease.capture_id
        or offer.fencing_token != cohort.lease.fencing_token
        or offer.request_sha256 != request_sha256
        or offer.owner_id != partition.owner_id
        or (offer.partition is not None) != partition.active
        or (offer.metadata is not None) != partition.include_aux
        or (offer.created_at is not None) != partition.include_aux
        or (
            offer.partition is not None
            and offer.partition.owner_id != partition.owner_id
        )
    ):
        raise ContractError("snapshot offer differs from its cohort or canonical owner")


def agree_snapshot(payloads, *, cohort, request_sha256, allocator):
    layout = allocator.layout
    if len(payloads) != len(layout.partitions):
        raise ContractError("incomplete snapshot offer set")
    offers = [msgspec.json.decode(payload, type=SnapshotOffer) for payload in payloads]
    for offer, partition in zip(offers, layout.partitions, strict=True):
        _check_owner(offer, cohort, request_sha256, partition)
    if len({offer.execution_sha256 for offer in offers}) != 1:
        raise ContractError("ranks disagree on effective request execution")
    aux = next(offer for offer in offers if offer.metadata is not None)
    metadata = aux.metadata
    lease = cohort.lease
    if (
        (metadata.dataset_id, metadata.sample_id, metadata.generation_id)
        != (lease.dataset_id, lease.sample_id, lease.generation_id)
        or metadata.contract_id != allocator.config.contract_id
        or metadata.teacher != allocator.teacher
        or metadata.kv != allocator.kv
    ):
        raise ContractError("snapshot metadata differs from the reserved contract")
    manifest = assemble_snapshot(
        metadata,
        [offer.partition for offer in offers if offer.partition is not None],
        layout=layout,
    )
    # All ranks use the aux owner's timestamp, not their local assembly clock.
    manifest = msgspec.structs.replace(manifest, created_at=aux.created_at)
    validate_manifest(manifest)
    result = canonical_bytes(manifest)
    if len(result) > allocator.config.manifest_buffer_bytes:
        raise ContractError("agreed manifest exceeds its registered buffer")
    return result


def check_receipt(receipt, *, cohort, owner_id, manifest_payload):
    receipt = msgspec.convert(msgspec.to_builtins(receipt), type=OwnerWriteReceipt)
    if (
        receipt.capture_id != cohort.lease.capture_id
        or receipt.fencing_token != cohort.lease.fencing_token
        or receipt.owner_id != owner_id
        or receipt.manifest_sha256 != digest_bytes(manifest_payload)
    ):
        raise ContractError("write receipt differs from its owner, fence or manifest")
    return receipt


def agree_receipts(payloads, *, cohort, manifest_payload, layout):
    if len(payloads) != len(layout.partitions):
        raise ContractError("incomplete receipt exchange")
    manifest = decode_manifest(manifest_payload)
    receipts = []
    for payload, partition in zip(payloads, layout.partitions, strict=True):
        receipt = msgspec.json.decode(payload, type=OwnerWriteReceipt | None)
        if (receipt is not None) != partition.active:
            raise ContractError("receipt set differs from active owners")
        if receipt is not None:
            receipts.append(
                check_receipt(
                    receipt,
                    cohort=cohort,
                    owner_id=partition.owner_id,
                    manifest_payload=manifest_payload,
                )
            )
    if {receipt.owner_id for receipt in receipts} != set(manifest.topology.owners):
        raise ContractError("receipts do not cover the agreed manifest")
    return canonical_bytes(receipts)

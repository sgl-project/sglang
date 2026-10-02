"""Manifest-last publication with a durable, metadata-only recovery journal."""

from __future__ import annotations

import base64
import fcntl
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import msgspec
import torch
from sglang.srt.training_capture.catalog import CaptureLease, Catalog, CatalogConflict
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import (
    DTYPES,
    ContractError,
    Digest,
    Identifier,
    Manifest,
    Positive,
    StrictStruct,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
    validate_manifest,
    validate_tensors,
)


class OwnerWriteReceipt(StrictStruct):
    """Trusted producer acknowledgement, never a Catalog lease or retention proof."""

    capture_id: Identifier
    fencing_token: Positive
    owner_id: Identifier
    manifest_sha256: Digest


class PublicationJournal:
    """One process owns a journal; tensor bytes never enter this directory."""

    def __init__(self, directory: str):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.root / "producer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise

    def _path(self, capture_id: str):
        if (
            not capture_id
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for c in capture_id
            )
            or capture_id in (".", "..")
        ):
            raise ContractError("invalid journal capture ID")
        return self.root / f"{capture_id}.json"

    def _sync_directory(self):
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def save(self, lease: CaptureLease, manifest_bytes: bytes) -> None:
        path = self._path(lease.capture_id)
        entry = {
            "lease": msgspec.to_builtins(lease),
            "manifest_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        }
        payload = canonical_bytes(entry)
        if path.exists():
            if path.read_bytes() != payload:
                raise ContractError(
                    "journal entry already has different immutable content"
                )
            return
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            self._sync_directory()
        finally:
            temporary.unlink(missing_ok=True)

    def pending(self):
        for path in sorted(self.root.glob("*.json")):
            if path.stat().st_size > 12 << 20:
                raise ContractError("publication journal metadata is too large")
            value = msgspec.json.decode(path.read_bytes())
            lease = msgspec.convert(value["lease"], type=CaptureLease)
            if path != self._path(lease.capture_id):
                raise ContractError("journal filename and capture identity disagree")
            data = base64.b64decode(value["manifest_base64"], validate=True)
            decode_manifest(data)
            yield lease, data

    def has_pending(self, capture_id: str) -> bool:
        return self._path(capture_id).exists()

    def complete(self, capture_id):
        self._path(capture_id).unlink()
        self._sync_directory()

    def close(self):
        self.lock.close()


class SnapshotWriter:
    """Called by the producer's single background writer, never its hot path.

    All source buffers must already be registered and have completed D2H. The
    Catalog must validate fences against current state, including idempotent
    seal retries, and must not reclaim any published sample via capture expiry.
    """

    def __init__(
        self,
        store: MooncakeSnapshotStore,
        catalog: Catalog,
        journal: PublicationJournal,
    ):
        self.store = store
        self.catalog = catalog
        self.journal = journal

    @staticmethod
    def _check_identity(manifest: Manifest, lease: CaptureLease):
        if (manifest.dataset_id, manifest.sample_id, manifest.generation_id) != (
            lease.dataset_id,
            lease.sample_id,
            lease.generation_id,
        ):
            raise ContractError(
                "capture lease belongs to a different sample generation"
            )

    @staticmethod
    def _manifest_object(manifest, data):
        return {
            "object_id": "manifest",
            "kind": "manifest",
            "key": manifest.key_prefix + "manifest",
            "nbytes": len(data),
            "sha256": digest_bytes(data),
            "owner_id": manifest.topology.aux_owner,
        }

    def _seal(self, manifest, data, lease):
        descriptor = self._manifest_object(manifest, data)
        receipt = self.catalog.seal(
            lease,
            {
                "owner_id": manifest.topology.aux_owner,
                "sequence": msgspec.to_builtins(manifest.sequence),
                "manifest": descriptor,
                "manifest_base64": base64.b64encode(data).decode("ascii"),
                "total_tensor_bytes": manifest.total_tensor_bytes,
                "idempotency_key": f"seal-{lease.capture_id}-{descriptor['sha256']}",
            },
        )
        if (
            receipt.get("state") not in ("PREPARED", "READY")
            or receipt.get("manifest_sha256") != descriptor["sha256"]
        ):
            raise CatalogConflict("Catalog did not prepare this exact manifest")
        return receipt["state"]

    def write(
        self,
        manifest: Manifest,
        tensors: dict[str, torch.Tensor],
        manifest_buffer: torch.Tensor,
        lease: CaptureLease,
    ) -> dict:
        self._check_identity(manifest, lease)
        validate_tensors(manifest, tensors)
        data = canonical_bytes(manifest)
        if len(data) > manifest_buffer.numel() or manifest_buffer.dtype != torch.uint8:
            raise ContractError("manifest exceeds reserved Host buffer")
        objects = [msgspec.to_builtins(obj) for obj in manifest.objects]
        descriptor = self._manifest_object(manifest, data)
        self.catalog.objects(
            lease,
            {
                "phase": "REGISTERED",
                "objects": objects + [descriptor],
                "idempotency_key": f"register-{lease.capture_id}-{descriptor['sha256']}",
            },
        )
        self.store.put_registered_batch(
            [(obj.key, tensors[obj.key], obj.sha256) for obj in manifest.objects]
        )
        self.catalog.objects(
            lease,
            {
                "phase": "WRITTEN",
                "objects": objects,
                "idempotency_key": f"written-{lease.capture_id}-{descriptor['sha256']}",
            },
        )
        # A lost seal response must also be recoverable from exact metadata bytes.
        self.journal.save(lease, data)
        self._seal(manifest, data, lease)
        return self._publish(manifest, data, manifest_buffer, lease)

    def write_partition(
        self,
        manifest: Manifest,
        tensors: dict[str, torch.Tensor],
        lease: CaptureLease,
        *,
        owner_id: str,
    ) -> OwnerWriteReceipt:
        """Write one complete owner partition without publishing a manifest.

        The coordinator first assembles and distributes the same full manifest
        to all owners. Only metadata is shared; each writer holds its own
        completed, registered Host buffers. Catalog capture expiry owns cleanup
        if a writer exits before returning its receipt.
        """
        self._check_identity(manifest, lease)
        if owner_id not in manifest.topology.owners:
            raise ContractError("unregistered tensor partition owner")
        validate_tensors(manifest, tensors, owner_id=owner_id)
        digest = digest_bytes(canonical_bytes(manifest))
        objects = [obj for obj in manifest.objects if obj.owner_id == owner_id]
        descriptors = [msgspec.to_builtins(obj) for obj in objects]
        operation = f"{lease.capture_id}-{owner_id}-{digest}"
        self.catalog.objects(
            lease,
            {
                "phase": "REGISTERED",
                "objects": descriptors,
                "idempotency_key": f"register-partition-{operation}",
            },
        )
        self.store.put_registered_batch(
            [(obj.key, tensors[obj.key], obj.sha256) for obj in objects]
        )
        self.catalog.objects(
            lease,
            {
                "phase": "WRITTEN",
                "objects": descriptors,
                "idempotency_key": f"written-partition-{operation}",
            },
        )
        return OwnerWriteReceipt(
            capture_id=lease.capture_id,
            fencing_token=lease.fencing_token,
            owner_id=owner_id,
            manifest_sha256=digest,
        )

    def publish_partitions(
        self,
        manifest: Manifest,
        receipts: list[OwnerWriteReceipt],
        manifest_buffer: torch.Tensor,
        lease: CaptureLease,
    ) -> dict:
        """The aux coordinator publishes only after every owner confirms WRITTEN.

        Receipts bind the entire manifest and current fence, not just a rank.
        The Catalog must independently verify every WRITTEN descriptor at seal.
        The durable publication journal then uses the ordinary recovery path.
        """
        self._check_identity(manifest, lease)
        validate_manifest(manifest)
        data = canonical_bytes(manifest)
        if len(data) > manifest_buffer.numel() or manifest_buffer.dtype != torch.uint8:
            raise ContractError("manifest exceeds reserved Host buffer")
        self._check_receipts(manifest, receipts, lease, data)
        descriptor = self._manifest_object(manifest, data)
        self.catalog.objects(
            lease,
            {
                "phase": "REGISTERED",
                "objects": [descriptor],
                "idempotency_key": f"register-manifest-{lease.capture_id}-{descriptor['sha256']}",
            },
        )
        self.journal.save(lease, data)
        self._seal(manifest, data, lease)
        return self._publish(manifest, data, manifest_buffer, lease)

    @staticmethod
    def _check_receipts(manifest, receipts, lease, data):
        digest = digest_bytes(data)
        seen = set()
        for receipt in receipts:
            if not isinstance(receipt, OwnerWriteReceipt) or (
                receipt.capture_id != lease.capture_id
                or receipt.fencing_token != lease.fencing_token
                or receipt.manifest_sha256 != digest
                or receipt.owner_id not in manifest.topology.owners
                or receipt.owner_id in seen
            ):
                raise ContractError("stale, duplicate or mismatched owner receipt")
            seen.add(receipt.owner_id)
        if seen != set(manifest.topology.owners):
            raise ContractError("missing owner write receipts")

    @contextmanager
    def _recovery_buffer(self, data):
        retained = sum(
            self.store.registered[p].numel() * self.store.registered[p].element_size()
            for p in self.store.quarantined
        )
        if retained + len(data) > self.store.max_receive_bytes:
            raise ContractError("manifest recovery exceeds receive/quarantine budget")
        buffer = torch.empty(len(data), dtype=torch.uint8, device="cpu")
        self.store.register(buffer)
        try:
            yield buffer
        finally:
            if buffer.data_ptr() not in self.store.quarantined:
                self.store.unregister(buffer)

    def recover_partitions(self, manifest, receipts, lease):
        """Reconcile an attempted publication using frozen in-memory metadata.

        A missing/unreadable journal after an exception cannot prove failure:
        publication may have committed before journal cleanup failed. Retry the
        same identity and bytes, checking Store contents first and using a fresh
        registered manifest buffer. Never reuse an uncertain source arena.
        """
        self._check_identity(manifest, lease)
        validate_manifest(manifest)
        data = canonical_bytes(manifest)
        self._check_receipts(manifest, receipts, lease, data)
        for obj in manifest.objects:
            self.store.get_tensor(obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256)
        with self._recovery_buffer(data) as buffer:
            return self.publish_partitions(manifest, receipts, buffer, lease)

    def _publish(self, manifest, data, manifest_buffer, lease):
        descriptor = self._manifest_object(manifest, data)
        view = manifest_buffer[: len(data)]
        view.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        self.store.put_registered(descriptor["key"], view, descriptor["sha256"])
        self.catalog.objects(
            lease,
            {
                "phase": "WRITTEN",
                "objects": [descriptor],
                "idempotency_key": f"manifest-written-{lease.capture_id}-{descriptor['sha256']}",
            },
        )
        payload = {
            **lease.credentials(),
            "dataset_id": manifest.dataset_id,
            "sample_id": manifest.sample_id,
            "generation_id": manifest.generation_id,
            "manifest_key": descriptor["key"],
            "manifest_sha256": descriptor["sha256"],
            "manifest_nbytes": len(data),
            "contract_id": manifest.contract_id,
            "idempotency_key": f"publish-{manifest.sample_id}-{manifest.generation_id}-{descriptor['sha256']}",
        }
        receipt = self.catalog.publish(payload)
        if (
            receipt.get("state") != "AVAILABLE"
            or not receipt.get("publication_id")
            or "catalog_cursor" not in receipt
        ):
            raise CatalogConflict("Catalog publication receipt is incomplete")
        self.journal.complete(lease.capture_id)
        return receipt

    def recover(self) -> list[dict]:
        """Retry prepared publications without rerunning target inference.

        A rejected fence leaves the entry for operator/Catalog reconciliation.
        Unknown publication outcomes are never treated as permission to delete.
        """
        receipts = []
        for lease, data in self.journal.pending():
            manifest = decode_manifest(data)
            self._check_identity(manifest, lease)
            self._seal(manifest, data, lease)
            # A journal survives producer/data-node loss. Confirm every immutable
            # object still exists before making the recovered reference visible.
            for obj in manifest.objects:
                self.store.get_tensor(obj.key, obj.shape, DTYPES[obj.dtype], obj.sha256)
            with self._recovery_buffer(data) as buffer:
                receipts.append(self._publish(manifest, data, buffer, lease))
        return receipts

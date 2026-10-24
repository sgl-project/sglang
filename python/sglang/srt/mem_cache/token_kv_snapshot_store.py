"""Atomic, digest-checked files for an opt-in same-version decode checkpoint.

This stores bytes exported by the engine. It does not infer that a token prefix,
cache hit, or successfully written file constitutes a restored inference state.
Each TP rank must write its actual tensors before rank zero commits the group.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class TokenKVSnapshotError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(destination: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TokenKVSnapshotStore:
    """Write-once snapshots on a shared filesystem; no pickle is loaded here.

    The scheduler supplies identities from the actual engine, not request labels.
    A unique snapshot ID is also its storage generation. Repeated writes with the
    exact same bytes are harmless; conflicting writers and incomplete TP groups
    are rejected. The engine must deserialize only after ``read_rank`` succeeds.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, snapshot_id: str) -> Path:
        if not isinstance(snapshot_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", snapshot_id
        ):
            raise TokenKVSnapshotError("invalid snapshot ID")
        directory = self.root / snapshot_id
        if directory.is_symlink():
            raise TokenKVSnapshotError("snapshot directory may not be a symlink")
        return directory

    @staticmethod
    def _validate_rank(tp_rank: int, tp_size: int) -> None:
        if (
            type(tp_rank) is not int
            or type(tp_size) is not int
            or tp_size < 1
            or not 0 <= tp_rank < tp_size
        ):
            raise TokenKVSnapshotError("invalid TP rank or world size")

    @staticmethod
    def _validate_identity(identity: dict[str, Any]) -> None:
        if not isinstance(identity, dict) or not identity:
            raise TokenKVSnapshotError("actual engine identity is required")
        if "weight_version" not in identity or identity["weight_version"] is None:
            raise TokenKVSnapshotError("actual engine weight version is required")
        _canonical(identity)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise TokenKVSnapshotError(f"missing or invalid {path.name}") from error
        if not isinstance(value, dict):
            raise TokenKVSnapshotError(f"invalid object in {path.name}")
        return value

    def write_rank(
        self,
        snapshot_id: str,
        *,
        tp_rank: int,
        tp_size: int,
        identity: dict[str, Any],
        request_metadata: dict[str, Any],
        tensor_metadata: dict[str, Any],
        payload: bytes,
    ) -> dict[str, Any]:
        self._validate_rank(tp_rank, tp_size)
        self._validate_identity(identity)
        if not isinstance(payload, bytes) or not payload:
            raise TokenKVSnapshotError("actual nonempty tensor payload is required")
        if not isinstance(request_metadata, dict) or not request_metadata:
            raise TokenKVSnapshotError("request state metadata is required")
        if not isinstance(tensor_metadata, dict) or not tensor_metadata:
            raise TokenKVSnapshotError("tensor layout metadata is required")
        directory = self._directory(snapshot_id)
        directory.mkdir(exist_ok=True)
        metadata = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "identity": identity,
            "request_metadata": request_metadata,
            "tensor_metadata": tensor_metadata,
            "payload_name": f"rank-{tp_rank}.bin",
            "payload_bytes": len(payload),
            "payload_sha256": _digest(payload),
        }
        metadata_path = directory / f"rank-{tp_rank}.json"
        # One engine scheduler owns each TP rank. The exclusive lock also rejects
        # a concurrent/late writer attempting to replace its committed history.
        lock = directory / f"rank-{tp_rank}.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise TokenKVSnapshotError("concurrent rank writer") from error
        os.close(fd)
        try:
            if metadata_path.exists():
                existing = self._load_json(metadata_path)
                if _canonical(existing) != _canonical(metadata):
                    raise TokenKVSnapshotError("conflicting snapshot rank write")
                self._checked_payload(directory, existing)
                return existing
            if (directory / "COMMITTED.json").exists():
                raise TokenKVSnapshotError("cannot append to committed snapshot")
            _atomic_write(directory / metadata["payload_name"], payload)
            _atomic_write(metadata_path, _canonical(metadata))
            return metadata
        finally:
            lock.unlink()

    @staticmethod
    def _checked_payload(directory: Path, metadata: dict[str, Any]) -> bytes:
        expected_name = f"rank-{metadata['tp_rank']}.bin"
        if metadata.get("payload_name") != expected_name:
            raise TokenKVSnapshotError("invalid payload name")
        path = directory / expected_name
        if path.is_symlink():
            raise TokenKVSnapshotError("payload may not be a symlink")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise TokenKVSnapshotError("missing tensor payload") from error
        if len(payload) != metadata.get("payload_bytes") or _digest(
            payload
        ) != metadata.get("payload_sha256"):
            raise TokenKVSnapshotError("tensor payload integrity mismatch")
        return payload

    def commit(self, snapshot_id: str, *, tp_size: int) -> dict[str, Any]:
        self._validate_rank(0, tp_size)
        directory = self._directory(snapshot_id)
        members = []
        shared = None
        for rank in range(tp_size):
            member = self._load_json(directory / f"rank-{rank}.json")
            if (
                member.get("schema_version") != 1
                or member.get("snapshot_id") != snapshot_id
                or member.get("tp_rank") != rank
                or member.get("tp_size") != tp_size
            ):
                raise TokenKVSnapshotError("rank layout mismatch")
            self._validate_identity(member.get("identity"))
            current = _canonical([member["identity"], member["request_metadata"]])
            if shared is not None and current != shared:
                raise TokenKVSnapshotError("TP ranks disagree on model/request state")
            shared = current
            self._checked_payload(directory, member)
            members.append(
                {"rank": rank, "metadata_sha256": _digest(_canonical(member))}
            )
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "tp_size": tp_size,
            "shared_state_sha256": _digest(shared),
            "members": members,
        }
        committed = directory / "COMMITTED.json"
        if committed.exists():
            if _canonical(self._load_json(committed)) != _canonical(manifest):
                raise TokenKVSnapshotError("conflicting snapshot commit")
        else:
            _atomic_write(committed, _canonical(manifest))
        return manifest

    def read_rank(
        self,
        snapshot_id: str,
        *,
        tp_rank: int,
        tp_size: int,
        expected_identity: dict[str, Any],
        expected_request_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        self._validate_rank(tp_rank, tp_size)
        self._validate_identity(expected_identity)
        directory = self._directory(snapshot_id)
        manifest = self._load_json(directory / "COMMITTED.json")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("snapshot_id") != snapshot_id
            or manifest.get("tp_size") != tp_size
            or [m.get("rank") for m in manifest.get("members", [])]
            != list(range(tp_size))
        ):
            raise TokenKVSnapshotError("uncommitted or incompatible TP snapshot")
        metadata = self._load_json(directory / f"rank-{tp_rank}.json")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("snapshot_id") != snapshot_id
            or metadata.get("tp_rank") != tp_rank
            or metadata.get("tp_size") != tp_size
            or _digest(_canonical(metadata))
            != manifest["members"][tp_rank].get("metadata_sha256")
        ):
            raise TokenKVSnapshotError("rank metadata integrity mismatch")
        if _canonical(metadata["identity"]) != _canonical(expected_identity):
            raise TokenKVSnapshotError("model/version/cache layout identity mismatch")
        shared = _canonical([metadata["identity"], metadata["request_metadata"]])
        if _digest(shared) != manifest.get("shared_state_sha256"):
            raise TokenKVSnapshotError("request prefix integrity mismatch")
        if expected_request_metadata is not None and _canonical(
            metadata["request_metadata"]
        ) != _canonical(expected_request_metadata):
            raise TokenKVSnapshotError("request history mismatch")
        return metadata, self._checked_payload(directory, metadata)

    def receipt(self, snapshot_id: str) -> dict[str, Any]:
        """Return a reference to the actual commit, rather than an HTTP status."""
        path = self._directory(snapshot_id) / "COMMITTED.json"
        manifest = self._load_json(path)
        if (
            manifest.get("schema_version") != 1
            or manifest.get("snapshot_id") != snapshot_id
        ):
            raise TokenKVSnapshotError("invalid committed snapshot")
        raw = path.read_bytes()
        return {
            "snapshot_id": snapshot_id,
            "snapshot_committed": True,
            "tp_size": manifest["tp_size"],
            "manifest_ref": {
                "path": str(path),
                "sha256": _digest(raw),
                "bytes": len(raw),
            },
        }

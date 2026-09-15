# SPDX-License-Identifier: Apache-2.0
"""Portable checkpoint content manifests and daemon-verified stat receipts.

The publisher declares the exact consumed files; the daemon hashes and verifies
them once, and clients compare the same manifest plus the daemon's stat receipt.
This assumes trusted, immutable publication, not hostile same-UID modification.
Stat checks alone cannot detect same-size writes within a filesystem timestamp
tick: every checkpoint rewrite MUST republish the content manifest. Readers
must use a stable published generation, not a directory being rewritten in place.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .descriptors import canonical_digest
from .identity import FileStamp, hash_file

MANIFEST_FILENAME = "sglang_weight_cache_manifest.json"


def _paths(root: Path, relative_files: list[str]) -> list[tuple[str, Path]]:
    root = root.resolve(strict=True)
    result = {}
    for name in relative_files:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Checkpoint path must stay relative to its root: {name}")
        normalized = relative.as_posix()
        if normalized == MANIFEST_FILENAME:
            continue
        if normalized in result:
            raise ValueError(f"Duplicate normalized checkpoint path: {normalized}")
        path = root / normalized
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Checkpoint symlink escapes its root: {name}")
        result[normalized] = path
    if not result:
        raise ValueError("A checkpoint manifest requires at least one consumed file")
    return sorted(result.items())


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CheckpointManifest:
    files: tuple[CheckpointFile, ...]
    schema: int = 1

    def validate(self) -> None:
        if type(self.schema) is not int or self.schema != 1 or not self.files:
            raise ValueError("Unsupported/empty checkpoint manifest")
        names = [entry.path for entry in self.files]
        if names != sorted(set(names)) or MANIFEST_FILENAME in names:
            raise ValueError(
                "Checkpoint paths must be sorted, unique and exclude the manifest"
            )
        for entry in self.files:
            path = PurePosixPath(entry.path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.path
                or path.as_posix() != entry.path
                or entry.path == "."
                or type(entry.size) is not int
                or entry.size < 0
                or not re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
            ):
                raise ValueError(f"Invalid checkpoint descriptor: {entry.path}")

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(asdict(self))

    @classmethod
    def read(cls, path: Path) -> CheckpointManifest:
        value = json.loads(path.read_text())
        if set(value) != {"schema", "files"}:
            raise ValueError("Unexpected checkpoint manifest fields")
        result = cls(
            tuple(CheckpointFile(**item) for item in value["files"]), value["schema"]
        )
        result.validate()
        return result


@dataclass(frozen=True)
class VerifiedCheckpoint:
    manifest_digest: str
    stamps: tuple[tuple[str, FileStamp], ...]


def build_manifest(root: Path, relative_files: list[str]) -> CheckpointManifest:
    entries, stamps = [], []
    for name, path in _paths(root, relative_files):
        digest, stamp = hash_file(path)
        entries.append(CheckpointFile(name, stamp.size, digest))
        stamps.append((path, stamp))
    if any(FileStamp.read(path) != stamp for path, stamp in stamps):
        raise ValueError("Checkpoint changed during manifest generation")
    return CheckpointManifest(tuple(entries))


def verify_manifest(
    root: Path, manifest: CheckpointManifest, consumed_files: list[str]
) -> VerifiedCheckpoint:
    """Daemon-only content verification, scoped to the frozen load recipe."""
    manifest.validate()
    paths = _paths(root, consumed_files)
    if [name for name, _ in paths] != [entry.path for entry in manifest.files]:
        raise ValueError(
            "Manifest file set differs from the consumed checkpoint recipe"
        )
    stamps = []
    for (name, path), expected in zip(paths, manifest.files):
        digest, stamp = hash_file(path)
        if digest != expected.sha256 or stamp.size != expected.size:
            raise ValueError(f"Checkpoint content does not match manifest: {name}")
        stamps.append((name, stamp))
    receipt = VerifiedCheckpoint(manifest.digest, tuple(stamps))
    check_verified_stats(root, manifest, receipt)
    return receipt


def check_verified_stats(
    root: Path, manifest: CheckpointManifest, receipt: VerifiedCheckpoint
) -> None:
    """Client fast path: never an unverified stat-only checkpoint identity."""
    if manifest.digest != receipt.manifest_digest:
        raise ValueError("Checkpoint manifest differs from verified producer identity")
    if [name for name, _ in receipt.stamps] != [entry.path for entry in manifest.files]:
        raise ValueError("Verified checkpoint receipt file set differs")
    paths = _paths(root, [entry.path for entry in manifest.files])
    for (name, path), (_, stamp) in zip(paths, receipt.stamps):
        if FileStamp.read(path) != stamp:
            raise ValueError(f"Checkpoint changed since producer verification: {name}")


def write_manifest(root: Path, manifest: CheckpointManifest) -> Path:
    manifest.validate()
    root = root.resolve(strict=True)
    destination = root / MANIFEST_FILENAME
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, prefix=".weight-cache-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(asdict(manifest), stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "files", nargs="+", help="Consumed config/weight paths relative to root"
    )
    args = parser.parse_args()
    manifest = build_manifest(args.root, args.files)
    print(write_manifest(args.root, manifest))
    print(f"sha256:{manifest.digest}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Conservative file identities and short, collision-checked socket locators."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .descriptors import canonical_digest


@dataclass(frozen=True)
class FileStamp:
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int

    @classmethod
    def read(cls, path: Path) -> FileStamp:
        stat = path.stat()
        if not path.is_file():
            raise ValueError(f"Identity input is not a regular file: {path}")
        return cls(
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino
        )


def hash_file(path: Path) -> tuple[str, FileStamp]:
    """Reject an input that changes during hashing; never trust size alone."""
    before = FileStamp.read(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if FileStamp.read(path) != before:
        raise ValueError(f"Identity input changed during hashing: {path}")
    return digest.hexdigest(), before


def source_digest(package_root: Path) -> str:
    """Hash the WHOLE installed sglang Python package, including reused SRT/kernels.

    The caller passes sglang.__file__'s parent, not just multimodal_gen. Native
    dependencies and immutable artifact IDs belong to the deployment stamp.
    """
    root = package_root.resolve(strict=True)
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError(f"No Python source files under {root}")
    entries, stamps = [], []
    for path in files:
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Source symlink escapes package root: {path}")
        digest, stamp = hash_file(path)
        stamps.append((path, stamp))
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size": stamp.size,
            }
        )
    if sorted(root.rglob("*.py")) != files:
        raise ValueError("Package file set changed while computing source identity")
    if any(FileStamp.read(path) != stamp for path, stamp in stamps):
        raise ValueError("Package source changed while computing source identity")
    return canonical_digest({"schema": 1, "files": entries})


def default_runtime_dir() -> Path:
    override = os.environ.get("SGLANG_DIFFUSION_WEIGHT_CACHE_DIR")
    if override:
        return Path(override)
    return (
        Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
        / "sglang_diffusion_weight_cache"
    )


def socket_path(
    device_uuid: str, compatibility_digest: str, *, runtime_dir: Path | None = None
) -> Path:
    """A locator only: peers MUST still compare the full identity on connect.

    The default path is 89 bytes, not the 145-byte UUID/full-digest recipe.
    Directory ownership, locking and ready publication belong to discovery.
    """
    if not device_uuid or not isinstance(device_uuid, str):
        raise ValueError("A physical device UUID is required")
    if not re.fullmatch(r"[0-9a-f]{64}", compatibility_digest):
        raise ValueError("Compatibility digest must be a full SHA-256 hex digest")
    root = runtime_dir if runtime_dir is not None else default_runtime_dir()
    if not root.is_absolute():
        raise ValueError("Weight-cache runtime directory must be absolute")
    device_key = hashlib.sha256(device_uuid.encode()).hexdigest()[:16]
    result = root / device_key / f"{compatibility_digest[:32]}.sock"
    if len(os.fsencode(result)) > 107:
        raise ValueError(
            "Weight-cache Unix socket path exceeds 107 bytes; choose a shorter runtime directory"
        )
    return result

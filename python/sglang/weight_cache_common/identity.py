# SPDX-License-Identifier: Apache-2.0
"""Conservative file identities and short, collision-checked socket locators."""

from __future__ import annotations

import hashlib
import os
import re
import stat
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
    def read(cls, path: Path | str) -> FileStamp:
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Identity input is not a regular file: {path}")
        return cls(
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_dev, info.st_ino
        )


def hash_file(path: Path | str) -> tuple[str, FileStamp]:
    """Reject an input that changes during hashing; never trust size alone."""
    before = FileStamp.read(path)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if FileStamp.read(path) != before:
        raise ValueError(f"Identity input changed during hashing: {path}")
    return digest.hexdigest(), before


def _source_files(root: str) -> list[str]:
    def on_error(error):
        raise error

    files = []
    # Like Path.rglob, do not recurse through directory symlinks. Include
    # directories named *.py as well, so hash_file rejects them as before.
    for parent, directories, names in os.walk(root, onerror=on_error):
        files.extend(
            os.path.join(parent, name)
            for name in (*directories, *names)
            if name.endswith(".py")
        )
    # Preserve pathlib's component-wise order, not whole-path string order.
    return sorted(files, key=lambda name: os.path.normcase(name).split(os.sep))


def source_digest(package_root: Path) -> str:
    """Hash the WHOLE installed sglang Python package, including reused SRT/kernels.

    The caller passes sglang.__file__'s parent, not just multimodal_gen. Native
    dependencies and immutable artifact IDs belong to the deployment stamp.
    """
    root = package_root.resolve(strict=True)
    root_name = os.fspath(root)
    files = _source_files(root_name)
    if not files:
        raise ValueError(f"No Python source files under {root}")
    entries, stamps = [], []
    for path in files:
        # Path.is_relative_to/relative_to repeatedly construct ancestor Paths.
        # For thousands of files this costs more than hashing their contents.
        # Keep strict realpath and the path-component boundary check, but use
        # string-based path operations. This is NOT a stat-only digest cache:
        # every source byte and the complete file set are checked on each call.
        resolved = os.path.realpath(path, strict=True)
        if os.path.commonpath((root_name, resolved)) != root_name:
            raise ValueError(f"Source symlink escapes package root: {path}")
        digest, stamp = hash_file(path)
        stamps.append((path, stamp))
        entries.append(
            {
                "path": os.path.relpath(path, root_name).replace(os.sep, "/"),
                "sha256": digest,
                "size": stamp.size,
            }
        )
    if _source_files(root_name) != files:
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

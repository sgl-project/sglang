#!/usr/bin/env python3
"""Compute a content identity for local inputs to SGLang's Rust extensions."""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path


def _git_build_inputs(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "rust",
            "proto",
        ]
    )
    return [Path(item.decode()) for item in output.rstrip(b"\0").split(b"\0") if item]


def _content_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _entry_digest(relative_path: Path, absolute_path: Path) -> bytes:
    """Return an unambiguously framed identity for one repository path."""
    digest = hashlib.sha256()
    path_bytes = os.fsencode(relative_path)
    digest.update(len(path_bytes).to_bytes(8, "big"))
    digest.update(path_bytes)

    if absolute_path.is_symlink():
        target = os.fsencode(os.readlink(absolute_path))
        digest.update(b"symlink\0")
        digest.update(len(target).to_bytes(8, "big"))
        digest.update(target)
    elif absolute_path.is_file():
        digest.update(b"file\0")
        digest.update(_content_digest(absolute_path))
    else:
        # A tracked path deleted in a dirty checkout must not retain the
        # identity of the version that still contains it.
        digest.update(b"missing\0")
    return digest.digest()


def compute_build_input_digest(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(_git_build_inputs(repo_root), key=os.fspath):
        # Fixed-size entry digests prevent one file's arbitrary bytes from
        # being interpreted as the framing for a subsequent path.
        digest.update(_entry_digest(relative_path, repo_root / relative_path))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(compute_build_input_digest(args.repo_root.resolve()))


if __name__ == "__main__":
    main()

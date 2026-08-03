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


def compute_build_input_digest(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(_git_build_inputs(repo_root), key=os.fspath):
        absolute_path = repo_root / relative_path
        digest.update(os.fsencode(relative_path))
        digest.update(b"\0")
        if absolute_path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(absolute_path)))
        elif absolute_path.is_file():
            digest.update(b"file\0")
            with absolute_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            # A tracked path deleted in a dirty checkout must not retain the
            # identity of the version that still contains it.
            digest.update(b"missing")
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    print(compute_build_input_digest(args.repo_root.resolve()))


if __name__ == "__main__":
    main()

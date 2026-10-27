#!/usr/bin/env python3
"""
Pin sgl-amd-mono in python/pyproject_other.toml at a commit of its repository.

Sets GitHub Actions output variables to indicate if a bump is needed.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SGL_AMD_MONO_REPO = "https://github.com/sgl-project/sgl-amd-mono.git"
PIN_PATTERN = re.compile(r"(sgl-amd-mono\.git@)([0-9a-f]{40})")


def resolve_commit(ref: str) -> str:
    """Resolve a branch of sgl-amd-mono to its head commit; a full SHA passes through."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref

    refs = subprocess.run(
        ["git", "ls-remote", SGL_AMD_MONO_REPO, f"refs/heads/{ref}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not refs:
        print(f"Error: {ref} is neither a full commit SHA nor a branch of sgl-amd-mono")
        sys.exit(1)

    return refs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Pin sgl-amd-mono in python/pyproject_other.toml at a commit"
    )
    parser.add_argument(
        "ref",
        nargs="?",
        default="main",
        help="sgl-amd-mono branch or full commit SHA (default: main)",
    )
    args = parser.parse_args()

    new_commit = resolve_commit(args.ref)

    pyproject_path = Path("python/pyproject_other.toml")
    content = pyproject_path.read_text()
    pins = PIN_PATTERN.findall(content)
    if len(pins) != 1:
        print(
            f"Error: expected one sgl-amd-mono pin in {pyproject_path}, found {len(pins)}"
        )
        sys.exit(1)
    old_commit = pins[0][1]

    needs_bump = old_commit != new_commit
    if needs_bump:
        pyproject_path.write_text(PIN_PATTERN.sub(rf"\g<1>{new_commit}", content))
        print(f"✓ Bumped sgl-amd-mono in {pyproject_path}: {old_commit} → {new_commit}")
    else:
        print(f"✓ {pyproject_path} already pins sgl-amd-mono at {new_commit}")

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"needs_bump={'true' if needs_bump else 'false'}\n")
            f.write(f"old_commit={old_commit}\n")
            f.write(f"new_commit={new_commit}\n")


if __name__ == "__main__":
    main()

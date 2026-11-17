#!/usr/bin/env python3
"""Pin sgl-amd-mono in python/pyproject_other.toml and report needs_bump to GitHub Actions."""

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
        sys.exit(f"Error: {ref} is not a full commit SHA or an sgl-amd-mono branch")

    return refs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ref", nargs="?", default="main", help="sgl-amd-mono branch or full commit SHA"
    )
    args = parser.parse_args()

    new_commit = resolve_commit(args.ref)

    pyproject_path = Path("python/pyproject_other.toml")
    content = pyproject_path.read_text()
    pins = PIN_PATTERN.findall(content)
    if len(pins) != 1:
        sys.exit(f"Error: expected one sgl-amd-mono pin, found {len(pins)}")
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
            f.write(
                f"needs_bump={str(needs_bump).lower()}\n"
                f"old_commit={old_commit}\nnew_commit={new_commit}\n"
            )


if __name__ == "__main__":
    main()

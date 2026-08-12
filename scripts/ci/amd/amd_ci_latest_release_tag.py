#!/usr/bin/env python3
"""Print the highest sglang release tag, read off the remote.

The AMD container scripts need this only to build the rocm/sgl-dev image tag
(`v0.5.17-rocm700-mi30x-<date>`), so they need the tag *name* and nothing the tag
points at. `git ls-remote` transfers refs and no objects, which keeps that cheap
on the shallow CI checkout. Do not go back to `git fetch --tags origin`: that
pulls every branch and tag object, and cost up to an hour per job once the
nightlies had ~90 jobs fetching at once.

Ordering is reused from python/tools/get_version_tag.py rather than
reimplemented here, so the tag this picks is the same one the nightly release
workflow published the image under, including stable/post sorting above rc.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_HELPER_PATH = REPO_ROOT / "python" / "tools" / "get_version_tag.py"
TAG_PATTERN = "v*.*.*"
REF_PREFIX = "refs/tags/"


def load_parse_version_tuple():
    """Import the shared PEP 440 tag ordering without importing the CLI."""
    spec = importlib.util.spec_from_file_location(
        "get_version_tag", VERSION_HELPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_version_tuple


def run_git(*args: str) -> str:
    """Run a git command, returning stripped stdout or "" on failure."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True)
    except OSError as exc:
        print(f"WARNING: failed to run 'git {' '.join(args)}': {exc}", file=sys.stderr)
        return ""
    if result.returncode != 0:
        print(
            f"WARNING: git {' '.join(args)} failed "
            f"(exit {result.returncode}): {result.stderr.strip()}",
            file=sys.stderr,
        )
        return ""
    return result.stdout.strip()


def list_remote_tags(remote: str = "origin") -> list:
    """Return version tag names on `remote`, or [] if it cannot be reached.

    `--refs` drops the peeled `<tag>^{}` entries that annotated tags add.
    """
    raw = run_git("ls-remote", "--tags", "--refs", remote, TAG_PATTERN)
    tags = []
    for line in raw.splitlines():
        _, _, ref = line.partition("\t")
        if ref.startswith(REF_PREFIX):
            tags.append(ref[len(REF_PREFIX) :])
    return tags


def list_local_tags() -> list:
    """Return version tag names already in the local ref store."""
    return run_git("tag", "--list", TAG_PATTERN).splitlines()


def get_latest_release_tag(remote: str = "origin") -> str:
    tags = list_remote_tags(remote) or list_local_tags()
    if not tags:
        return ""
    return sorted(tags, key=load_parse_version_tuple(), reverse=True)[0]


def main() -> None:
    tag = get_latest_release_tag()
    if not tag:
        print(
            f"ERROR: no {TAG_PATTERN} tags found on origin or in the local clone",
            file=sys.stderr,
        )
        sys.exit(1)
    print(tag)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print the highest sglang release tag, read off the remote.

The AMD container scripts need this only to spell the nightly image name
(`rocm/sgl-dev:v0.5.17-rocm720-mi30x-<date>`), so they need the tag *name* and
nothing the tag points at. `git ls-remote` transfers refs and no objects, which
keeps that cheap on the depth-1 checkout `actions/checkout` leaves behind.

Do not go back to `git fetch --tags origin`: on a depth-1 checkout that pulls
tag objects and the history behind them, which no later step in these jobs
reads, and which grew into minutes per job once a nightly had every AMD job
fetching at once.

Ordering comes from scripts/release/get_version_tag.py instead of being
reimplemented here, so this picks the same tag the nightly release workflow
published the image under, including stable and post releases sorting above rc.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION_HELPER_PATH = REPO_ROOT / "scripts" / "release" / "get_version_tag.py"
TAG_PATTERN = "v*.*.*"
REF_PREFIX = "refs/tags/"


def load_parse_version_tuple():
    """Import the shared PEP 440 tag ordering by path, without running its CLI.

    Returns None when the release helper is not in this checkout. That happens on
    branches from before #35196 moved it under scripts/release/, where it is
    still at python/tools/get_version_tag.py -- picking a tag without the shared
    ordering risks naming an image the nightly never published, so callers are
    left with their default instead.
    """
    if not VERSION_HELPER_PATH.is_file():
        print(
            f"WARNING: {VERSION_HELPER_PATH} is missing, so release tags cannot "
            "be ordered the way the nightly image was published",
            file=sys.stderr,
        )
        return None
    spec = importlib.util.spec_from_file_location(
        "get_version_tag", VERSION_HELPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_version_tuple


def run_git(*args: str) -> str:
    """Run a git command, returning stripped stdout, or "" if it failed."""
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

    `--refs` drops the peeled `<tag>^{}` lines that annotated tags also emit.
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
    """Return the highest release tag, or "" so callers can keep a default."""
    parse_version_tuple = load_parse_version_tuple()
    if parse_version_tuple is None:
        return ""
    tags = list_remote_tags(remote) or list_local_tags()
    if not tags:
        print(
            f"WARNING: no {TAG_PATTERN} tags on {remote} or in the local clone",
            file=sys.stderr,
        )
        return ""
    return sorted(tags, key=parse_version_tuple, reverse=True)[0]


def main() -> None:
    tag = get_latest_release_tag()
    if not tag:
        print("ERROR: could not resolve a release tag", file=sys.stderr)
        sys.exit(1)
    print(tag)


if __name__ == "__main__":
    main()

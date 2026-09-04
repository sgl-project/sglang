#!/usr/bin/env python3
"""Check that the rust-ext cache key stays in sync across its sites.

The build workflow looks up and saves cache entries under its own prefix and
hashed inputs; the download action restores with its own. Neither file can
reference the other, and a mismatch in EITHER half makes every pool silently
fall back to source builds at install time.
"""

import re
import sys

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"

_HASH_FILES = re.compile(r"hashFiles\(([^)]*)\)")
_QUOTED = re.compile(r"'([^']*)'")


def hashed_inputs(path: str) -> list[tuple[str, ...]]:
    """The argument tuple of every ``hashFiles(...)`` cache key in a file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [tuple(_QUOTED.findall(args)) for args in _HASH_FILES.findall(text)]


def main() -> int:
    with open(BUILD_WORKFLOW, encoding="utf-8") as f:
        workflow = yaml.safe_load(f)
    with open(DOWNLOAD_ACTION, encoding="utf-8") as f:
        action = yaml.safe_load(f)

    # yaml 1.1 parses the `on:` key as boolean True
    triggers = workflow.get("on", workflow.get(True))
    save_prefix = triggers["workflow_call"]["inputs"]["cache_key_prefix"]["default"]
    restore_prefix = action["inputs"]["cache_key_prefix"]["default"]

    if save_prefix != restore_prefix:
        print("ERROR: rust-ext cache_key_prefix defaults do not match.")
        print(f"  {BUILD_WORKFLOW} saves under:    {save_prefix}")
        print(f"  {DOWNLOAD_ACTION} restores with: {restore_prefix}")
        print("Bump both together, or every pool falls back to source builds.")
        return 1

    # Adding a file to one key alone permanently misses the other's entries.
    sites = [(BUILD_WORKFLOW, inputs) for inputs in hashed_inputs(BUILD_WORKFLOW)]
    sites += [(DOWNLOAD_ACTION, inputs) for inputs in hashed_inputs(DOWNLOAD_ACTION)]

    if not sites:
        print("ERROR: no hashFiles(...) cache key found; this check is dead.")
        return 1

    if len({inputs for _, inputs in sites}) > 1:
        print("ERROR: rust-ext cache key inputs do not match.")
        for path, inputs in sites:
            print(f"  {path}: {list(inputs)}")
        print("Every lookup/save/restore site must hash the same inputs.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

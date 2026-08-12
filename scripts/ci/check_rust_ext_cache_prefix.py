#!/usr/bin/env python3
"""Check that the rust-ext cache_key_prefix defaults stay in sync.

The build workflow saves cache entries under its default; the download action
restores with its own. Neither file can reference the other, and a mismatch
makes every pool silently fall back to source builds at install time.
"""

import sys

import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"


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

    return 0


if __name__ == "__main__":
    sys.exit(main())

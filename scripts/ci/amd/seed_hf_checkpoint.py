#!/usr/bin/env python3
"""Download one checkpoint into the shared AMD CI HuggingFace cache.

Called by reclaim_and_seed_hf_cache.sh, which frees the space first.

This goes through sglang's own `download_weights_from_hf` rather than calling
`snapshot_download` directly, so the seeded cache is what the model loader
accepts. The loader decides a checkpoint is cached by re-running its own
validation over the snapshot, takes the same `.sglang_locks` file, and
re-downloads whatever does not match; a seed that laid the files out even
slightly differently would be silently thrown away by the job it was meant to
serve. Under SGLANG_IS_IN_CI (set by amd_ci_exec.sh) that call is the validating,
retrying CI path, which is exactly the path the consumer job will take.
"""

from __future__ import annotations

import argparse
import logging
import sys

# What LoadFormat.AUTO resolves to in ModelLoader._prepare_weights.
# download_weights_from_hf narrows it to the first pattern the repo matches.
WEIGHT_PATTERNS = ["*.safetensors", "*.bin"]

# Config, tokenizer and remote code: nothing next to the weights, but a consumer
# job that still has to fetch them is not actually warm.
METADATA_PATTERNS = ["*.json", "*.txt", "*.model", "*.py", "*.jinja"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Repo id, e.g. amd/GLM-5.2-MXFP4.")
    parser.add_argument("--revision", default=None, help="Revision to pin.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from huggingface_hub import snapshot_download

    from sglang.srt.model_loader.weight_utils import download_weights_from_hf

    # Metadata first. It is seconds of transfer and it fails on a bad repo id or
    # a gated repo before the multi-hundred-GB download starts.
    snapshot_download(
        args.model, allow_patterns=METADATA_PATTERNS, revision=args.revision
    )

    folder = download_weights_from_hf(args.model, None, WEIGHT_PATTERNS, args.revision)
    print(f"Seeded {args.model} at {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

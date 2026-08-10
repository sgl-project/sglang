#!/usr/bin/env python3
"""Pre-download a suite's models so weight fetching leaves the server-launch path.

Server boot is dominated by weight loading, and on a cold cache most of that is
download: `test_glm_46` has been observed at 818s wall with sglang reporting only
129s of actual loading, the other ~600s spent in snapshot_download. That makes
boot time a function of cache warmth rather than of the model, which in turn
makes any per-test launch timeout a guess -- the same file has run 461s / 557s /
697s across three nightly runs.

Fetching here instead moves that variance into a step of its own, where a slow
or failing download reports as a slow or failing download.

The model list comes from list_stage_models.py (static AST analysis of the
registered tests). Recall is best-effort: a model it misses is simply fetched by
the test as before, so this step is advisory and never fails the job.

Usage:
    python3 scripts/ci/cuda/ensure_model_cache.py --suite nightly-test-8-gpu-h200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import list_stage_models  # noqa: E402


def resolve_models(repo_root: str, backend: str, suite: str) -> List[str]:
    overrides_path = os.path.join(repo_root, "scripts/ci/stage_models_overrides.json")
    with open(overrides_path) as f:
        overrides = json.load(f)
    inventory = list_stage_models.build_inventory(
        repo_root=repo_root, backend_name=backend, overrides=overrides, commit=""
    )
    entry = inventory["suites"].get(suite)
    if entry is None:
        known = ", ".join(sorted(inventory["suites"])[:8])
        print(f"suite {suite!r} not in the inventory (known: {known}, ...)")
        return []
    unresolved = entry.get("unresolved_files") or []
    if unresolved:
        # Not an error: these files fetch their own weights at launch, the way
        # every file did before this step existed. Printed so the recall gap is
        # attributable when a launch is still slow.
        print(f"{len(unresolved)} file(s) contributed no model id:")
        for path in unresolved:
            print(f"    {path}")
    return list(entry.get("models") or [])


def fetch(model: str) -> tuple[bool, float, str]:
    from huggingface_hub import snapshot_download

    start = time.perf_counter()
    try:
        snapshot_download(repo_id=model)
        return True, time.perf_counter() - start, ""
    except Exception as e:  # noqa: BLE001 - report every failure kind the same way
        return False, time.perf_counter() - start, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument(
        "--repo-root", default=os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    models = resolve_models(repo_root, args.backend, args.suite)
    if not models:
        print(f"No models resolved for {args.suite}; nothing to fetch.")
        return 0

    print(f"Fetching {len(models)} model(s) for {args.suite}")
    failed = []
    total = 0.0
    for i, model in enumerate(models, 1):
        ok, elapsed, err = fetch(model)
        total += elapsed
        status = "ok" if ok else "FAILED"
        print(f"  [{i}/{len(models)}] {status:6s} {elapsed:7.1f}s  {model}")
        if not ok:
            print(f"           {err}")
            failed.append(model)
    print(f"Fetched {len(models) - len(failed)}/{len(models)} in {total:.0f}s")
    if failed:
        # Exit 0 regardless: the tests that need these still fetch on launch.
        print(f"Left to the tests: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

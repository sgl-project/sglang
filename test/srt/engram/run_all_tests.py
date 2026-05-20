#!/usr/bin/env python3
"""Run all Engram integration tests in sequence.

Usage (from repo root):
    python test/srt/engram/run_all_tests.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

TESTS = [
    ("Test 1: Import paths", "test_import_paths.py"),
    ("Test 2: Local store + manager", "test_local_store_manager.py"),
    ("Test 3: Engram E2E", "test_engram_e2e.py"),
]


def main() -> None:
    print("=" * 70)
    print("Engram Integration Test Suite")
    print("=" * 70)

    failed = []
    for label, script in TESTS:
        path = BENCH_DIR / script
        print(f"\n{'─' * 70}")
        print(f"Running: {label}  ({script})")
        print(f"{'─' * 70}")
        ret = subprocess.call([sys.executable, str(path)])
        if ret != 0:
            failed.append(label)

    print(f"\n{'=' * 70}")
    if not failed:
        print(f"ALL {len(TESTS)} TESTS PASSED")
    else:
        print(f"FAILED {len(failed)}/{len(TESTS)} TESTS:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

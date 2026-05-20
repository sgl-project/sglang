#!/usr/bin/env python3
"""Test 1: Verify all import paths are correct after restructuring.

Usage:
    python test/srt/engram/test_import_paths.py
"""
from __future__ import annotations

import sys
import traceback


def _check(label: str, fn):
    try:
        fn()
        print(f"  [PASS] {label}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        traceback.print_exc()
        return False


def main() -> None:
    results = []
    print("=" * 60)
    print("Test 1: Import path verification")
    print("=" * 60)

    # --- mem_cache.engram package ---
    print("\n[mem_cache.engram]")
    results.append(
        _check(
            "EngramStore, EngramStoreConfig",
            lambda: __import__(
                "sglang.srt.mem_cache.engram",
                fromlist=["EngramStore", "EngramStoreConfig"],
            ),
        )
    )
    results.append(
        _check(
            "EngramStoreManager + global accessors",
            lambda: __import__(
                "sglang.srt.mem_cache.engram",
                fromlist=[
                    "EngramStoreManager",
                    "get_global_engram_store_manager",
                    "set_global_engram_store_manager",
                ],
            ),
        )
    )
    results.append(
        _check(
            "LocalEngramStore",
            lambda: __import__(
                "sglang.srt.mem_cache.engram.local_engram_store",
                fromlist=["LocalEngramStore"],
            ),
        )
    )
    results.append(
        _check(
            "engram_store_manager module",
            lambda: __import__(
                "sglang.srt.mem_cache.engram.engram_store_manager",
                fromlist=[
                    "EngramStoreManager",
                    "get_global_engram_store_manager",
                    "set_global_engram_store_manager",
                    "close_global_engram_store_manager",
                ],
            ),
        )
    )

    # --- models.engram package ---
    print("\n[models.engram]")
    results.append(
        _check(
            "Engram, backbone_config, engram_cfg",
            lambda: __import__(
                "sglang.srt.models.engram.engram",
                fromlist=["Engram", "backbone_config", "engram_cfg"],
            ),
        )
    )
    results.append(
        _check(
            "EngramConfig, BackBoneConfig",
            lambda: __import__(
                "sglang.srt.models.engram.engram",
                fromlist=["EngramConfig", "BackBoneConfig"],
            ),
        )
    )
    results.append(
        _check(
            "engram module (for engram_mod pattern)",
            lambda: __import__("sglang.srt.models.engram", fromlist=["engram"]),
        )
    )

    # --- cross-package: qwen2.py ---
    print("\n[models.qwen2 -> engram integration]")
    results.append(
        _check(
            "Qwen2MoelEngram class",
            lambda: __import__("sglang.srt.models.qwen2", fromlist=["Qwen2MoelEngram"]),
        )
    )

    # --- summary ---
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    if passed == total:
        print(f"ALL PASSED ({passed}/{total})")
    else:
        print(f"FAILED: {total - passed}/{total} checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

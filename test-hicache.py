#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HiCache Test Suite

Tests various cache scenarios:
1. Basic L1 (GPU) cache hits
2. L1 -> L2 (host) eviction and retrieval
3. Prefix sharing across requests
4. LRU eviction patterns
5. Rapid sequential requests
6. Different prompt lengths
"""

import argparse
import requests
import time
import random
import string

BASE_URL = "http://localhost:30000"
MODEL = "Qwen/Qwen3-0.6B"


def get_metrics():
    """Fetch cache metrics from server."""
    r = requests.get(f"{BASE_URL}/metrics")
    m = {"device": 0, "host": 0}
    for line in r.text.split("\n"):
        if "cached_tokens_total" in line and not line.startswith("#"):
            val = float(line.split()[-1])
            if 'cache_source="device"' in line:
                m["device"] = val
            elif 'cache_source="host"' in line:
                m["host"] = val
    return m


def req(prompt, label, verbose=True):
    """Send a completion request and report cache stats."""
    start = time.time()
    r = requests.post(f"{BASE_URL}/v1/completions", json={
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
        "return_cached_tokens_details": True,
    })
    elapsed = (time.time() - start) * 1000
    d = r.json()
    usage = d.get("usage", {})
    c = d.get("sglext", {}).get("cached_tokens_details", {})

    device, host = c.get('device', 0), c.get('host', 0)
    tokens = usage.get('prompt_tokens', '?')

    if verbose:
        print(f"  {label}:")
        print(f"    tokens={tokens}, device={device}, host={host}, latency={elapsed:.0f}ms")
    return {"device": device, "host": host, "tokens": tokens, "latency": elapsed}


def flood_cache(n=10, tokens_per_req=100, verbose=False):
    """Send flood requests to evict cache entries."""
    for i in range(n):
        prompt = f"FLOOD_{i}_{random.randint(0,9999)} " * tokens_per_req
        requests.post(f"{BASE_URL}/v1/completions", json={
            "model": MODEL, "prompt": prompt, "max_tokens": 1
        })
        if verbose:
            print(f"    Flood request {i+1}/{n}")


def test_basic_l1_l2():
    """Test basic L1 cache hit and L2 retrieval after eviction."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic L1 -> L2 Flow")
    print("=" * 60)

    PREFIX = "The quick brown fox jumps over the lazy dog. " * 15  # ~150 tokens

    print("\n[Phase 1] Cold start - expect miss")
    c1 = req(PREFIX + " Question A", "cold")

    print("\n[Phase 2] Warm - expect L1 (GPU) hit")
    c2 = req(PREFIX + " Question B", "warm")

    print("\n[Phase 3] Flood to evict from GPU")
    flood_cache(n=15, tokens_per_req=100)
    time.sleep(2)  # Allow write-through to complete

    print("\n[Phase 4] After eviction - expect L2 (host) hit")
    c3 = req(PREFIX + " Question C", "after evict")

    success = c2['device'] > 0 and c3['host'] > 0
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_prefix_sharing():
    """Test that multiple prompts with same prefix share cache."""
    print("\n" + "=" * 60)
    print("TEST 2: Prefix Sharing")
    print("=" * 60)

    SHARED_PREFIX = "This is a shared prefix that should be cached. " * 12

    print("\n[Phase 1] First request with prefix")
    c1 = req(SHARED_PREFIX + " Suffix Alpha", "first")

    print("\n[Phase 2] Second request with same prefix, different suffix")
    c2 = req(SHARED_PREFIX + " Suffix Beta", "second (same prefix)")

    print("\n[Phase 3] Third request with same prefix, another suffix")
    c3 = req(SHARED_PREFIX + " Suffix Gamma", "third (same prefix)")

    # All subsequent requests should hit L1 cache for the prefix
    success = c2['device'] > 0 and c3['device'] > 0
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_varying_lengths():
    """Test cache behavior with varying prompt lengths (nested prefixes)."""
    print("\n" + "=" * 60)
    print("TEST 3: Varying Prompt Lengths (Nested Prefixes)")
    print("=" * 60)

    # Use nested prefixes so shorter is prefix of longer
    BASE = "Token " * 150  # ~150 tokens base

    # Test: full -> shorter -> even shorter -> shorter again -> full again
    prompts = [
        (BASE + " suffix_full", "full (150 tokens)"),
        (BASE[:len(BASE)//2] + " suffix_half", "half (~75 tokens)"),
        (BASE[:len(BASE)//3] + " suffix_third", "third (~50 tokens)"),
        (BASE[:len(BASE)//2] + " suffix_half2", "half again"),
        (BASE + " suffix_full2", "full again"),
    ]

    results = []
    for prompt, label in prompts:
        c = req(prompt, label, verbose=True)
        results.append(c)

    # Second half and second full should have cache hits
    success = results[3]['device'] > 0 and results[4]['device'] > 0
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_lru_eviction():
    """Test LRU eviction by accessing items in specific order."""
    print("\n" + "=" * 60)
    print("TEST 4: LRU Eviction Pattern")
    print("=" * 60)

    # Create distinct prefixes
    prefixes = {
        'A': "PREFIX_A " * 40,
        'B': "PREFIX_B " * 40,
        'C': "PREFIX_C " * 40,
    }

    print("\n[Phase 1] Populate cache with A, B, C")
    for name, prefix in prefixes.items():
        req(prefix + f" query_{name}", f"insert {name}")

    print("\n[Phase 2] Access A again (makes A most recently used)")
    req(prefixes['A'] + " query_A2", "re-access A")

    print("\n[Phase 3] Flood to trigger eviction")
    flood_cache(n=12, tokens_per_req=100)
    time.sleep(2)

    print("\n[Phase 4] Check what's still in cache")
    # A should still be in L1 (most recently used)
    # B and C might be evicted to L2
    ca = req(prefixes['A'] + " query_A3", "check A")
    cb = req(prefixes['B'] + " query_B2", "check B")
    cc = req(prefixes['C'] + " query_C2", "check C")

    print(f"\n  A: device={ca['device']}, host={ca['host']}")
    print(f"  B: device={cb['device']}, host={cb['host']}")
    print(f"  C: device={cc['device']}, host={cc['host']}")

    # A should have better cache locality than B and C
    return True  # Informational test


def test_rapid_sequential():
    """Test rapid sequential requests with same prefix."""
    print("\n" + "=" * 60)
    print("TEST 5: Rapid Sequential Requests")
    print("=" * 60)

    PREFIX = "Rapid test prefix for sequential access pattern. " * 10
    suffixes = [f"Query number {i}" for i in range(10)]

    print("\n[Phase 1] Send 10 rapid requests with same prefix")
    results = []
    total_device_hits = 0
    for i, suffix in enumerate(suffixes):
        c = req(PREFIX + suffix, f"req {i+1}/10", verbose=False)
        results.append(c)
        total_device_hits += c['device']
        print(f"    req {i+1}: device={c['device']}, host={c['host']}, latency={c['latency']:.0f}ms")

    # After first request, all others should hit L1 cache
    success = total_device_hits > 0
    print(f"\n  Total L1 hits across 10 requests: {total_device_hits}")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_interleaved_prefixes():
    """Test interleaved access to multiple different prefixes."""
    print("\n" + "=" * 60)
    print("TEST 6: Interleaved Prefix Access")
    print("=" * 60)

    prefixes = {
        'X': "System prompt for assistant X. " * 12,
        'Y': "System prompt for assistant Y. " * 12,
        'Z': "System prompt for assistant Z. " * 12,
    }

    # Interleaved access pattern: X, Y, Z, X, Y, Z, X, Y, Z
    pattern = ['X', 'Y', 'Z'] * 3

    print("\n[Phase 1] Interleaved access pattern")
    results = {k: [] for k in prefixes}
    for i, name in enumerate(pattern):
        c = req(prefixes[name] + f" Turn {i+1}", f"{name} (turn {i+1})", verbose=False)
        results[name].append(c)
        print(f"    {name} turn {i//3 + 1}: device={c['device']}, host={c['host']}")

    # Second and third access to each prefix should hit cache
    for name in prefixes:
        hits = sum(1 for c in results[name][1:] if c['device'] > 0)
        print(f"\n  {name}: {hits}/2 cache hits after first access")

    return True  # Informational test


def test_l2_to_l1_reload():
    """Test that data can be reloaded from L2 back to L1."""
    print("\n" + "=" * 60)
    print("TEST 7: L2 -> L1 Reload Cycle")
    print("=" * 60)

    PREFIX = "This prefix will travel GPU -> Host -> GPU. " * 15

    print("\n[Phase 1] Initial insert into GPU")
    c1 = req(PREFIX + " Initial", "insert")

    print("\n[Phase 2] Verify L1 hit")
    c2 = req(PREFIX + " Verify", "verify L1")

    print("\n[Phase 3] Evict to L2")
    flood_cache(n=15, tokens_per_req=100)
    time.sleep(2)

    print("\n[Phase 4] Access from L2 (should reload to L1)")
    c3 = req(PREFIX + " FromL2", "from L2")

    print("\n[Phase 5] Verify back in L1")
    c4 = req(PREFIX + " BackInL1", "verify back in L1")

    success = c2['device'] > 0 and c3['host'] > 0 and c4['device'] > 0
    print(f"\n  Flow: L1 hit ({c2['device']}) -> L2 hit ({c3['host']}) -> L1 hit ({c4['device']})")
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def run_all_tests(quick=False):
    """Run all tests and report summary."""
    print("=" * 60)
    print("HICACHE COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    m0 = get_metrics()
    print(f"\nInitial metrics: device={m0['device']}, host={m0['host']}")

    tests = [
        ("Basic L1->L2 Flow", test_basic_l1_l2),
        ("Prefix Sharing", test_prefix_sharing),
        ("Varying Lengths", test_varying_lengths),
        ("Rapid Sequential", test_rapid_sequential),
    ]

    if not quick:
        tests.extend([
            ("LRU Eviction", test_lru_eviction),
            ("Interleaved Prefixes", test_interleaved_prefixes),
            ("L2->L1 Reload", test_l2_to_l1_reload),
        ])

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results[name] = False

    m1 = get_metrics()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nMetric deltas:")
    print(f"  device: +{m1['device'] - m0['device']:.0f} tokens")
    print(f"  host:   +{m1['host'] - m0['host']:.0f} tokens")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiCache test suite")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Run quick tests only")
    parser.add_argument("--test", "-t", type=int, default=0,
                        help="Run specific test (1-7), 0 for all")
    args = parser.parse_args()

    if args.test == 1:
        test_basic_l1_l2()
    elif args.test == 2:
        test_prefix_sharing()
    elif args.test == 3:
        test_varying_lengths()
    elif args.test == 4:
        test_lru_eviction()
    elif args.test == 5:
        test_rapid_sequential()
    elif args.test == 6:
        test_interleaved_prefixes()
    elif args.test == 7:
        test_l2_to_l1_reload()
    else:
        run_all_tests(quick=args.quick)

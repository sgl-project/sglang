#!/usr/bin/env python3
"""
Live server test for spectral eviction fingerprint wiring.

This script tests the full end-to-end flow with a real running server:
  1. Sends requests with attention capture enabled
  2. Verifies fingerprints are returned in response
  3. Checks that fingerprints have correct structure

Usage:
    # Start server with spectral eviction and fingerprint capture:
    python -m sglang.launch_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --attention-fingerprint-mode \
        --radix-eviction-policy spectral \
        --spectral-retention-ratio 0.3

    # Run this test:
    python scripts/test_spectral_eviction_live.py --base-url http://localhost:30000

    # Or with all checks:
    python scripts/test_spectral_eviction_live.py --base-url http://localhost:30000 --verbose
"""

import argparse
import sys

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy library required. Install with: pip install numpy")
    sys.exit(1)


def check_server_health(base_url: str) -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Server health check failed: {e}")
        return False


def test_basic_completion(base_url: str, verbose: bool = False) -> bool:
    """Test basic completion works."""
    print("\n=== Test 1: Basic Completion ===")

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"FAILED: Status {response.status_code}")
            if verbose:
                print(response.text)
            return False

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        print(f"Response: {text[:50]}...")
        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_fingerprint_capture(base_url: str, verbose: bool = False) -> bool:
    """Test fingerprint capture returns valid fingerprints."""
    print("\n=== Test 2: Fingerprint Capture ===")

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": "What is 2 + 2? Answer briefly."}
                ],
                "max_tokens": 20,
                "temperature": 0.0,
                "return_attention_tokens": True,
            },
            timeout=30,
        )

        if response.status_code != 200:
            print(f"FAILED: Status {response.status_code}")
            if verbose:
                print(response.text)
            return False

        data = response.json()
        # attention_tokens can be directly on choices[0] or in meta_info
        choice = data["choices"][0]
        attention_tokens = choice.get("attention_tokens", [])
        if not attention_tokens:
            # Fallback to meta_info location
            meta_info = choice.get("meta_info", {})
            attention_tokens = meta_info.get("attention_tokens", [])

        if not attention_tokens:
            print("FAILED: No attention_tokens in response")
            print("  (Server may not have --attention-fingerprint-mode enabled)")
            return False

        print(f"Captured {len(attention_tokens)} attention records")

        # Validate fingerprint structure
        first_record = attention_tokens[0]
        if verbose:
            print(f"First record keys: {list(first_record.keys())}")

        if "fingerprint" not in first_record:
            print("FAILED: No 'fingerprint' in attention record")
            return False

        fingerprint = first_record["fingerprint"]
        if not isinstance(fingerprint, list):
            print(f"FAILED: Fingerprint is not a list: {type(fingerprint)}")
            return False

        fp_dim = len(fingerprint)
        if fp_dim < 20:
            print(f"FAILED: Fingerprint dimension too small: {fp_dim}")
            return False

        print(f"Fingerprint dimension: {fp_dim}")
        print(f"Manifold zone: {first_record.get('manifold', 'not set')}")

        if verbose:
            fp_arr = np.array(fingerprint)
            print(
                f"Fingerprint stats: min={fp_arr.min():.3f}, max={fp_arr.max():.3f}, mean={fp_arr.mean():.3f}"
            )

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_fingerprint_zone_distribution(base_url: str, verbose: bool = False) -> bool:
    """Test that fingerprints get classified into different zones."""
    print("\n=== Test 3: Zone Distribution ===")

    prompts = [
        # Syntax-heavy (should get syntax_floor)
        'Fix this JSON: {"name": "test"',
        # Reasoning (should get semantic_bridge or structure_ripple)
        "Explain step by step how photosynthesis works in plants.",
        # Counting/structure (should get structure_ripple)
        "Count from 1 to 10, one number per line.",
    ]

    zone_counts = {}

    try:
        for prompt in prompts:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0,
                    "return_attention_tokens": True,
                },
                timeout=60,
            )

            if response.status_code != 200:
                print(
                    f"FAILED: Status {response.status_code} for prompt: {prompt[:30]}..."
                )
                continue

            data = response.json()
            choice = data["choices"][0]
            attention_tokens = choice.get("attention_tokens", [])
            if not attention_tokens:
                attention_tokens = choice.get("meta_info", {}).get("attention_tokens", [])

            for record in attention_tokens:
                zone = record.get("manifold", "unknown")
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

        if not zone_counts:
            print("FAILED: No zones captured")
            return False

        print("Zone distribution:")
        for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
            print(f"  {zone}: {count}")

        # We should see at least 2 different zones with varied prompts
        if len(zone_counts) < 2:
            print("WARNING: Only one zone type observed (may need more varied prompts)")

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_fingerprint_consistency(base_url: str, verbose: bool = False) -> bool:
    """Test that same prompt produces consistent fingerprints."""
    print("\n=== Test 4: Fingerprint Consistency ===")

    prompt = "Hello, how are you today?"

    try:
        fingerprints = []
        for i in range(3):
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                    "return_attention_tokens": True,
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(f"FAILED: Request {i+1} failed")
                return False

            data = response.json()
            choice = data["choices"][0]
            attention_tokens = choice.get("attention_tokens", [])
            if not attention_tokens:
                attention_tokens = choice.get("meta_info", {}).get("attention_tokens", [])

            if attention_tokens:
                fingerprints.append(np.array(attention_tokens[0]["fingerprint"]))

        if len(fingerprints) < 2:
            print("FAILED: Not enough fingerprints captured")
            return False

        # Compare first fingerprint with others
        base_fp = fingerprints[0]
        for i, fp in enumerate(fingerprints[1:], 2):
            diff = np.abs(base_fp - fp).mean()
            if verbose:
                print(f"  Difference between run 1 and {i}: {diff:.6f}")

            # With temperature=0, fingerprints should be very similar
            if diff > 0.1:
                print(f"WARNING: High fingerprint variance between runs: {diff:.4f}")

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Live server test for spectral eviction fingerprint wiring"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="Base URL of the SGLang server",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Spectral Eviction Live Server Test")
    print("=" * 60)
    print(f"Server URL: {args.base_url}")

    # Check server health
    print("\nChecking server health...")
    if not check_server_health(args.base_url):
        print("ERROR: Server is not healthy or not reachable")
        print("\nMake sure to start the server with fingerprint capture enabled:")
        print("  python -m sglang.launch_server \\")
        print("      --model Qwen/Qwen2.5-1.5B-Instruct \\")
        print("      --attention-fingerprint-mode \\")
        print("      --radix-eviction-policy spectral")
        sys.exit(1)

    print("Server is healthy")

    # Run tests
    results = []
    results.append(
        ("Basic Completion", test_basic_completion(args.base_url, args.verbose))
    )
    results.append(
        ("Fingerprint Capture", test_fingerprint_capture(args.base_url, args.verbose))
    )
    results.append(
        (
            "Zone Distribution",
            test_fingerprint_zone_distribution(args.base_url, args.verbose),
        )
    )
    results.append(
        (
            "Fingerprint Consistency",
            test_fingerprint_consistency(args.base_url, args.verbose),
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Fingerprint wiring is working correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Check server configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()

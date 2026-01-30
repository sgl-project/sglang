"""
Test script for /v1/kv_cache endpoint.

Usage:
    # Option 1: Run against an existing server
    python test_kv_cache_endpoint.py --base-url http://localhost:30000

    # Option 2: Run as unittest (starts its own server)
    python -m pytest test_kv_cache_endpoint.py -v

Example curl commands:
    curl http://localhost:30000/v1/kv_cache
    curl "http://localhost:30000/v1/kv_cache?include=tree,memory"
    curl "http://localhost:30000/v1/kv_cache?include=evictions&max_evictions=10"
"""

import argparse
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def test_kv_cache_endpoint(base_url: str, verbose: bool = True):
    """Test the /v1/kv_cache endpoint."""
    results = {"passed": 0, "failed": 0, "errors": []}

    def log(msg):
        if verbose:
            print(msg)

    # Test 1: Basic endpoint call
    log("\n=== Test 1: Basic /v1/kv_cache call ===")
    try:
        resp = requests.get(f"{base_url}/v1/kv_cache", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        assert "kv_cache_state" in data, "Missing 'kv_cache_state' in response"
        assert len(data["kv_cache_state"]) > 0, "Empty kv_cache_state"
        state = data["kv_cache_state"][0]
        assert "dp_rank" in state, "Missing 'dp_rank'"
        assert "timestamp" in state, "Missing 'timestamp'"
        log(f"✓ Basic call passed. Got state for dp_rank={state['dp_rank']}")
        results["passed"] += 1
    except Exception as e:
        log(f"✗ Basic call failed: {e}")
        results["failed"] += 1
        results["errors"].append(str(e))

    # Test 2: Include specific sections
    log("\n=== Test 2: Include specific sections (tree,memory) ===")
    try:
        resp = requests.get(
            f"{base_url}/v1/kv_cache", params={"include": "tree,memory"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        state = data["kv_cache_state"][0]
        assert state.get("tree") is not None, "Missing 'tree' when requested"
        assert state.get("memory") is not None, "Missing 'memory' when requested"
        log(f"✓ Tree: total_size={state['tree']['total_size']}")
        log(f"✓ Memory: available_slots={state['memory']['available_slots']}")
        results["passed"] += 1
    except Exception as e:
        log(f"✗ Section filter failed: {e}")
        results["failed"] += 1
        results["errors"].append(str(e))

    # Test 3: Include evictions
    log("\n=== Test 3: Include evictions ===")
    try:
        resp = requests.get(
            f"{base_url}/v1/kv_cache",
            params={"include": "evictions", "max_evictions": 10},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        state = data["kv_cache_state"][0]
        # recent_evictions may be empty if no evictions happened yet
        assert "recent_evictions" in state, "Missing 'recent_evictions' key"
        evictions = state.get("recent_evictions") or []
        log(f"✓ Evictions endpoint works. Found {len(evictions)} recent evictions")
        if evictions:
            log(f"  Latest eviction: {evictions[0]}")
        results["passed"] += 1
    except Exception as e:
        log(f"✗ Evictions test failed: {e}")
        results["failed"] += 1
        results["errors"].append(str(e))

    # Test 4: Generate some requests and check requests info
    log("\n=== Test 4: Check requests info during generation ===")
    try:
        # Start a slow generation in background
        def slow_generate():
            requests.post(
                f"{base_url}/generate",
                json={
                    "text": "Write a very long essay about artificial intelligence.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0.7},
                },
                timeout=60,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_generate)
            time.sleep(0.5)  # Let the request start

            # Check KV cache state while request is running
            resp = requests.get(
                f"{base_url}/v1/kv_cache", params={"include": "requests"}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            state = data["kv_cache_state"][0]
            reqs = state.get("requests") or []
            log(f"✓ Found {len(reqs)} running requests")
            for r in reqs[:3]:  # Show first 3
                log(
                    f"  - {r['request_id']}: state={r.get('state', 'unknown')}, "
                    f"tokens={r['num_tokens']}, prefix_len={r['prefix_len']}"
                )

        results["passed"] += 1
    except Exception as e:
        log(f"✗ Requests info test failed: {e}")
        results["failed"] += 1
        results["errors"].append(str(e))

    # Test 5: Full dump
    log("\n=== Test 5: Full dump (all sections) ===")
    try:
        resp = requests.get(
            f"{base_url}/v1/kv_cache", params={"include": "all"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if verbose:
            log(json.dumps(data, indent=2, default=str)[:2000] + "...")
        results["passed"] += 1
    except Exception as e:
        log(f"✗ Full dump failed: {e}")
        results["failed"] += 1
        results["errors"].append(str(e))

    # Summary
    log("\n" + "=" * 50)
    log(f"Results: {results['passed']} passed, {results['failed']} failed")
    if results["errors"]:
        log("Errors:")
        for e in results["errors"]:
            log(f"  - {e}")

    return results["failed"] == 0


class TestKVCacheEndpoint(unittest.TestCase):
    """Unit test class that starts its own server."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=("--mem-fraction-static", "0.7"),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_kv_cache_basic(self):
        """Test basic /v1/kv_cache call."""
        resp = requests.get(f"{self.base_url}/v1/kv_cache", timeout=10)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("kv_cache_state", data)
        self.assertGreater(len(data["kv_cache_state"]), 0)

    def test_kv_cache_sections(self):
        """Test section filtering."""
        resp = requests.get(
            f"{self.base_url}/v1/kv_cache",
            params={"include": "tree,memory"},
            timeout=10,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        state = data["kv_cache_state"][0]
        self.assertIsNotNone(state.get("tree"))
        self.assertIsNotNone(state.get("memory"))

    def test_kv_cache_evictions(self):
        """Test evictions section."""
        resp = requests.get(
            f"{self.base_url}/v1/kv_cache",
            params={"include": "evictions", "max_evictions": 10},
            timeout=10,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        state = data["kv_cache_state"][0]
        self.assertIn("recent_evictions", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test /v1/kv_cache endpoint")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of running server (e.g., http://localhost:30000)",
    )
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.base_url:
        # Manual test against existing server
        success = test_kv_cache_endpoint(args.base_url, verbose=not args.quiet)
        exit(0 if success else 1)
    else:
        # Run as unittest
        print("No --base-url provided. Running as unittest (will start a server)...")
        print("Tip: Use --base-url http://localhost:30000 to test an existing server")
        unittest.main(argv=[""], exit=True)

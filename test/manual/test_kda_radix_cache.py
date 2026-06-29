"""
Test KDA MambaRadixCache prefix caching.

Prerequisites:
  - Server running with KimiLinear model and extra_buffer strategy:
    python -m sglang.launch_server \
      --model-path <kimi-linear-model-path> \
      --tp 2 --trust-remote-code --port 30000 \
      --mem-fraction-static 0.85 \
      --mamba-scheduler-strategy extra_buffer

Expected result:
  - First request: cached_tokens=0
  - Subsequent requests with shared prefix: cached_tokens > 0 (e.g. 384)
"""

import sys

import requests

SERVER_URL = "http://localhost:30000"


def generate(text, max_tokens=10):
    resp = requests.post(
        f"{SERVER_URL}/generate",
        json={
            "text": text,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
        },
        timeout=180,
    )
    return resp.json()


def check_server():
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    if not check_server():
        print("ERROR: Server not running at", SERVER_URL)
        sys.exit(1)

    print("=" * 70)
    print("  KDA MambaRadixCache Prefix Caching Test")
    print("=" * 70)

    # Long prefix to exceed mamba_track_interval (256 tokens)
    prefix = (
        "You are a helpful AI assistant. In the vast landscape of artificial intelligence, "
        "there are many fascinating topics. From deep learning to NLP, from computer vision "
        "to reinforcement learning. Researchers work on model compression, efficient inference, "
        "multi-modal learning, and robust AI systems. State space models and linear attention "
        "aim to reduce quadratic complexity of transformers. "
    ) * 6  # ~400+ tokens

    # Test 1: Cold request
    print("\n1. Cold request (no cache):")
    r1 = generate(prefix + "What is deep learning?")
    print(
        f"   prompt_tokens={r1['meta_info']['prompt_tokens']}, "
        f"cached_tokens={r1['meta_info']['cached_tokens']}"
    )
    assert (
        r1["meta_info"]["cached_tokens"] == 0
    ), "Expected 0 cached tokens on cold request"

    # Test 2: Shared prefix, different suffix
    print("\n2. Shared prefix, different suffix (should hit cache):")
    r2 = generate(prefix + "What is machine learning?")
    print(
        f"   prompt_tokens={r2['meta_info']['prompt_tokens']}, "
        f"cached_tokens={r2['meta_info']['cached_tokens']}"
    )
    assert (
        r2["meta_info"]["cached_tokens"] > 0
    ), f"Expected cached_tokens > 0, got {r2['meta_info']['cached_tokens']}"

    # Test 3: Another suffix
    print("\n3. Another suffix (should hit cache):")
    r3 = generate(prefix + "Explain neural networks.")
    print(
        f"   prompt_tokens={r3['meta_info']['prompt_tokens']}, "
        f"cached_tokens={r3['meta_info']['cached_tokens']}"
    )
    assert (
        r3["meta_info"]["cached_tokens"] > 0
    ), f"Expected cached_tokens > 0, got {r3['meta_info']['cached_tokens']}"

    # Test 4: Exact same as request 1
    print("\n4. Exact same as request 1 (should hit cache):")
    r4 = generate(prefix + "What is deep learning?")
    print(
        f"   prompt_tokens={r4['meta_info']['prompt_tokens']}, "
        f"cached_tokens={r4['meta_info']['cached_tokens']}"
    )
    assert (
        r4["meta_info"]["cached_tokens"] > 0
    ), f"Expected cached_tokens > 0, got {r4['meta_info']['cached_tokens']}"

    # Verify consistency
    cached_tokens = r2["meta_info"]["cached_tokens"]
    print(f"\n5. Consistency check:")
    print(f"   All cache hits: {cached_tokens} tokens")
    assert (
        r3["meta_info"]["cached_tokens"] == cached_tokens
    ), "Cached tokens should be consistent"

    # Test output quality (basic sanity)
    print(f"\n6. Output sanity check:")
    r_quality = generate(
        prefix + "What is 2+2? Answer with just the number.", max_tokens=5
    )
    print(f"   Output: {r_quality['text']!r}")
    print(f"   cached_tokens={r_quality['meta_info']['cached_tokens']}")

    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED")
    print(
        f"  Prefix caching working: {cached_tokens} tokens cached from ~400 token prefix"
    )
    print("=" * 70)

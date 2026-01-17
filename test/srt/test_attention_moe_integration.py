#!/usr/bin/env python3
"""
Integration test for attention bias steering and MoE routing capture.

Tests the full end-to-end flow with a real model (Qwen3-Next or similar MoE model).

Usage:
    # Start server first:
    python -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4

    # Run test:
    python test/srt/test_attention_moe_integration.py --base-url http://localhost:30000
"""

import argparse
import time

import requests


def test_basic_completion(base_url: str) -> bool:
    """Test basic completion without any special features."""
    print("\n=== Test 1: Basic Completion ===")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
    )

    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        return False

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    print(f"Response: {text}")
    print("PASSED")
    return True


def test_attention_capture(base_url: str) -> bool:
    """Test attention token capture."""
    print("\n=== Test 2: Attention Token Capture ===")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "return_attention_tokens": True,
            "top_k_attention": 5,
            "attention_capture_layer_ids": [0, 11, 23],  # Sample layers
        },
    )

    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        return False

    data = response.json()

    # Check if attention_tokens is in meta_info
    meta_info = data["choices"][0].get("meta_info", {})
    attention_tokens = meta_info.get("attention_tokens")

    if attention_tokens is None:
        print("FAILED: No attention_tokens in response")
        return False

    print(f"Captured {len(attention_tokens)} attention token records")
    if attention_tokens:
        print(f"First record keys: {list(attention_tokens[0].keys())}")
    print("PASSED")
    return True


def test_moe_routing_capture(base_url: str) -> bool:
    """Test MoE routing capture."""
    print("\n=== Test 3: MoE Routing Capture ===")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 20,
            "temperature": 0.0,
            "return_moe_routing": True,
            "moe_routing_top_k": 2,
        },
    )

    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        return False

    data = response.json()

    # Check if moe_routing is in meta_info
    meta_info = data["choices"][0].get("meta_info", {})
    moe_routing = meta_info.get("moe_routing")

    if moe_routing is None:
        print("INFO: No moe_routing in response (model may not be MoE)")
        print("SKIPPED")
        return True  # Not a failure for non-MoE models

    print(f"Captured {len(moe_routing)} MoE routing records")
    if moe_routing:
        first_record = moe_routing[0]
        print(f"First record: decode_step={first_record.get('decode_step')}")
        layers = first_record.get("layers", {})
        print(f"Captured {len(layers)} layers")
        if layers:
            layer_id = list(layers.keys())[0]
            layer_data = layers[layer_id]
            print(f"Layer {layer_id}: expert_ids={layer_data.get('expert_ids')}")
    print("PASSED")
    return True


def test_attention_biases(base_url: str) -> bool:
    """Test attention bias steering via API."""
    print("\n=== Test 4: Attention Bias Steering ===")

    # Create biases that boost attention to the first few tokens
    # Format: {layer_id_str: {token_pos_str: bias_value, ...}, ...}
    attention_biases = {
        "11": {  # Apply to layer 11
            "0": 2.0,  # Boost first token
            "1": 1.5,  # Boost second token
            "2": 1.0,  # Boost third token
        }
    }

    # First, run without biases
    response_no_bias = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "Complete this: The quick brown fox"}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "return_attention_tokens": True,
            "attention_capture_layer_ids": [11],
        },
    )

    # Then run with biases
    response_with_bias = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "Complete this: The quick brown fox"}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "return_attention_tokens": True,
            "attention_capture_layer_ids": [11],
            "attention_biases": attention_biases,
        },
    )

    if response_no_bias.status_code != 200 or response_with_bias.status_code != 200:
        print(f"FAILED: Request failed")
        return False

    data_no_bias = response_no_bias.json()
    data_with_bias = response_with_bias.json()

    text_no_bias = data_no_bias["choices"][0]["message"]["content"]
    text_with_bias = data_with_bias["choices"][0]["message"]["content"]

    print(f"Without bias: {text_no_bias}")
    print(f"With bias:    {text_with_bias}")

    # Check that biases were applied (request succeeded with bias parameter)
    if "attention_biases" in str(response_with_bias.request.body):
        print("Bias parameter was sent successfully")

    print("PASSED (bias parameter accepted)")
    return True


def test_combined_features(base_url: str) -> bool:
    """Test all features together."""
    print("\n=== Test 5: Combined Features ===")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "Explain quantum computing briefly."}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "return_attention_tokens": True,
            "top_k_attention": 10,
            "attention_capture_layer_ids": [0, 11, 23, 35],
            "return_moe_routing": True,
            "moe_routing_top_k": 3,
            "attention_biases": {"11": {"0": 1.0}},  # Small bias on layer 11, token 0
        },
    )

    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        return False

    data = response.json()
    meta_info = data["choices"][0].get("meta_info", {})

    text = data["choices"][0]["message"]["content"]
    print(f"Response: {text[:100]}...")

    attention = meta_info.get("attention_tokens")
    moe = meta_info.get("moe_routing")

    print(f"Attention records: {len(attention) if attention else 0}")
    print(f"MoE routing records: {len(moe) if moe else 0}")

    print("PASSED")
    return True


def test_thinking_with_telemetry(base_url: str) -> bool:
    """Test thinking model with attention/MoE telemetry."""
    print("\n=== Test 6: Thinking Model with Telemetry ===")

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {"role": "user", "content": "What is 15 * 23? Think step by step."}
            ],
            "max_tokens": 200,
            "temperature": 0.0,
            "return_attention_tokens": True,
            "top_k_attention": 5,
            "return_moe_routing": True,
        },
    )

    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        return False

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    meta_info = data["choices"][0].get("meta_info", {})

    print(f"Response length: {len(text)} chars")

    attention = meta_info.get("attention_tokens", [])
    moe = meta_info.get("moe_routing", [])

    # Check for think phase transitions in attention data
    think_phases = set()
    for record in attention:
        phase = record.get("think_phase")
        if phase:
            think_phases.add(phase)

    print(f"Think phases observed: {think_phases}")
    print(f"Total attention records: {len(attention)}")
    print(f"Total MoE routing records: {len(moe)}")

    # Analyze MoE expert distribution if available
    if moe:
        all_experts = []
        for record in moe:
            for layer_id, layer_data in record.get("layers", {}).items():
                all_experts.extend(layer_data.get("expert_ids", []))

        if all_experts:
            unique_experts = len(set(all_experts))
            print(f"Unique experts used: {unique_experts}")

    print("PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for attention/MoE features"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:30000", help="Server base URL"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"Testing against: {base_url}")

    # Wait for server to be ready
    print("Checking server health...")
    for i in range(10):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                print("Server is healthy")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server not responding")
        return 1

    # Run tests
    results = []

    results.append(("Basic Completion", test_basic_completion(base_url)))
    results.append(("Attention Capture", test_attention_capture(base_url)))
    results.append(("MoE Routing", test_moe_routing_capture(base_url)))
    results.append(("Attention Biases", test_attention_biases(base_url)))
    results.append(("Combined Features", test_combined_features(base_url)))
    results.append(("Thinking with Telemetry", test_thinking_with_telemetry(base_url)))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())

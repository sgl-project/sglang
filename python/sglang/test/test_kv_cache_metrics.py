#!/usr/bin/env python3
"""
Test script to verify KV cache utilization metrics are properly exposed via /metrics endpoint.
"""

import requests
import time

BASE_URL = "http://localhost:8080"


def test_metrics_endpoint():
    """Test that /metrics endpoint returns KV cache utilization metrics."""
    print("Testing /metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        response.raise_for_status()
        
        metrics_content = response.text
        print("\n=== Metrics Response ===")
        print(metrics_content[:1000] + "..." if len(metrics_content) > 1000 else metrics_content)
        print("\n=== End of Metrics Response ===")
        
        # Check for KV cache utilization metrics
        kv_cache_metrics = [
            "sglang:kv_cache_avg_active_ratio",
            "sglang:kv_cache_avg_access_frequency",
            "sglang:kv_cache_total_nodes",
            "sglang:kv_cache_total_slots",
            "sglang:kv_cache_eviction_total",
            "sglang:kv_cache_evicted_tokens_total"
        ]
        
        print("\n=== Checking for KV Cache Metrics ===")
        found_metrics = []
        for metric in kv_cache_metrics:
            if metric in metrics_content:
                found_metrics.append(metric)
                print(f"✓ Found: {metric}")
            else:
                print(f"✗ Missing: {metric}")
        
        if found_metrics:
            print(f"\n✓ Success: Found {len(found_metrics)} out of {len(kv_cache_metrics)} KV cache metrics")
        else:
            print("\n✗ Error: No KV cache metrics found in /metrics response")
            
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error: Could not reach /metrics endpoint: {e}")
        print("Please make sure the SGLang server is running on localhost:8080")


def test_kv_cache_endpoint():
    """Test that /v1/kv_cache endpoint returns utilization metrics."""
    print("\nTesting /v1/kv_cache endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/v1/kv_cache", timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print("\n=== KV Cache State Response ===")
        print(data)
        print("\n=== End of KV Cache State Response ===")
        
        # Check for utilization metrics
        if "kv_cache_state" in data and data["kv_cache_state"]:
            state = data["kv_cache_state"][0]
            if "utilization" in state:
                utilization = state["utilization"]
                print("\n=== Utilization Metrics ===")
                print(utilization)
                print("\n✓ Success: Found utilization metrics in /v1/kv_cache response")
            else:
                print("\n✗ Error: Missing utilization metrics in /v1/kv_cache response")
        else:
            print("\n✗ Error: Invalid /v1/kv_cache response format")
            
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Error: Could not reach /v1/kv_cache endpoint: {e}")
        print("Please make sure the SGLang server is running on localhost:8080")


if __name__ == "__main__":
    print("Testing KV Cache Utilization Metrics")
    print("====================================")
    print(f"Base URL: {BASE_URL}")
    print()
    
    test_metrics_endpoint()
    test_kv_cache_endpoint()

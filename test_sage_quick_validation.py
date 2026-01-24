#!/usr/bin/env python3
"""
Quick SageAttention Validation Tests

Fast tests that can run against a single running server to validate:
1. Logits extraction and comparison readiness
2. Throughput measurements
3. Memory baseline

Run against existing server:
python3 test_sage_quick_validation.py --port 30000
"""

import argparse
import time
import json
import requests
import numpy as np
from typing import Dict, List


def test_logits_ready(base_url: str, model: str):
    """Test if we can extract logits for comparison"""
    print("\n" + "="*70)
    print("TEST 1: Logits Extraction Readiness")
    print("="*70)
    
    prompts = [
        ("short", "The capital of France is", 10),
        ("medium", "In a galaxy far away", 20),
    ]
    
    results = []
    
    for context_type, prompt, max_tokens in prompts:
        print(f"\n📝 Testing {context_type} context...")
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": 5,
            },
            timeout=30,
        )
        
        data = response.json()
        output_text = data['choices'][0]['text']
        tokens_used = data['usage']['completion_tokens']
        
        # Check if we have logprobs
        has_logprobs = 'logprobs' in data['choices'][0]
        
        result = {
            "context": context_type,
            "prompt_words": len(prompt.split()),
            "completion_tokens": tokens_used,
            "output_sample": output_text[:60],
            "has_logprobs": has_logprobs,
        }
        
        results.append(result)
        
        print(f"  ✅ Generated {tokens_used} tokens")
        print(f"  Output: {output_text[:60]}...")
        print(f"  Logprobs available: {has_logprobs}")
    
    return results


def test_throughput_various_lengths(base_url: str, model: str):
    """Test throughput for various input lengths"""
    print("\n" + "="*70)
    print("TEST 2: Throughput Measurements")
    print("="*70)
    
    test_configs = [
        (50, 50, "small"),
        (200, 100, "medium"),
        (500, 100, "large"),
    ]
    
    results = []
    
    for input_words, output_tokens, size_label in test_configs:
        print(f"\n📊 Testing {size_label} (input={input_words} words, output={output_tokens} tokens)...")
        
        prompt = " ".join(["test"] * input_words)
        
        # Warmup
        requests.post(
            f"{base_url}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 10, "temperature": 0},
            timeout=30,
        )
        
        # Measure
        latencies = []
        token_counts = []
        
        for i in range(3):
            start = time.time()
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0.5,
                },
                timeout=60,
            )
            latency = time.time() - start
            
            data = response.json()
            tokens = data['usage']['completion_tokens']
            
            latencies.append(latency)
            token_counts.append(tokens)
            
            print(f"    Run {i+1}: {latency:.2f}s, {tokens} tokens, {tokens/latency:.1f} tok/s")
        
        avg_latency = np.mean(latencies)
        avg_tokens = np.mean(token_counts)
        throughput = avg_tokens / avg_latency
        
        result = {
            "size": size_label,
            "input_words": input_words,
            "target_output_tokens": output_tokens,
            "avg_latency_sec": round(avg_latency, 3),
            "avg_tokens": round(avg_tokens, 1),
            "throughput_tokens_per_sec": round(throughput, 2),
        }
        
        results.append(result)
        
        print(f"  ✅ Average: {avg_latency:.2f}s, {throughput:.1f} tokens/s")
    
    return results


def test_consistency(base_url: str, model: str):
    """Test output consistency with temperature=0"""
    print("\n" + "="*70)
    print("TEST 3: Output Consistency (temperature=0)")
    print("="*70)
    
    prompt = "The three laws of robotics are:"
    max_tokens = 50
    num_runs = 3
    
    print(f"\n🔄 Running {num_runs} identical requests...")
    
    outputs = []
    
    for i in range(num_runs):
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=30,
        )
        
        data = response.json()
        output_text = data['choices'][0]['text']
        outputs.append(output_text)
        
        print(f"  Run {i+1}: {output_text[:50]}...")
    
    # Check if all outputs are identical
    all_identical = all(out == outputs[0] for out in outputs)
    
    result = {
        "num_runs": num_runs,
        "all_identical": all_identical,
        "sample_output": outputs[0][:100],
    }
    
    if all_identical:
        print(f"  ✅ All outputs identical (deterministic)")
    else:
        print(f"  ⚠️  Outputs differ (expected for some backends)")
        for i, out in enumerate(outputs):
            if out != outputs[0]:
                print(f"    Run {i+1} differs at: {_find_first_diff(outputs[0], out)}")
    
    return result


def _find_first_diff(s1: str, s2: str) -> str:
    """Find first difference between two strings"""
    for i, (c1, c2) in enumerate(zip(s1, s2)):
        if c1 != c2:
            return f"position {i}: '{s1[max(0,i-10):i+10]}' vs '{s2[max(0,i-10):i+10]}'"
    return "different lengths"


def generate_report(results: Dict, output_file: str):
    """Generate and save report"""
    print("\n" + "="*70)
    print("📊 REPORT SUMMARY")
    print("="*70)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Full results saved to: {output_file}")
    
    # Print summary
    if 'throughput' in results:
        print("\n📈 Throughput Summary:")
        for item in results['throughput']:
            print(f"  {item['size'].capitalize()}: {item['throughput_tokens_per_sec']} tokens/s "
                  f"(latency: {item['avg_latency_sec']}s)")
    
    if 'consistency' in results:
        print(f"\n🔄 Consistency: {'✅ Deterministic' if results['consistency']['all_identical'] else '⚠️  Variable'}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Quick SageAttention validation tests")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model name")
    parser.add_argument("--output", type=str, default="/root/sglang_sage_quick_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print("\n" + "="*70)
    print("  SageAttention Quick Validation Suite")
    print(f"  Server: {base_url}")
    print(f"  Model: {args.model}")
    print("="*70)
    
    # Check server health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"\n✅ Server is healthy")
    except Exception as e:
        print(f"\n❌ Server not reachable: {e}")
        return 1
    
    results = {}
    
    try:
        # Run tests
        results['logits'] = test_logits_ready(base_url, args.model)
        results['throughput'] = test_throughput_various_lengths(base_url, args.model)
        results['consistency'] = test_consistency(base_url, args.model)
        
        # Generate report
        generate_report(results, args.output)
        
        print("\n✅ All validation tests complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


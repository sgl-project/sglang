#!/usr/bin/env python3
"""
Test script to validate DeepSeek-V3 EP accuracy fixes.
This script tests the fix for the accuracy drop when EP is enabled.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_gsm8k_test(model_path, server_args, env_vars=None, num_questions=50):
    """Run a GSM8K test with given configuration."""
    
    # Set up environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    # Start server
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model_path,
        "--trust-remote-code",
        "--mem-fraction-static", "0.8"
    ] + server_args
    
    print(f"Starting server: {' '.join(server_cmd)}")
    if env_vars:
        print(f"Environment: {env_vars}")
    
    server_process = subprocess.Popen(server_cmd, env=env)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(30)
    
    try:
        # Run benchmark
        benchmark_cmd = [
            sys.executable, "benchmark/gsm8k/bench_sglang.py",
            "--num-questions", str(num_questions)
        ]
        
        print(f"Running benchmark: {' '.join(benchmark_cmd)}")
        result = subprocess.run(benchmark_cmd, capture_output=True, text=True, timeout=600, env=env)
        
        if result.returncode == 0:
            # Parse accuracy from output
            for line in result.stdout.split('\n'):
                if line.startswith('Accuracy:'):
                    accuracy = float(line.split(':')[1].strip())
                    print(f"Accuracy: {accuracy:.3f}")
                    return accuracy
            print("Could not parse accuracy from output")
            return None
        else:
            print(f"Benchmark failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None
    finally:
        # Kill server
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek-V3 EP accuracy fix")
    parser.add_argument("--model-path", required=True, help="Path to DeepSeek-V3 model")
    parser.add_argument("--num-questions", type=int, default=50, help="Number of GSM8K questions to test")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer configurations")
    args = parser.parse_args()
    
    print("DeepSeek-V3 EP Accuracy Fix Test")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Questions: {args.num_questions}")
    print()
    
    # Test configurations
    if args.quick:
        configs = [
            {
                "name": "TP-8 Baseline",
                "server_args": ["--tp", "8"],
                "env_vars": {}
            },
            {
                "name": "EP with FP8 (original issue)",
                "server_args": ["--tp", "8", "--enable-ep-moe"],
                "env_vars": {}
            },
            {
                "name": "EP without FP8 (FIXED)",
                "server_args": ["--tp", "8", "--enable-ep-moe"],
                "env_vars": {"SGL_DISABLE_EP_FP8": "true"}
            }
        ]
    else:
        configs = [
            {
                "name": "TP-8 Baseline",
                "server_args": ["--tp", "8"],
                "env_vars": {}
            },
            {
                "name": "EP with FP8 (original issue)",
                "server_args": ["--tp", "8", "--enable-ep-moe"],
                "env_vars": {}
            },
            {
                "name": "EP without FP8 (FIXED)",
                "server_args": ["--tp", "8", "--enable-ep-moe"],
                "env_vars": {"SGL_DISABLE_EP_FP8": "true"}
            },
            {
                "name": "EP + DeepEP normal (no FP8)",
                "server_args": ["--tp", "8", "--enable-ep-moe", "--enable-deepep-moe", "--deepep-mode", "normal"],
                "env_vars": {"SGL_DISABLE_EP_FP8": "true"}
            },
            {
                "name": "EP + FlashInfer (no FP8)",
                "server_args": ["--tp", "8", "--enable-ep-moe", "--enable-flashinfer-moe"],
                "env_vars": {"SGL_DISABLE_EP_FP8": "true"}
            }
        ]
    
    results = {}
    
    # Run tests
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")
        print("-" * 60)
        
        accuracy = run_gsm8k_test(
            args.model_path,
            config["server_args"],
            config["env_vars"],
            args.num_questions
        )
        
        results[config["name"]] = accuracy
        
        if accuracy is not None:
            print(f"✓ {config['name']}: {accuracy:.3f}")
        else:
            print(f"✗ {config['name']}: FAILED")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    baseline_accuracy = results.get("TP-8 Baseline")
    ep_original_accuracy = results.get("EP with FP8 (original issue)")
    ep_fixed_accuracy = results.get("EP without FP8 (FIXED)")
    
    for name, accuracy in results.items():
        if accuracy is not None:
            if baseline_accuracy and name != "TP-8 Baseline":
                diff = accuracy - baseline_accuracy
                print(f"{name:35}: {accuracy:.3f} ({diff:+.3f})")
            else:
                print(f"{name:35}: {accuracy:.3f}")
        else:
            print(f"{name:35}: FAILED")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if baseline_accuracy and ep_original_accuracy and ep_fixed_accuracy:
        original_drop = baseline_accuracy - ep_original_accuracy
        fixed_drop = baseline_accuracy - ep_fixed_accuracy
        
        print(f"Baseline accuracy (TP-8):     {baseline_accuracy:.3f}")
        print(f"EP original accuracy:         {ep_original_accuracy:.3f} (drop: {original_drop:.3f})")
        print(f"EP fixed accuracy:            {ep_fixed_accuracy:.3f} (drop: {fixed_drop:.3f})")
        
        if abs(fixed_drop) < 0.02:  # Within 2%
            print("\n✅ SUCCESS: EP accuracy issue has been RESOLVED!")
            print(f"   The fix reduced accuracy drop from {original_drop:.3f} to {fixed_drop:.3f}")
        elif fixed_drop < original_drop:
            print(f"\n🔄 PARTIAL SUCCESS: Accuracy improved but still has some drop")
            print(f"   Improvement: {original_drop - fixed_drop:.3f}")
        else:
            print(f"\n❌ FAILURE: Fix did not improve accuracy")
    else:
        print("❓ Unable to analyze results due to test failures")
    
    # Save results
    results_file = "ep_fix_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "num_questions": args.num_questions,
            "results": results,
            "analysis": {
                "baseline_accuracy": baseline_accuracy,
                "ep_original_accuracy": ep_original_accuracy,
                "ep_fixed_accuracy": ep_fixed_accuracy,
                "original_drop": baseline_accuracy - ep_original_accuracy if baseline_accuracy and ep_original_accuracy else None,
                "fixed_drop": baseline_accuracy - ep_fixed_accuracy if baseline_accuracy and ep_fixed_accuracy else None
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Debug script for DeepSeek-V3 EP accuracy issues.
This script helps identify the root cause of accuracy drop when EP is enabled.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_gsm8k_benchmark(model_path, server_args, output_file):
    """Run GSM8K benchmark with given server configuration."""
    print(f"Starting server with args: {' '.join(server_args)}")
    
    # Start server
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model_path,
        "--trust-remote-code",
        "--mem-fraction-static", "0.8"
    ] + server_args
    
    print(f"Server command: {' '.join(server_cmd)}")
    server_process = subprocess.Popen(server_cmd)
    
    # Wait for server to start
    time.sleep(30)
    
    try:
        # Run benchmark
        benchmark_cmd = [
            sys.executable, "benchmark/gsm8k/bench_sglang.py",
            "--num-questions", "200",
            "--result-file", output_file
        ]
        
        print(f"Running benchmark: {' '.join(benchmark_cmd)}")
        result = subprocess.run(benchmark_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("Benchmark completed successfully")
            print("STDOUT:", result.stdout)
            return result.stdout
        else:
            print("Benchmark failed")
            print("STDERR:", result.stderr)
            return None
            
    finally:
        # Kill server
        server_process.terminate()
        server_process.wait()


def test_configuration(model_path, config_name, server_args, results):
    """Test a specific configuration and record results."""
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config_name}")
    print(f"{'='*60}")
    
    output_file = f"gsm8k_results_{config_name.replace(' ', '_').lower()}.jsonl"
    
    try:
        output = run_gsm8k_benchmark(model_path, server_args, output_file)
        if output:
            # Parse accuracy from output
            for line in output.split('\n'):
                if line.startswith('Accuracy:'):
                    accuracy = float(line.split(':')[1].strip())
                    results[config_name] = {
                        'accuracy': accuracy,
                        'server_args': server_args,
                        'output_file': output_file
                    }
                    print(f"Configuration '{config_name}' achieved accuracy: {accuracy:.3f}")
                    break
        else:
            results[config_name] = {
                'accuracy': None,
                'server_args': server_args,
                'error': 'Benchmark failed'
            }
            
    except Exception as e:
        print(f"Error testing configuration '{config_name}': {e}")
        results[config_name] = {
            'accuracy': None,
            'server_args': server_args,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Debug DeepSeek-V3 EP accuracy issues")
    parser.add_argument("--model-path", required=True, help="Path to DeepSeek-V3 model")
    parser.add_argument("--output-dir", default="debug_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    results = {}
    
    # Test configurations
    configurations = [
        # Baseline TP configuration
        ("TP-8 Baseline", ["--tp", "8"]),
        
        # Basic EP configuration (reproduces the issue)
        ("EP Basic", ["--tp", "8", "--enable-ep-moe"]),
        
        # EP with different DeepEP modes
        ("EP Normal Mode", ["--tp", "8", "--enable-ep-moe", "--enable-deepep-moe", "--deepep-mode", "normal"]),
        ("EP Low Latency Mode", ["--tp", "8", "--enable-ep-moe", "--enable-deepep-moe", "--deepep-mode", "low_latency"]),
        
        # EP with FlashInfer MoE
        ("EP FlashInfer", ["--tp", "8", "--enable-ep-moe", "--enable-flashinfer-moe"]),
        
        # EP with redundant experts
        ("EP Redundant Experts", ["--tp", "8", "--enable-ep-moe", "--ep-num-redundant-experts", "1"]),
        
        # EP with different dispatch algorithms
        ("EP Static Dispatch", ["--tp", "8", "--enable-ep-moe", "--ep-dispatch-algorithm", "static"]),
        ("EP Dynamic Dispatch", ["--tp", "8", "--enable-ep-moe", "--ep-dispatch-algorithm", "dynamic"]),
        
        # EP without CUDA graph (potential synchronization fix)
        ("EP No CUDA Graph", ["--tp", "8", "--enable-ep-moe", "--disable-cuda-graph"]),
        
        # EP with expert distribution metrics
        ("EP With Metrics", ["--tp", "8", "--enable-ep-moe", "--enable-expert-distribution-metrics"]),
    ]
    
    # Run all configurations
    for config_name, server_args in configurations:
        test_configuration(args.model_path, config_name, server_args, results)
    
    # Save results
    results_file = "debug_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    for config_name, result in results.items():
        accuracy = result.get('accuracy')
        if accuracy is not None:
            print(f"{config_name:25}: {accuracy:.3f}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{config_name:25}: FAILED ({error})")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Identify best EP configuration
    ep_results = {k: v for k, v in results.items() if k != "TP-8 Baseline" and v.get('accuracy') is not None}
    if ep_results:
        best_ep = max(ep_results.items(), key=lambda x: x[1]['accuracy'])
        baseline_accuracy = results.get("TP-8 Baseline", {}).get('accuracy')
        
        print(f"\nBest EP configuration: {best_ep[0]} (accuracy: {best_ep[1]['accuracy']:.3f})")
        if baseline_accuracy:
            accuracy_diff = best_ep[1]['accuracy'] - baseline_accuracy
            print(f"Accuracy difference from baseline: {accuracy_diff:+.3f}")
            
            if abs(accuracy_diff) < 0.01:
                print("✅ EP accuracy issue appears to be resolved!")
            else:
                print("❌ EP accuracy issue persists")


if __name__ == "__main__":
    main()

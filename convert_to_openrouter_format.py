#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

def extract_metrics_from_output(output: str) -> Dict[str, float]:
    metrics = {}
    
    patterns = {
        'mean_latency': r'Mean E2E Latency \(ms\):\s+([\d.]+)',
        'median_latency': r'Median E2E Latency \(ms\):\s+([\d.]+)',
        'total_throughput': r'Total token throughput \(tok/s\):\s+([\d.]+)',
        'output_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'input_throughput': r'Input token throughput \(tok/s\):\s+([\d.]+)',
        'ttft_mean': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'total_tokens': r'Total generated tokens:\s+(\d+)',
        'duration': r'Benchmark duration \(s\):\s+([\d.]+)',
        'num_requests': r'Successful requests:\s+(\d+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics

def calculate_openrouter_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_latency = None
    baseline_throughput = None
    baseline_ttft = None
    
    # Look for the baseline test (single concurrency) to match OpenRouter methodology
    for result in results:
        if not result.get('success', False):
            continue
            
        config = result.get('config', {})
        # Use the single-user baseline test (max_concurrency = 1)
        if config.get('max_concurrency') == 1:
            metrics = extract_metrics_from_output(result.get('output', ''))
            
            if 'mean_latency' in metrics:
                baseline_latency = metrics['mean_latency'] / 1000.0
            
            if 'output_throughput' in metrics:
                baseline_throughput = metrics['output_throughput']
                
            if 'ttft_mean' in metrics:
                baseline_ttft = metrics['ttft_mean']
            break
    
    # If no baseline found, fall back to first successful result
    if baseline_latency is None:
        for result in results:
            if result.get('success', False):
                metrics = extract_metrics_from_output(result.get('output', ''))
                if 'mean_latency' in metrics:
                    baseline_latency = metrics['mean_latency'] / 1000.0
                if 'output_throughput' in metrics:
                    baseline_throughput = metrics['output_throughput']
                if 'ttft_mean' in metrics:
                    baseline_ttft = metrics['ttft_mean']
                break
    
    if baseline_latency is None or baseline_throughput is None:
        return {}
    
    return {
        'latency_seconds': baseline_latency,
        'throughput_tps': baseline_throughput,
        'ttft_ms': baseline_ttft or 0,
        'uptime_percent': 100.0,
    }

def print_openrouter_format(metrics: Dict[str, Any], model_name: str = "Qwen/Qwen3-30B-A3B"):
    print(f"\nAndre Inference")
    print("=" * 40)
    print(f"Latency: {metrics.get('latency_seconds', 0):.2f}s")
    print(f"Throughput: {metrics.get('throughput_tps', 0):.1f}tps") 
    print(f"TTFT: {metrics.get('ttft_ms', 0):.1f}ms")
    print(f"Context: 32K tokens")

def compare_with_openrouter_examples(your_metrics: Dict[str, Any]):
    examples = [
        {"name": "Chutes", "latency": 1.92, "throughput": 85.65},
        {"name": "nCompass", "latency": 1.63, "throughput": 47.57},
        {"name": "DeepInfra", "latency": 1.00, "throughput": 36.48}, 
        {"name": "SiliconFlow", "latency": 0.86, "throughput": 68.78},
        {"name": "NovitaAI", "latency": 1.00, "throughput": 66.44},
        {"name": "Parasail", "latency": 0.58, "throughput": 57.08},
        {"name": "Nebius AI", "latency": 0.64, "throughput": 35.06},
        {"name": "Friendli", "latency": 0.43, "throughput": 160.9},
        {"name": "Andre Inference", "latency": your_metrics.get('latency_seconds', 0), "throughput": your_metrics.get('throughput_tps', 0)},
    ]
    
    print(f"\nCOMPARISON WITH OPENROUTER")
    print("=" * 50)
    print(f"{'Provider':<15} {'Latency':<12} {'Throughput'}")
    print("-" * 50)
    
    for example in examples:
        highlight = " *" if "Andre" in example["name"] else ""
        print(f"{example['name']:<15} {example['latency']:<11.2f}s {example['throughput']:<8.1f}tps{highlight}")
    
    print("-" * 50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SGLang results to OpenRouter format")
    parser.add_argument(
        "--summary-file", 
        default=None,
        help="Path to benchmark summary JSON file (auto-detected if not specified)"
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-30B-A3B", 
        help="Model path to auto-detect results directory"
    )
    
    args = parser.parse_args()
    
    # Auto-detect summary file if not specified
    if args.summary_file is None:
        model_name = args.model_path.replace("/", "_").replace("-", "_").lower()
        auto_dir = f"./{model_name}_sglang_benchmark_results"
        args.summary_file = f"{auto_dir}/benchmark_summary.json"
    
    if not Path(args.summary_file).exists():
        print(f"Error: Summary file '{args.summary_file}' not found!")
        if args.summary_file.endswith("_sglang_benchmark_results/benchmark_summary.json"):
            print("Make sure you've run the benchmark first for this model.")
        sys.exit(1)
    
    with open(args.summary_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    model_name = data.get('model', 'Unknown Model')
    
    if not results:
        print("No benchmark results found!")
        sys.exit(1)
    
    or_metrics = calculate_openrouter_metrics(results)
    
    if not or_metrics:
        print("Could not extract enough metrics for conversion!")
        sys.exit(1)
    
    print_openrouter_format(or_metrics, model_name)
    
    compare_with_openrouter_examples(or_metrics)
    
    print(f"\nAndre Inference deployment:")
    print(f"Latency: {or_metrics['latency_seconds']:.2f}s")
    print(f"Throughput: {or_metrics['throughput_tps']:.0f}tps")
    print(f"TTFT: {or_metrics['ttft_ms']:.1f}ms")
    print(f"Cost: FREE (vs $0.02-0.60 per 1K tokens)")

if __name__ == "__main__":
    main()

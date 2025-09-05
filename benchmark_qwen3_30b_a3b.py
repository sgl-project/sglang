#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def run_benchmark_command(
    backend: str,
    model_path: str,
    dataset_name: str,
    num_prompts: int,
    request_rate: float = None,
    max_concurrency: int = None,
    random_input_len: int = None,
    random_output_len: int = None,
    sharegpt_output_len: int = None,
    host: str = "127.0.0.1",
    port: int = 30000,
    base_url: str = None,
    output_file: str = None,
    extra_args: List[str] = None
) -> Dict[str, Any]:
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", backend,
        "--model", model_path,
        "--dataset-name", dataset_name,
        "--num-prompts", str(num_prompts),
    ]
    
    if base_url:
        cmd.extend(["--base-url", base_url])
    else:
        cmd.extend(["--host", host, "--port", str(port)])
    
    if request_rate is not None:
        cmd.extend(["--request-rate", str(request_rate)])
    
    if max_concurrency is not None:
        cmd.extend(["--max-concurrency", str(max_concurrency)])
    
    if random_input_len is not None:
        cmd.extend(["--random-input-len", str(random_input_len)])
    
    if random_output_len is not None:
        cmd.extend(["--random-output-len", str(random_output_len)])
    
    if sharegpt_output_len is not None:
        cmd.extend(["--sharegpt-output-len", str(sharegpt_output_len)])
    
    if output_file:
        cmd.extend(["--output-file", output_file, "--output-details"])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return {"success": True, "output": result.stdout, "error": None}
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return {"success": False, "output": e.stdout, "error": e.stderr}

def run_ttft_baseline_benchmarks(model_path: str, output_dir: str) -> List[Dict[str, Any]]:
    results = []
    
    ttft_configs = [
        {
            "name": "ttft_baseline",
            "dataset": "random",
            "num_prompts": 20,
            "request_rate": 1.0,
            "max_concurrency": 1,
            "random_input_len": 512,
            "random_output_len": 64,
            "description": "Quick TTFT baseline"
        },
        {
            "name": "ttft_concurrent",
            "dataset": "random", 
            "num_prompts": 30,
            "request_rate": 5.0,
            "max_concurrency": 4,
            "random_input_len": 1024,
            "random_output_len": 128,
            "description": "TTFT with some concurrency"
        }
    ]
    
    for config in ttft_configs:
        print(f"\n{'='*60}")
        print(f"Running TTFT Benchmark: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_dir, f"ttft_{config['name']}.jsonl")
        
        kwargs = {
            "backend": "sglang",
            "model_path": model_path,
            "dataset_name": config["dataset"],
            "num_prompts": config["num_prompts"],
            "request_rate": config["request_rate"],
            "max_concurrency": config["max_concurrency"],
            "output_file": output_file,
            "extra_args": ["--disable-ignore-eos"]
        }
        
        if "random_input_len" in config:
            kwargs["random_input_len"] = config["random_input_len"]
            kwargs["random_output_len"] = config["random_output_len"]
        
        if "sharegpt_output_len" in config:
            kwargs["sharegpt_output_len"] = config["sharegpt_output_len"]
        
        result = run_benchmark_command(**kwargs)
        result["config"] = config
        results.append(result)
        
        time.sleep(2)
    
    return results

def run_throughput_baseline_benchmarks(model_path: str, output_dir: str) -> List[Dict[str, Any]]:
    results = []
    
    throughput_configs = [
        {
            "name": "throughput_burst",
            "dataset": "random",
            "num_prompts": 50,
            "request_rate": float('inf'),
            "max_concurrency": 16,
            "random_input_len": 512,
            "random_output_len": 128,
            "description": "Quick burst throughput test"
        },
        {
            "name": "throughput_sustained",
            "dataset": "random",
            "num_prompts": 40,
            "request_rate": 10.0,
            "max_concurrency": 8,
            "random_input_len": 1024,
            "random_output_len": 256,
            "description": "Sustained throughput baseline"
        }
    ]
    
    for config in throughput_configs:
        print(f"\n{'='*60}")
        print(f"Running Throughput Benchmark: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_dir, f"throughput_{config['name']}.jsonl")
        
        kwargs = {
            "backend": "sglang",
            "model_path": model_path,
            "dataset_name": config["dataset"],
            "num_prompts": config["num_prompts"],
            "request_rate": config["request_rate"],
            "max_concurrency": config["max_concurrency"],
            "output_file": output_file,
            "extra_args": ["--disable-ignore-eos"]
        }
        
        if "random_input_len" in config:
            kwargs["random_input_len"] = config["random_input_len"]
            kwargs["random_output_len"] = config["random_output_len"]
        
        if "sharegpt_output_len" in config:
            kwargs["sharegpt_output_len"] = config["sharegpt_output_len"]
        
        result = run_benchmark_command(**kwargs)
        result["config"] = config
        results.append(result)
        
        time.sleep(2)
    
    return results

def run_chat_api_benchmarks(model_path: str, output_dir: str) -> List[Dict[str, Any]]:
    results = []
    
    chat_configs = [
        {
            "name": "chat_api_baseline",
            "dataset": "random",
            "num_prompts": 20,
            "request_rate": 5.0,
            "max_concurrency": 4,
            "random_input_len": 512,
            "random_output_len": 128,
            "description": "Quick chat API test"
        }
    ]
    
    for config in chat_configs:
        print(f"\n{'='*60}")
        print(f"Running Chat API Benchmark: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        output_file = os.path.join(output_dir, f"chat_{config['name']}.jsonl")
        
        result = run_benchmark_command(
            backend="sglang-oai-chat",
            model_path=model_path,
            dataset_name=config["dataset"],
            num_prompts=config["num_prompts"],
            request_rate=config["request_rate"],
            max_concurrency=config["max_concurrency"],
            random_input_len=config["random_input_len"],
            random_output_len=config["random_output_len"],
            output_file=output_file,
            extra_args=["--apply-chat-template", "--disable-ignore-eos"]
        )
        result["config"] = config
        results.append(result)
        
        time.sleep(2)
    
    return results

def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    
    summary = {
        "model": "Qwen/Qwen3-30B-A3B",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_benchmarks": len(all_results),
        "successful_benchmarks": sum(1 for r in all_results if r.get("success", False)),
        "failed_benchmarks": sum(1 for r in all_results if not r.get("success", False)),
        "results": all_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {summary['model']}")
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful_benchmarks']}")
    print(f"Failed: {summary['failed_benchmarks']}")
    print(f"Summary saved to: {summary_file}")
    
    for result in all_results:
        if result.get("success", False):
            config = result.get("config", {})
            print(f"\n{config.get('name', 'Unknown')}: {config.get('description', 'No description')}")
        else:
            config = result.get("config", {})
            print(f"\n{config.get('name', 'Unknown')}: FAILED")

def check_server_status(host: str = "127.0.0.1", port: int = 30000) -> bool:
    import requests
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-30B-A3B model with SGLang"
    )
    parser.add_argument(
        "--model-path", 
        default="Qwen/Qwen3-30B-A3B",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save benchmark results (auto-generated from model name if not specified)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port"
    )
    parser.add_argument(
        "--skip-ttft",
        action="store_true",
        help="Skip TTFT baseline benchmarks"
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip throughput baseline benchmarks"
    )
    parser.add_argument(
        "--skip-chat-api",
        action="store_true",
        help="Skip chat API benchmarks"
    )
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Check if server is running before starting benchmarks"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output directory name from model if not specified
    if args.output_dir is None:
        model_name = args.model_path.replace("/", "_").replace("-", "_").lower()
        args.output_dir = f"./{model_name}_sglang_benchmark_results"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.check_server:
        print(f"Checking server status at {args.host}:{args.port}...")
        if not check_server_status(args.host, args.port):
            print(f"ERROR: Server not accessible at {args.host}:{args.port}")
            print("Please start the SGLang server first:")
            print(f"python -m sglang.launch_server --model-path {args.model_path} --port {args.port}")
            sys.exit(1)
        print("Server is accessible")
    
    print(f"Starting benchmark for model: {args.model_path}")
    print(f"Results will be saved to: {output_dir}")
    
    all_results = []
    
    if not args.skip_ttft:
        print("\n" + "="*60)
        print("STARTING TTFT BASELINE BENCHMARKS")
        print("="*60)
        ttft_results = run_ttft_baseline_benchmarks(args.model_path, str(output_dir))
        all_results.extend(ttft_results)
    
    if not args.skip_throughput:
        print("\n" + "="*60)
        print("STARTING THROUGHPUT BASELINE BENCHMARKS")
        print("="*60)
        throughput_results = run_throughput_baseline_benchmarks(args.model_path, str(output_dir))
        all_results.extend(throughput_results)
    
    if not args.skip_chat_api:
        print("\n" + "="*60)
        print("STARTING CHAT API BENCHMARKS")
        print("="*60)
        chat_results = run_chat_api_benchmarks(args.model_path, str(output_dir))
        all_results.extend(chat_results)
    
    generate_summary_report(all_results, str(output_dir))
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    main()

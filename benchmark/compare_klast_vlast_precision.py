#!/usr/bin/env python3
"""
K-last vs V-last Precision Comparison Script

This script:
1. Launches K-last server (with CUDA graph disabled), runs benchmark, logs tensors
2. Launches V-last server (with CUDA graph disabled), runs benchmark, logs tensors
3. Compares tensor precision differences layer by layer

Usage (inside sglang_dev docker container):
  python benchmark/compare_klast_vlast_precision.py

Or with custom parameters:
  python benchmark/compare_klast_vlast_precision.py --num-prompts 10 --tp 2
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import shutil
import requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Configuration - auto-detect paths
SCRIPT_DIR = Path(__file__).parent.resolve()
SGLANG_GDN_PATH = str(SCRIPT_DIR.parent)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 30000


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def kill_existing_server():
    """Kill any existing sglang server."""
    print("Killing existing servers...")
    try:
        subprocess.run(f"lsof -ti:{SERVER_PORT} | xargs -r kill -9", shell=True, capture_output=True)
        # NOTE: Do not blanket-kill all sglang servers on the machine.
        # This script is frequently run alongside other benchmarks (different ports).
        # Only kill the process bound to the target port.
        time.sleep(3)
    except Exception as e:
        print(f"Warning: {e}")


def launch_server(
    k_last: bool,
    log_dir: str,
    model_path: str,
    tp_size: int,
    base_gpu_id: int = 0,
    disable_cuda_graph: bool = False,
    disable_speculative: bool = False,
    skip_server_warmup: bool = False,
    chunked_prefill_size: int = None,
):
    """Launch server with specified layout."""
    mode = "klast" if k_last else "vlast"
    log_file = os.path.join(log_dir, f"server_{mode}.log")
    tensor_log_dir = os.path.join(log_dir, mode)
    os.makedirs(tensor_log_dir, exist_ok=True)
    
    # Clear old tensor log
    tensor_log_file = os.path.join(tensor_log_dir, "gdn_tensor_log.jsonl")
    if os.path.exists(tensor_log_file):
        os.remove(tensor_log_file)
    
    print(f"\n{'='*60}")
    print(f"Launching {mode.upper()} server...")
    print(f"Tensor logs: {tensor_log_dir}")
    print(f"{'='*60}")
    
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", str(tp_size),
        "--base-gpu-id", str(base_gpu_id),
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--trust-remote-code",
        "--mem-fraction-static", "0.70",
        "--disable-radix-cache",
        "--watchdog-timeout", "600",  # Increase timeout for JIT compilation
    ]
    if not disable_speculative:
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
        ]

    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    if skip_server_warmup:
        cmd.append("--skip-server-warmup")
    
    if chunked_prefill_size is not None:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    
    if k_last:
        cmd.append("--mamba-ssm-k-last")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"
    env["SGLANG_GDN_DEBUG"] = "0"  # Disable tensor logging for performance test
    env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"  # Disable FlashInfer version check
    # env["SGLANG_GDN_DEBUG_DIR"] = tensor_log_dir
    
    print(f"Command: {' '.join(cmd)}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            env=env, preexec_fn=os.setsid
        )
    
    return process, tensor_log_file


def wait_for_server(timeout: int = 1800):
    """Wait for server to be ready."""
    import socket
    import json
    import urllib.request
    print(f"Waiting for server (timeout: {timeout}s)...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            # If the server process exited early, fail fast.
            if hasattr(wait_for_server, "_process") and wait_for_server._process is not None:
                if wait_for_server._process.poll() is not None:
                    print("ERROR: Server process exited before becoming ready.")
                    return False

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            if sock.connect_ex((SERVER_HOST, SERVER_PORT)) == 0:
                sock.close()
                try:
                    req = urllib.request.Request(f"http://{SERVER_HOST}:{SERVER_PORT}/health")
                    resp = urllib.request.urlopen(req, timeout=10)
                    if resp.status == 200:
                        # Confirm model is actually loaded (health can be up earlier).
                        models_req = urllib.request.Request(
                            f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models"
                        )
                        models_resp = urllib.request.urlopen(models_req, timeout=10)
                        if models_resp.status == 200:
                            _ = json.loads(models_resp.read().decode("utf-8"))
                            print(f"Server ready! ({time.time() - start:.1f}s)")
                            return True
                except:
                    pass
            sock.close()
        except:
            pass
        time.sleep(5)
    
    print(f"ERROR: Server timeout after {timeout}s")
    return False


def start_profiling(output_dir: str, num_steps: int = 5, activities: list = None, profile_by_stage: bool = True):
    """Start torch profiling via HTTP API."""
    if activities is None:
        activities = ["CPU", "GPU"]
    
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/start_profile"
    json_data = {
        "output_dir": output_dir,
        "num_steps": str(num_steps),
        "activities": activities,
        "profile_by_stage": profile_by_stage,  # True: profile num_steps prefills AND num_steps decodes separately
        "merge_profiles": True,
    }
    
    print(f"Starting profiler: {num_steps} steps, activities={activities}, profile_by_stage={profile_by_stage}, output_dir={output_dir}")
    try:
        response = requests.post(url=url, json=json_data, timeout=300)
        response.raise_for_status()
        print(f"Profiler started. Output dir: {output_dir}")
        return True
    except Exception as e:
        print(f"Warning: Profiling start failed: {e}")
        return False


def run_profile_requests(model_path: str, num_requests: int = 8, input_len: int = 128, output_len: int = 256, seed: int = 42):
    """Run requests to trigger profiling steps."""
    print(f"Running {num_requests} requests to trigger profiling...")
    
    cmd = [
        "python", f"{SGLANG_GDN_PATH}/python/sglang/bench_serving.py",
        "--backend", "sglang",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--model", model_path,
        "--dataset-name", "random",
        "--random-input", str(input_len),
        "--random-output", str(output_len),
        "--num-prompts", str(num_requests),
        "--seed", str(seed + 1000),  # Different seed for profiling run
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    print("Profiling requests completed.")
    return result.returncode == 0


def run_one_batch_profile(
    batch_size: int = 1,
    input_len: int = 1024,
    output_len: int = 8,
    profile_steps: int = 5,
    profile_output_dir: str = None,
    profile_prefix: str = None,
):
    """Run bench_one_batch_server.py with --profile to get trace.
    
    This uses the already-running server via --base-url.
    """
    print(f"\n{'='*60}")
    print(f"Running one-batch profiling: batch_size={batch_size}, input_len={input_len}, output_len={output_len}")
    print(f"profile_steps={profile_steps}, output_dir={profile_output_dir}")
    print(f"{'='*60}")
    
    os.makedirs(profile_output_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "sglang.bench_one_batch_server",
        "--model", "None",  # Use existing server
        "--base-url", f"http://{SERVER_HOST}:{SERVER_PORT}",
        "--batch-size", str(batch_size),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--profile",
        "--profile-steps", str(profile_steps),
        "--profile-by-stage",
        "--profile-output-dir", profile_output_dir,
    ]
    if profile_prefix:
        cmd += ["--profile-prefix", profile_prefix]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    
    print("=== STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    
    print(f"One-batch profiling completed. Return code: {result.returncode}")
    return result.returncode == 0


def stop_profiling():
    """Stop torch profiling via HTTP API."""
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/stop_profile"
    
    print("Stopping profiler...")
    try:
        response = requests.post(url=url, timeout=300)
        response.raise_for_status()
        print("Profiler stopped and traces saved.")
        return True
    except Exception as e:
        print(f"Warning: Profiling stop failed: {e}")
        return False


def run_benchmark(model_path: str, num_prompts: int, input_len: int, output_len: int, result_file: str, seed: int = 42, 
                  enable_profile: bool = False, profile_num_steps: int = 5, profile_activities: list = None, profile_output_dir: str = None,
                  profile_batch_size: int = 1, profile_input_len: int = None, profile_prefix: str = None):
    """Run benchmark."""
    print(f"\nRunning benchmark: {num_prompts} prompts (seed={seed})...")
    
    cmd = [
        "python", f"{SGLANG_GDN_PATH}/python/sglang/bench_serving.py",
        "--backend", "sglang",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--model", model_path,
        "--dataset-name", "random",
        "--random-input", str(input_len),
        "--random-output", str(output_len),
        "--num-prompts", str(num_prompts),
        "--seed", str(seed),  # Fixed seed for reproducibility
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1200)
    
    # Run profiling if enabled: use bench_one_batch_server --profile for better traces
    if enable_profile and profile_output_dir:
        # Use profile_input_len if specified, otherwise use input_len
        actual_profile_input_len = profile_input_len if profile_input_len is not None else input_len
        run_one_batch_profile(
            batch_size=profile_batch_size,
            input_len=actual_profile_input_len,
            output_len=output_len,
            profile_steps=profile_num_steps,
            profile_output_dir=profile_output_dir,
            profile_prefix=profile_prefix,
        )
    
    # Save benchmark output
    output_file = result_file.replace(".jsonl", "_output.txt")
    with open(output_file, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
    
    # Print stdout
    print(result.stdout)
    
    # Extract metrics
    accept_length = None
    output_throughput = None
    for line in result.stdout.split("\n"):
        if "Accept length:" in line:
            try:
                accept_length = float(line.split(":")[-1].strip())
            except:
                pass
        if "Output token throughput" in line:
            try:
                output_throughput = float(line.split(":")[-1].strip().split()[0])
            except:
                pass
    
    print(f"Benchmark complete. Accept length: {accept_length}, Output throughput: {output_throughput}")
    return accept_length, output_throughput


def run_gsm8k_eval(
    log_dir: str,
    num_questions: int = 200,
    num_shots: int = 5,
    max_new_tokens: int = 512,
    parallel: int = 128,
):
    """Run few-shot GSM8K accuracy evaluation against the current running server."""
    print(
        f"\nRunning GSM8K eval: questions={num_questions}, shots={num_shots}, "
        f"max_new_tokens={max_new_tokens}, parallel={parallel}..."
    )

    cmd = [
        "python",
        "-m",
        "sglang.test.few_shot_gsm8k",
        "--num-questions",
        str(num_questions),
        "--num-shots",
        str(num_shots),
        "--max-new-tokens",
        str(max_new_tokens),
        "--parallel",
        str(parallel),
        "--host",
        f"http://{SERVER_HOST}",
        "--port",
        str(SERVER_PORT),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"

    os.makedirs(log_dir, exist_ok=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=log_dir, timeout=7200
    )

    # Save output
    out_path = os.path.join(log_dir, "gsm8k_eval_output.txt")
    with open(out_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    print(result.stdout)
    if result.returncode != 0:
        print(f"Warning: GSM8K eval returned {result.returncode}")
        print(result.stderr)

    # Parse key metrics
    acc = None
    invalid = None
    out_tput = None
    for line in result.stdout.split("\n"):
        if line.startswith("Accuracy:"):
            try:
                acc = float(line.split(":", 1)[1].strip())
            except:
                pass
        if line.startswith("Invalid:"):
            try:
                invalid = float(line.split(":", 1)[1].strip())
            except:
                pass
        if line.startswith("Output throughput:"):
            try:
                out_tput = float(line.split(":", 1)[1].strip().split()[0])
            except:
                pass

    print(f"GSM8K eval complete. Accuracy: {acc}, Invalid: {invalid}, Output throughput: {out_tput}")
    return {"accuracy": acc, "invalid": invalid, "output_throughput": out_tput, "output_file": out_path}


def run_mmlu_eval(
    log_dir: str,
    num_examples: int = 1000,
    num_threads: int = 128,
    max_tokens: int = 512,
):
    """Run MMLU evaluation (simple-evals) against the current running server."""
    print(
        f"\nRunning MMLU eval: examples={num_examples}, threads={num_threads}, max_tokens={max_tokens}..."
    )

    cmd = [
        "python",
        "-m",
        "sglang.test.run_eval",
        "--eval-name",
        "mmlu",
        "--num-examples",
        str(num_examples),
        "--num-threads",
        str(num_threads),
        "--max-tokens",
        str(max_tokens),
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SGLANG_GDN_PATH}/python:{env.get('PYTHONPATH', '')}"

    os.makedirs(log_dir, exist_ok=True)
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=log_dir, timeout=7200
    )

    out_path = os.path.join(log_dir, "mmlu_eval_output.txt")
    with open(out_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    print(result.stdout)
    if result.returncode != 0:
        print(f"Warning: MMLU eval returned {result.returncode}")
        print(result.stderr)

    score = None
    latency = None
    for line in result.stdout.split("\n"):
        if line.startswith("Total latency:"):
            try:
                latency = float(line.split(":", 1)[1].strip().split()[0])
            except:
                pass
        if line.startswith("Score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except:
                pass

    print(f"MMLU eval complete. Score: {score}, Total latency: {latency}")
    return {"score": score, "latency": latency, "output_file": out_path}


def cleanup_server(process):
    """Cleanup server process."""
    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass
    kill_existing_server()


def main():
    global SERVER_HOST, SERVER_PORT
    parser = argparse.ArgumentParser(description="K-last vs V-last Precision Comparison")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--host", type=str, default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--base-gpu-id", type=int, default=0, help="Base GPU ID for tensor parallelism (will use base_gpu_id to base_gpu_id+tp-1)")
    parser.add_argument("--num-prompts", type=int, default=64)  # Increased for better stats
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=1024)  # Long output for more verify steps
    parser.add_argument("--seed", type=int, default=42)  # Fixed seed for reproducibility
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")
    parser.add_argument("--skip-klast", action="store_true")
    parser.add_argument("--skip-vlast", action="store_true")
    parser.add_argument(
        "--disable-speculative",
        action="store_true",
        help="Disable speculative decoding (omit --speculative-* server args).",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph on server launch (recommended for NCU/profilers).",
    )
    parser.add_argument(
        "--skip-server-warmup",
        action="store_true",
        help="Pass --skip-server-warmup when launching server (useful for profiling).",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=None,
        help="Chunked prefill size for the server (e.g., 32768 for 8192*4)",
    )
    # Dataset accuracy eval (GSM8K)
    parser.add_argument(
        "--gsm8k-eval",
        action="store_true",
        help="Run few-shot GSM8K eval instead of serving benchmark",
    )
    parser.add_argument("--gsm8k-num-questions", type=int, default=200)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-parallel", type=int, default=128)
    # Dataset accuracy eval (MMLU)
    parser.add_argument(
        "--mmlu-eval",
        action="store_true",
        help="Run MMLU eval (simple-evals) instead of serving benchmark",
    )
    parser.add_argument("--mmlu-num-examples", type=int, default=1000)
    parser.add_argument("--mmlu-num-threads", type=int, default=128)
    parser.add_argument("--mmlu-max-tokens", type=int, default=512)
    # Torch profiling
    parser.add_argument(
        "--enable-profile",
        action="store_true",
        help="Enable torch profiling after benchmark",
    )
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=5,
        help="Number of forward steps to profile (default: 5)",
    )
    parser.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "GPU"],
        help="Profiling activities: CPU, GPU, MEM (default: CPU GPU)",
    )
    parser.add_argument(
        "--profile-batch-size",
        type=int,
        default=1,
        help="Batch size for profiling (default: 1)",
    )
    parser.add_argument(
        "--profile-input-len",
        type=int,
        default=None,
        help="Input length for profiling. If not set, uses --input-len. Set to larger value (e.g. 4096) for meaningful prefill traces.",
    )
    args = parser.parse_args()

    # Update global host/port for this run
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    
    timestamp = get_timestamp()
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# K-last vs V-last Precision Comparison")
    print(f"# Output: {log_dir}")
    print(f"# GPUs: {args.base_gpu_id} to {args.base_gpu_id + args.tp - 1}")
    if args.gsm8k_eval or args.mmlu_eval:
        print(f"# TP={args.tp}, Accuracy Eval Mode")
    else:
        print(f"# TP={args.tp}, Prompts={args.num_prompts}, Seed={args.seed}")
    print(f"{'#'*60}")
    
    klast_accept = None
    vlast_accept = None
    klast_throughput = None
    vlast_throughput = None
    klast_gsm8k = None
    vlast_gsm8k = None
    klast_mmlu = None
    vlast_mmlu = None
    
    try:
        # Run K-last
        if not args.skip_klast:
            kill_existing_server()
            process, _ = launch_server(
                True,
                log_dir,
                args.model_path,
                args.tp,
                base_gpu_id=args.base_gpu_id,
                disable_cuda_graph=(args.disable_cuda_graph or args.gsm8k_eval or args.mmlu_eval),
                disable_speculative=args.disable_speculative,
                skip_server_warmup=args.skip_server_warmup,
                chunked_prefill_size=args.chunked_prefill_size,
            )
            wait_for_server._process = process
            if wait_for_server():
                if args.gsm8k_eval or args.mmlu_eval:
                    if args.gsm8k_eval:
                        klast_gsm8k = run_gsm8k_eval(
                            log_dir=os.path.join(log_dir, "klast"),
                            num_questions=args.gsm8k_num_questions,
                            num_shots=args.gsm8k_num_shots,
                            max_new_tokens=args.gsm8k_max_new_tokens,
                            parallel=args.gsm8k_parallel,
                        )
                    if args.mmlu_eval:
                        klast_mmlu = run_mmlu_eval(
                            log_dir=os.path.join(log_dir, "klast"),
                            num_examples=args.mmlu_num_examples,
                            num_threads=args.mmlu_num_threads,
                            max_tokens=args.mmlu_max_tokens,
                        )
                else:
                    klast_accept, klast_throughput = run_benchmark(
                        args.model_path,
                        args.num_prompts, args.input_len, args.output_len,
                        os.path.join(log_dir, "benchmark_klast.jsonl"),
                        seed=args.seed,
                        enable_profile=args.enable_profile,
                        profile_num_steps=args.profile_num_steps,
                        profile_activities=args.profile_activities,
                        profile_output_dir=os.path.join(log_dir, "profile_klast") if args.enable_profile else None,
                        profile_batch_size=args.profile_batch_size,
                        profile_input_len=args.profile_input_len,
                        profile_prefix="klast",
                    )
            cleanup_server(process)
            time.sleep(5)
        
        # Run V-last
        if not args.skip_vlast:
            kill_existing_server()
            process, _ = launch_server(
                False,
                log_dir,
                args.model_path,
                args.tp,
                base_gpu_id=args.base_gpu_id,
                disable_cuda_graph=(args.disable_cuda_graph or args.gsm8k_eval or args.mmlu_eval),
                disable_speculative=args.disable_speculative,
                skip_server_warmup=args.skip_server_warmup,
                chunked_prefill_size=args.chunked_prefill_size,
            )
            wait_for_server._process = process
            if wait_for_server():
                if args.gsm8k_eval or args.mmlu_eval:
                    if args.gsm8k_eval:
                        vlast_gsm8k = run_gsm8k_eval(
                            log_dir=os.path.join(log_dir, "vlast"),
                            num_questions=args.gsm8k_num_questions,
                            num_shots=args.gsm8k_num_shots,
                            max_new_tokens=args.gsm8k_max_new_tokens,
                            parallel=args.gsm8k_parallel,
                        )
                    if args.mmlu_eval:
                        vlast_mmlu = run_mmlu_eval(
                            log_dir=os.path.join(log_dir, "vlast"),
                            num_examples=args.mmlu_num_examples,
                            num_threads=args.mmlu_num_threads,
                            max_tokens=args.mmlu_max_tokens,
                        )
                else:
                    vlast_accept, vlast_throughput = run_benchmark(
                        args.model_path,
                        args.num_prompts, args.input_len, args.output_len,
                        os.path.join(log_dir, "benchmark_vlast.jsonl"),
                        seed=args.seed,
                        enable_profile=args.enable_profile,
                        profile_num_steps=args.profile_num_steps,
                        profile_activities=args.profile_activities,
                        profile_output_dir=os.path.join(log_dir, "profile_vlast") if args.enable_profile else None,
                        profile_batch_size=args.profile_batch_size,
                        profile_input_len=args.profile_input_len,
                        profile_prefix="vlast",
                    )
            cleanup_server(process)
        
        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        if args.gsm8k_eval or args.mmlu_eval:
            print("Eval mode")
            if args.gsm8k_eval:
                print(f"K-last GSM8K: {klast_gsm8k}")
                print(f"V-last GSM8K: {vlast_gsm8k}")
                if (
                    klast_gsm8k
                    and vlast_gsm8k
                    and klast_gsm8k.get("accuracy") is not None
                    and vlast_gsm8k.get("accuracy") is not None
                ):
                    diff = klast_gsm8k["accuracy"] - vlast_gsm8k["accuracy"]
                    pct = diff * 100
                    print(f"GSM8K Accuracy Difference: {diff:+.3f} ({pct:+.1f}pp)")
            if args.mmlu_eval:
                print(f"K-last MMLU: {klast_mmlu}")
                print(f"V-last MMLU: {vlast_mmlu}")
                if (
                    klast_mmlu
                    and vlast_mmlu
                    and klast_mmlu.get("score") is not None
                    and vlast_mmlu.get("score") is not None
                ):
                    diff = klast_mmlu["score"] - vlast_mmlu["score"]
                    pct = diff * 100
                    print(f"MMLU Score Difference: {diff:+.3f} ({pct:+.1f}pp)")
        else:
            print(f"K-last Accept Length: {klast_accept}")
            print(f"V-last Accept Length: {vlast_accept}")
            print(f"K-last Output Throughput: {klast_throughput}")
            print(f"V-last Output Throughput: {vlast_throughput}")
            if klast_accept and vlast_accept:
                diff = klast_accept - vlast_accept
                pct = (diff / vlast_accept) * 100
                print(f"Accept Length Difference: {diff:+.3f} ({pct:+.1f}%)")
            if klast_throughput and vlast_throughput:
                diff = klast_throughput - vlast_throughput
                pct = (diff / vlast_throughput) * 100
                print(f"Throughput Difference: {diff:+.2f} ({pct:+.1f}%)")
        print(f"\nLogs saved to: {log_dir}")
        
        # Save summary to JSON
        summary = {
            "timestamp": timestamp,
            "config": {
                "model_path": args.model_path,
                "tp": args.tp,
                "base_gpu_id": args.base_gpu_id,
                "num_prompts": args.num_prompts,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "seed": args.seed,
                "gsm8k_eval": args.gsm8k_eval,
                "mmlu_eval": args.mmlu_eval,
            },
            "results": {
                "klast": {
                    "accept_length": klast_accept,
                    "output_throughput": klast_throughput,
                    "gsm8k": klast_gsm8k,
                    "mmlu": klast_mmlu,
                },
                "vlast": {
                    "accept_length": vlast_accept,
                    "output_throughput": vlast_throughput,
                    "gsm8k": vlast_gsm8k,
                    "mmlu": vlast_mmlu,
                }
            }
        }
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        kill_existing_server()


if __name__ == "__main__":
    main()

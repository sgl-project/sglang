import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


class TuningMode(Enum):
    LOW_LATENCY = "low_latency"
    HIGH_THROUGHPUT_TPOT20 = "high_throughput_tpot20"
    HIGH_THROUGHPUT_TPOT50 = "high_throughput_tpot50"


class DeployMode(Enum):
    MIXED = "mixed"
    PD_SEPARATED = "pd_separated"


@dataclass
class ServerConfig:
    model_path: str
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    mem_fraction_static: float = 0.85
    max_running_requests: int = 32
    cuda_graph_bs: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    attention_backend: str = "flashinfer"
    device: str = "cuda"
    quantization: Optional[str] = None
    enable_cuda_graph: bool = True
    enable_mtp: bool = True
    enable_deepep: bool = True
    deepep_mode: str = "auto"
    moe_a2a_backend: str = "deepep"
    speculative_algorithm: Optional[str] = None
    speculative_num_steps: int = 1
    disable_radix_cache: bool = True
    chunked_prefill_size: int = -1
    dtype: str = "bfloat16"
    host: str = "127.0.0.1"
    port: int = 8000
    extra_args: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    throughput: float = 0.0
    request_throughput: float = 0.0
    success: bool = False
    error: str = ""
    config: Optional[ServerConfig] = None


@dataclass
class TuningResult:
    best_config: ServerConfig
    best_metrics: BenchmarkResult
    all_results: List[BenchmarkResult]
    tuning_mode: TuningMode
    deploy_mode: DeployMode


class AutoTuneSkill:
    def __init__(self):
        self.server_proc = None
        self.router_proc = None
        self.results: List[BenchmarkResult] = []
        self.best_result: Optional[BenchmarkResult] = None
        
    def build_server_cmd(self, config: ServerConfig, mode: DeployMode, 
                        is_prefill: bool = False) -> List[str]:
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", config.model_path,
            "--host", config.host,
            "--port", str(config.port if not is_prefill else 8000),
            "--tp-size", str(config.tp_size),
            "--dp-size", str(config.dp_size),
            "--mem-fraction-static", str(config.mem_fraction_static),
            "--max-running-requests", str(config.max_running_requests),
            "--attention-backend", config.attention_backend,
            "--device", config.device,
            "--dtype", config.dtype,
        ]
        
        if config.quantization:
            cmd.extend(["--quantization", config.quantization])
            
        if config.enable_cuda_graph:
            cmd.extend(["--cuda-graph-bs"] + list(map(str, config.cuda_graph_bs)))
        else:
            cmd.append("--disable-cuda-graph")
            
        if config.enable_deepep:
            cmd.extend([
                "--moe-a2a-backend", config.moe_a2a_backend,
                "--deepep-mode", config.deepep_mode
            ])
            
        if config.disable_radix_cache:
            cmd.append("--disable-radix-cache")
            
        if config.chunked_prefill_size > 0:
            cmd.extend(["--chunked-prefill-size", str(config.chunked_prefill_size)])
        else:
            cmd.extend(["--chunked-prefill-size", "-1"])
            
        if config.speculative_algorithm:
            cmd.extend([
                "--speculative-algorithm", config.speculative_algorithm,
                "--speculative-num-steps", str(config.speculative_num_steps),
            ])
            
        if mode == DeployMode.PD_SEPARATED:
            role = "prefill" if is_prefill else "decode"
            cmd.extend([f"--disaggregation-mode", role])
            if is_prefill:
                cmd.extend(["--disaggregation-bootstrap-port", "8998"])
                
        cmd.extend(config.extra_args)
        
        return cmd
    
    def build_router_cmd(self, prefill_host: str = "127.0.0.1", 
                         decode_host: str = "127.0.0.1") -> List[str]:
        return [
            sys.executable, "-m", "sglang_router.launch_router",
            "--pd-disaggregation",
            "--policy", "cache_aware",
            "--prefill", f"http://{prefill_host}:8000", "8998",
            "--decode", f"http://{decode_host}:8001",
            "--host", "127.0.0.1",
            "--port", "6688",
            "--mini-lb"
        ]
    
    def build_benchmark_cmd(self, config: ServerConfig, input_len: int, 
                           output_len: int, mode: DeployMode,
                           num_prompts: int, max_concurrency: int,
                           request_rate: int = 16) -> List[str]:
        port = "6688" if mode == DeployMode.PD_SEPARATED else str(config.port)
        return [
            sys.executable, "-m", "sglang.bench_serving",
            "--dataset-name", "random",
            "--backend", "sglang",
            "--host", "127.0.0.1",
            "--port", port,
            "--max-concurrency", str(max_concurrency),
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--num-prompts", str(num_prompts),
            "--random-range-ratio", "1",
            "--request-rate", str(request_rate),
            "--warmup-requests", "2",
        ]
    
    def start_server(self, config: ServerConfig, mode: DeployMode) -> bool:
        try:
            if mode == DeployMode.PD_SEPARATED:
                prefill_cmd = self.build_server_cmd(config, mode, is_prefill=True)
                print(f"Starting prefill server: {' '.join(prefill_cmd)}")
                prefill_env = os.environ.copy()
                prefill_env["SGLANG_DEEPEP_MODE"] = "normal"
                self.server_proc = subprocess.Popen(
                    prefill_cmd, env=prefill_env, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
                time.sleep(120)
                
                decode_config = ServerConfig(**config.__dict__)
                decode_config.port = 8001
                decode_config.max_running_requests = config.max_running_requests * config.dp_size
                decode_cmd = self.build_server_cmd(decode_config, mode, is_prefill=False)
                print(f"Starting decode server: {' '.join(decode_cmd)}")
                decode_env = os.environ.copy()
                decode_env["SGLANG_DEEPEP_MODE"] = "low_latency"
                
                self.server_proc = subprocess.Popen(
                    decode_cmd, env=decode_env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
                time.sleep(120)
                
                router_cmd = self.build_router_cmd()
                print(f"Starting router: {' '.join(router_cmd)}")
                self.router_proc = subprocess.Popen(
                    router_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
                time.sleep(30)
            else:
                cmd = self.build_server_cmd(config, mode)
                print(f"Starting server: {' '.join(cmd)}")
                env = os.environ.copy()
                env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "64"
                self.server_proc = subprocess.Popen(
                    cmd, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                
            time.sleep(60)
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        if self.router_proc:
            self.router_proc.terminate()
            self.router_proc.wait()
            self.router_proc = None
        if self.server_proc:
            self.server_proc.terminate()
            self.server_proc.wait()
            self.server_proc = None
    
    def run_benchmark(self, config: ServerConfig, input_len: int, 
                     output_len: int, mode: DeployMode,
                     num_prompts: int, max_concurrency: int) -> BenchmarkResult:
        cmd = self.build_benchmark_cmd(config, input_len, output_len, mode, 
                                      num_prompts, max_concurrency)
        print(f"Running benchmark: {' '.join(cmd)}")
        
        try:
            result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            final_output = result2.stdout
            
            return self.parse_benchmark_output(final_output, config)
        except subprocess.TimeoutExpired:
            return BenchmarkResult(success=False, error="Benchmark timeout")
        except Exception as e:
            return BenchmarkResult(success=False, error=str(e))
    
    def parse_benchmark_output(self, output: str, config: ServerConfig) -> BenchmarkResult:
        result = BenchmarkResult(config=config, success=True)
        
        try:
            lines = output.split('\n')
            for line in lines:
                if "mean_tpot_ms" in line or "mean tpot" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "tpot" in part.lower() and i + 1 < len(parts):
                            try:
                                result.tpot_ms = float(parts[i + 1])
                            except:
                                pass
                elif "p99_tpot" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "p99" in part.lower() and i + 1 < len(parts):
                            try:
                                result.p99_tpot_ms = float(parts[i + 1])
                            except:
                                pass
                elif "ttft" in line.lower() and "mean" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "ttft" in part.lower() and i + 1 < len(parts):
                            try:
                                result.ttft_ms = float(parts[i + 1])
                            except:
                                pass
                elif "throughput" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "throughput" in part.lower() and i + 1 < len(parts):
                            try:
                                result.throughput = float(parts[i + 1])
                            except:
                                pass
                elif "request_throughput" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "request" in part.lower() and i + 1 < len(parts):
                            try:
                                result.request_throughput = float(parts[i + 1])
                            except:
                                pass
        except Exception as e:
            result.success = False
            result.error = f"Parse error: {e}"
        
        return result
    
    def get_cuda_graph_bs_candidates(self, max_running_requests: int, 
                                    dp_size: int) -> List[List[int]]:
        base = max_running_requests // dp_size if dp_size > 0 else max_running_requests
        candidates = []
        
        for max_bs in [base, base * 2, base * 4]:
            if max_bs >= 2:
                bs_list = list(range(2, max_bs + 1, 2))
                if bs_list:
                    candidates.append(bs_list)
        
        if not candidates:
            candidates = [[2, 4, 8]]
        
        return candidates
    
    def get_mem_fraction_candidates(self, base: float = 0.85) -> List[float]:
        return [base - 0.1, base, base + 0.05, base + 0.1]
    
    def objective_function(self, result: BenchmarkResult, 
                          mode: TuningMode) -> float:
        if not result.success:
            return float('inf')
        
        if mode == TuningMode.LOW_LATENCY:
            return result.p99_tpot_ms
        elif mode == TuningMode.HIGH_THROUGHPUT_TPOT20:
            if result.p99_tpot_ms > 20:
                return float('inf')
            return -result.request_throughput
        elif mode == TuningMode.HIGH_THROUGHPUT_TPOT50:
            if result.p99_tpot_ms > 50:
                return float('inf')
            return -result.request_throughput
        else:
            return result.p99_tpot_ms
    
    def tune(self, model_path: str, 
             tuning_mode: TuningMode = TuningMode.LOW_LATENCY,
             deploy_mode: DeployMode = DeployMode.MIXED,
             input_len: int = 3500,
             output_len: int = 1500,
             tp_size: int = 8,
             device: str = "cuda",
             quantization: Optional[str] = None) -> TuningResult:
        self.results = []
        
        base_config = ServerConfig(
            model_path=model_path,
            tp_size=tp_size,
            dp_size=1 if tuning_mode == TuningMode.LOW_LATENCY else min(4, tp_size),
            device=device,
            quantization=quantization,
        )
        
        if tuning_mode == TuningMode.LOW_LATENCY:
            base_config.max_running_requests = 8
            base_config.dp_size = 1
            base_config.speculative_algorithm = "NEXTN"
            base_config.speculative_num_steps = 3
        else:
            base_config.max_running_requests = 64
            base_config.dp_size = min(4, tp_size)
        
        mem_fraction_candidates = self.get_mem_fraction_candidates(0.8)
        cuda_graph_candidates = self.get_cuda_graph_bs_candidates(
            base_config.max_running_requests, base_config.dp_size
        )
        
        features_to_test = [
            {"enable_mtp": True, "speculative_num_steps": 3},
            {"enable_mtp": False, "speculative_num_steps": 1},
        ]
        
        best_objective = float('inf')
        best_result = None
        
        for mem_fraction in mem_fraction_candidates:
            for cuda_graph_bs in cuda_graph_candidates:
                for features in features_to_test:
                    config = ServerConfig(**base_config.__dict__)
                    config.mem_fraction_static = mem_fraction
                    config.cuda_graph_bs = cuda_graph_bs
                    config.enable_mtp = features["enable_mtp"]
                    config.speculative_num_steps = features["speculative_num_steps"]
                    
                    if tuning_mode == TuningMode.LOW_LATENCY:
                        config.max_running_requests = 4
                        config.dp_size = 1
                    else:
                        config.max_running_requests = max(cuda_graph_bs) * config.dp_size
                    
                    num_prompts = config.max_running_requests
                    max_concurrency = config.max_running_requests
                    
                    print(f"\n{'='*80}")
                    print(f"Testing config:")
                    print(f"  mem_fraction_static: {config.mem_fraction_static}")
                    print(f"  cuda_graph_bs: {config.cuda_graph_bs}")
                    print(f"  max_running_requests: {config.max_running_requests}")
                    print(f"  dp_size: {config.dp_size}")
                    print(f"  enable_mtp: {config.enable_mtp}")
                    print(f"  speculative_num_steps: {config.speculative_num_steps}")
                    print(f"{'='*80}")
                    
                    if self.start_server(config, deploy_mode):
                        try:
                            result = self.run_benchmark(
                                config, input_len, output_len, deploy_mode,
                                num_prompts, max_concurrency
                            )
                            self.results.append(result)
                            
                            if result.success:
                                print(f"\nBenchmark Result:")
                                print(f"  p99_tpot_ms: {result.p99_tpot_ms:.2f}")
                                print(f"  mean_tpot_ms: {result.tpot_ms:.2f}")
                                print(f"  ttft_ms: {result.ttft_ms:.2f}")
                                print(f"  request_throughput: {result.request_throughput:.2f}")
                                
                                objective = self.objective_function(result, tuning_mode)
                                if objective < best_objective:
                                    best_objective = objective
                                    best_result = result
                                    print("  --> NEW BEST CONFIG!")
                        finally:
                            self.stop_server()
                    else:
                        self.results.append(BenchmarkResult(
                            success=False, 
                            error="Failed to start server",
                            config=config
                        ))
        
        if best_result is None:
            best_result = BenchmarkResult(success=False, error="No successful run")
            best_config = base_config
        else:
            best_config = best_result.config
        
        return TuningResult(
            best_config=best_config,
            best_metrics=best_result,
            all_results=self.results,
            tuning_mode=tuning_mode,
            deploy_mode=deploy_mode
        )
    
    def save_results(self, result: TuningResult, output_path: str):
        output = {
            "tuning_mode": result.tuning_mode.value,
            "deploy_mode": result.deploy_mode.value,
            "best_config": {
                "model_path": result.best_config.model_path,
                "tp_size": result.best_config.tp_size,
                "dp_size": result.best_config.dp_size,
                "ep_size": result.best_config.ep_size,
                "mem_fraction_static": result.best_config.mem_fraction_static,
                "max_running_requests": result.best_config.max_running_requests,
                "cuda_graph_bs": result.best_config.cuda_graph_bs,
                "attention_backend": result.best_config.attention_backend,
                "device": result.best_config.device,
                "quantization": result.best_config.quantization,
                "enable_cuda_graph": result.best_config.enable_cuda_graph,
                "enable_mtp": result.best_config.enable_mtp,
                "enable_deepep": result.best_config.enable_deepep,
                "deepep_mode": result.best_config.deepep_mode,
                "moe_a2a_backend": result.best_config.moe_a2a_backend,
                "speculative_algorithm": result.best_config.speculative_algorithm,
                "speculative_num_steps": result.best_config.speculative_num_steps,
                "disable_radix_cache": result.best_config.disable_radix_cache,
                "chunked_prefill_size": result.best_config.chunked_prefill_size,
                "dtype": result.best_config.dtype,
            },
            "best_metrics": {
                "tpot_ms": result.best_metrics.tpot_ms,
                "p99_tpot_ms": result.best_metrics.p99_tpot_ms,
                "ttft_ms": result.best_metrics.ttft_ms,
                "p99_ttft_ms": result.best_metrics.p99_ttft_ms,
                "throughput": result.best_metrics.throughput,
                "request_throughput": result.best_metrics.request_throughput,
                "success": result.best_metrics.success,
            },
            "all_results": [
                {
                    "config": {
                        "mem_fraction_static": r.config.mem_fraction_static if r.config else 0,
                        "cuda_graph_bs": r.config.cuda_graph_bs if r.config else [],
                        "max_running_requests": r.config.max_running_requests if r.config else 0,
                        "dp_size": r.config.dp_size if r.config else 0,
                    },
                    "metrics": {
                        "tpot_ms": r.tpot_ms,
                        "p99_tpot_ms": r.p99_tpot_ms,
                        "ttft_ms": r.ttft_ms,
                        "throughput": r.throughput,
                        "request_throughput": r.request_throughput,
                        "success": r.success,
                    },
                }
                for r in result.all_results
            ],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def generate_deployment_script(self, result: TuningResult, output_path: str):
        config = result.best_config
        mode = result.deploy_mode
        
        script = f"""#!/bin/bash
# Auto-generated deployment script for {result.tuning_mode.value} mode
# Best metrics: p99_tpot={result.best_metrics.p99_tpot_ms:.2f}ms, throughput={result.best_metrics.request_throughput:.2f} req/s

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="{config.model_path}"
HOST="{config.host}"
PORT={config.port}

"""
        
        if mode == DeployMode.MIXED:
            script += f"""python -m sglang.launch_server \\
    --model-path ${{MODEL_PATH}} \\
    --host ${{HOST}} \\
    --port ${{PORT}} \\
    --tp-size {config.tp_size} \\
    --dp-size {config.dp_size} \\
    --mem-fraction-static {config.mem_fraction_static} \\
    --max-running-requests {config.max_running_requests} \\
    --attention-backend {config.attention_backend} \\
    --device {config.device} \\
    --dtype {config.dtype} \\
"""
            if config.quantization:
                script += f"    --quantization {config.quantization} \\\n"
            if config.enable_cuda_graph:
                script += f"    --cuda-graph-bs {' '.join(map(str, config.cuda_graph_bs))} \\\n"
            else:
                script += "    --disable-cuda-graph \\\n"
            if config.enable_deepep:
                script += f"    --moe-a2a-backend {config.moe_a2a_backend} \\\n"
                script += f"    --deepep-mode {config.deepep_mode} \\\n"
            if config.disable_radix_cache:
                script += "    --disable-radix-cache \\\n"
            if config.speculative_algorithm:
                script += f"    --speculative-algorithm {config.speculative_algorithm} \\\n"
                script += f"    --speculative-num-steps {config.speculative_num_steps} \\\n"
            script += "    --trust-remote-code\n"
        
        else:
            script += f"""# Prefill Server
export SGLANG_DEEPEP_MODE=normal
python -m sglang.launch_server \\
    --model-path ${{MODEL_PATH}} \\
    --host ${{HOST}} \\
    --port 8000 \\
    --disaggregation-mode prefill \\
    --disaggregation-bootstrap-port 8998 \\
    --tp-size {config.tp_size} \\
    --dp-size {config.dp_size} \\
    --mem-fraction-static {config.mem_fraction_static} \\
    --max-running-requests 16 \\
    --attention-backend {config.attention_backend} \\
    --device {config.device} \\
    --dtype {config.dtype} \\
    --disable-radix-cache \\
    --trust-remote-code &

sleep 120

# Decode Server
export SGLANG_DEEPEP_MODE=low_latency
python -m sglang.launch_server \\
    --model-path ${{MODEL_PATH}} \\
    --host ${{HOST}} \\
    --port 8001 \\
    --disaggregation-mode decode \\
    --tp-size {config.tp_size} \\
    --dp-size {config.dp_size} \\
    --mem-fraction-static {config.mem_fraction_static} \\
    --max-running-requests {config.max_running_requests} \\
    --attention-backend {config.attention_backend} \\
    --device {config.device} \\
    --dtype {config.dtype} \\
    --cuda-graph-bs {' '.join(map(str, config.cuda_graph_bs))} \\
    --disable-radix-cache \\
    --trust-remote-code &

sleep 60

# Router
python -m sglang_router.launch_router \\
    --pd-disaggregation \\
    --policy cache_aware \\
    --prefill http://${{HOST}}:8000 8998 \\
    --decode http://${{HOST}}:8001 \\
    --host 127.0.0.1 \\
    --port 6688 \\
    --mini-lb
"""
        
        with open(output_path, 'w') as f:
            f.write(script)
        
        print(f"Deployment script saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-tune SGLang server parameters")
    parser.add_argument("--model-path", required=True, help="Path to the model")
    parser.add_argument("--tuning-mode", choices=[m.value for m in TuningMode], 
                        default="low_latency", help="Tuning mode")
    parser.add_argument("--deploy-mode", choices=[m.value for m in DeployMode],
                        default="mixed", help="Deployment mode")
    parser.add_argument("--input-len", type=int, default=3500, help="Input sequence length")
    parser.add_argument("--output-len", type=int, default=1500, help="Output sequence length")
    parser.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--device", choices=["cuda", "npu"], default="cuda", help="Device type")
    parser.add_argument("--quantization", help="Quantization type (e.g., modelslim, awq)")
    parser.add_argument("--output-dir", default="./tune_results", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tuner = AutoTuneSkill()
    
    tuning_mode = TuningMode(args.tuning_mode)
    deploy_mode = DeployMode(args.deploy_mode)
    
    print(f"Starting auto-tuning for {tuning_mode.value} mode with {deploy_mode.value} deployment")
    print(f"Model: {args.model_path}")
    print(f"TP: {args.tp_size}, Input: {args.input_len}, Output: {args.output_len}")
    
    result = tuner.tune(
        model_path=args.model_path,
        tuning_mode=tuning_mode,
        deploy_mode=deploy_mode,
        input_len=args.input_len,
        output_len=args.output_len,
        tp_size=args.tp_size,
        device=args.device,
        quantization=args.quantization,
    )
    
    print(f"\n{'='*80}")
    print("TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Configuration:")
    print(f"  mem_fraction_static: {result.best_config.mem_fraction_static}")
    print(f"  max_running_requests: {result.best_config.max_running_requests}")
    print(f"  cuda_graph_bs: {result.best_config.cuda_graph_bs}")
    print(f"  dp_size: {result.best_config.dp_size}")
    print(f"  enable_mtp: {result.best_config.enable_mtp}")
    print(f"  speculative_num_steps: {result.best_config.speculative_num_steps}")
    print(f"\nBest Metrics:")
    print(f"  p99_tpot_ms: {result.best_metrics.p99_tpot_ms:.2f}")
    print(f"  mean_tpot_ms: {result.best_metrics.tpot_:ms:.2f}")
    print(f"  ttft_ms: {result.best_metrics.ttft_ms:.2f}")
    print(f"  request_throughput: {result.best_metrics.request_throughput:.2f} req/s")
    
    result_path = os.path.join(args.output_dir, "tuning_results.json")
    script_path = os.path.join(args.output_dir, "deploy.sh")
    
    tuner.save_results(result, result_path)
    tuner.generate_deployment_script(result, script_path)


if __name__ == "__main__":
    main()

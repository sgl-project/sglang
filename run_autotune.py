#!/usr/bin/env python3
"""
Auto-tuning script for SGLang model deployment.

Usage:
python run_autotune.py --model-path /path/to/model --mode low_latency
"""

import sys
import os

# Add the python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from autotune_skill import AutoTuneSkill, TuningMode, DeployMode


def main():
    parser = sys.modules['argparse']
    arg_parser = parser.ArgumentParser(description="Auto-tune SGLang server parameters")
    arg_parser.add_argument("--model-path", required=True, help="Path to the model")
    arg_parser.add_argument("--mode", choices=[m.value for m in TuningMode], 
                           default="low_latency", help="Tuning mode")
    arg_parser.add_argument("--deploy", choices=[m.value for m in DeployMode],
                           default="mixed", help="Deployment mode")
    arg_parser.add_argument("--input-len", type=int, default=3500, help="Input sequence length")
    arg_parser.add_argument("--output-len", type=int, default=1500, help="Output sequence length")
    arg_parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    arg_parser.add_argument("--device", choices=["cuda", "npu"], default="cuda", help="Device type")
    arg_parser.add_argument("--quantization", help="Quantization type (e.g., modelslim, awq)")
    arg_parser.add_argument("--output-dir", default="./tune_results", help="Output directory")
    
    args = arg_parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tuner = AutoTuneSkill()
    
    tuning_mode = TuningMode(args.mode)
    deploy_mode = DeployMode(args.deploy)
    
    print(f"Starting auto-tuning for {tuning_mode.value} mode with {deploy_mode.value} deployment")
    print(f"Model: {args.model_path}")
    print(f"TP: {args.tp}, Input: {args.input_len}, Output: {args.output_len}")
    
    result = tuner.tune(
        model_path=args.model_path,
        tuning_mode=tuning_mode,
        deploy_mode=deploy_mode,
        input_len=args.input_len,
        output_len=args.output_len,
        tp_size=args.tp,
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
    print(f"  mean_tpot_ms: {result.best_metrics.tpot_ms:.2f}")
    print(f"  ttft_ms: {result.best_metrics.ttft_ms:.2f}")
    print(f"  request_throughput: {result.best_metrics.request_throughput:.2f} req/s")
    
    result_path = os.path.join(args.output_dir, "tuning_results.json")
    script_path = os.path.join(args.output_dir, "deploy.sh")
    
    tuner.save_results(result, result_path)
    tuner.generate_deployment_script(result, script_path)


if __name__ == "__main__":
    main()

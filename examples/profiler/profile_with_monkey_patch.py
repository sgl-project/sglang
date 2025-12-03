"""
Profile shapes by monkey-patching the model's forward method.

This approach directly wraps the model's computation to capture shapes.

Usage:
    python profile_with_monkey_patch.py \\
        --model-path Qwen/Qwen2.5-14B-Instruct \\
        --tp-size 8
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from torch_shape_logger_rank import CompactRankAwareShapeLogger, get_current_rank
import dataclasses


def profile_with_wrapper(server_args, num_prompts=3, max_tokens=100, output_file="shapes.jsonl"):
    """Profile by wrapping model forward."""
    
    current_rank = get_current_rank()
    
    # Only rank 0 prints
    if current_rank == 0:
        print("=" * 80)
        print("Model Shape Profiler (Monkey Patch Method)")
        print("=" * 80)
        print(f"Model: {server_args.model_path}")
        print(f"TP Size: {server_args.tp_size}")
        print(f"Output: {output_file}")
        print("=" * 80)
    
    # Initialize engine
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    
    if current_rank == 0:
        print("\n✓ Engine initialized")
        print("\nStarting profiling...")
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Explain quantum computing:",
    ][:num_prompts]
    
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }
    
    # Profile with logger active
    logger = CompactRankAwareShapeLogger(
        output_file=output_file,
        verbose=False,
        only_rank=0,  # Only GPU 0
    )
    
    try:
        with logger:
            outputs = llm.generate(prompts, sampling_params)
        
        if current_rank == 0:
            print(f"\n✓ Generation completed")
            
            if logger.should_log:
                summary = logger.get_summary()
                print(f"\nCaptured {summary['total_operations']} operations on rank {current_rank}")
                
                if summary['total_operations'] == 0:
                    print("\n⚠ WARNING: No operations captured!")
                    print("This is expected - the logger can't see operations inside the engine.")
                    print("\nTo profile engine internals, you need to:")
                    print("1. Add profiling code inside SGLang's model implementation")
                    print("2. Or use torch profiler with kineto")
                    print("3. Or use NVIDIA Nsight Systems")
                
    finally:
        llm.shutdown()
        if current_rank == 0:
            print("\n✓ Engine shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--output-file", type=str, default="shapes_patched.jsonl")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)
    
    args = parser.parse_args()
    
    output_file = args.output_file
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    
    delattr(args, "output_file")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    
    server_args = ServerArgs.from_cli_args(args)
    
    if server_args.model_path is None:
        print("Error: --model-path required")
        sys.exit(1)
    
    profile_with_wrapper(server_args, num_prompts, max_tokens, output_file)

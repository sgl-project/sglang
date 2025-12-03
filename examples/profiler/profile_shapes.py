#!/usr/bin/env python3
"""
Standalone Shape Profiler for SGLang

A debug tool to profile tensor shapes during model inference without modifying core SGLang code.
Takes the same parameters as launch_server.

⚠️  IMPORTANT: This tool only works with TP=1 (single GPU).
    With TP > 1, operations happen in worker processes and 0 operations will be captured (expected).
    For multi-GPU profiling, use NVIDIA Nsight Systems or AMD ROCm Profiler.

Usage:
    # Single GPU (captures all operations)
    python profile_shapes.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --tp-size 1 \\
        --num-prompts 3 \\
        --max-tokens 50
    
    # Multi-GPU (will capture 0 operations - use external profilers instead)
    python profile_shapes.py \\
        --model-path Qwen/Qwen2.5-14B-Instruct \\
        --tp-size 8 \\
        --num-prompts 3 \\
        --max-tokens 50

This will profile shapes and save to a JSONL file.
"""

import argparse
import dataclasses
import os
import sys

import sglang as sgl
from sglang.srt.server_args import ServerArgs

# Add profiler utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_shape_logger import CompactShapeLogger, analyze_shape_log


def profile_shapes(
    server_args: ServerArgs,
    output_file: str = "shapes.jsonl",
    num_prompts: int = 3,
    max_tokens: int = 100,
    analyze: bool = True,
):
    """
    Profile tensor shapes during model inference.
    
    This is a STANDALONE DEBUG TOOL that doesn't modify SGLang's core code.
    It uses PyTorch dispatch mode to intercept operations in the current process.
    
    LIMITATION: Only works with TP=1. With TP > 1, operations happen in worker 
    processes and cannot be intercepted (0 operations will be captured).
    For multi-GPU profiling, use NVIDIA Nsight Systems or AMD ROCm Profiler.
    """
    print("=" * 80)
    print("SGLang Shape Profiler (Standalone Debug Tool)")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"TP Size: {server_args.tp_size}")
    print(f"Output: {output_file}")
    print("=" * 80)
    print()
    
    # Sample prompts
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about mountains:",
        "What is the meaning of life?",
        "Describe the process of photosynthesis:",
    ]
    prompts = all_prompts[:num_prompts]
    
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }
    
    print("Initializing engine...")
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    print("✓ Engine initialized\n")
    
    # Profile with shape logger
    print("Starting profiling...")
    logger = CompactShapeLogger(output_file=output_file, verbose=False)
    
    try:
        with logger:
            outputs = llm.generate(prompts, sampling_params)
        
        print("✓ Generation completed\n")
        
        # Print results
        print("=" * 80)
        print("Generation Results")
        print("=" * 80)
        for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
            generated = output['text']
            display = generated[:100] + '...' if len(generated) > 100 else generated
            print(f"\n[{i}] {prompt}")
            print(f"    → {display}")
        
        # Summary
        summary = logger.get_summary()
        print("\n" + "=" * 80)
        print("Profiling Summary")
        print("=" * 80)
        print(f"Total operations: {summary['total_operations']:,}")
        print(f"Unique operations: {summary['unique_operations']}")
        
        if summary['total_operations'] == 0:
            print("\n⚠ WARNING: No operations captured!")
            print("\nThis is EXPECTED with TP > 1 because:")
            print("  • Tensor operations happen in worker processes")
            print("  • PyTorch dispatch mode only intercepts the current process")
            print("  • Worker processes execute independently")
            print("\nTo profile tensor operations with TP > 1, use:")
            print("  • NVIDIA Nsight Systems: nsys profile python ...")
            print("  • AMD ROCm Profiler: rocprof python ...")
            print("  • PyTorch Profiler with multiprocessing support")
            print("\nTo use this tool, run with TP=1:")
            print(f"  python profile_shapes.py --model-path <smaller-model> --tp-size 1")
        else:
            if summary['operation_counts']:
                print(f"\nTop 10 operations:")
                for op, count in sorted(summary['operation_counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {count:6,d} : {op[:70]}")
            
            # Analyze
            if analyze:
                print("\n" + "=" * 80)
                print("Detailed Analysis")
                print("=" * 80)
                analyze_shape_log(output_file)
        
        # Check file
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"\n✓ Shapes saved to: {output_file} ({size:,} bytes)")
            
            if size > 0:
                print("\nAnalyze with:")
                print(f"  python analyze_shapes.py {output_file} --show-shapes")
        
    finally:
        llm.shutdown()
        print("\n✓ Engine shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Profile tensor shapes during SGLang inference (standalone debug tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU (RECOMMENDED - captures all operations)
  python profile_shapes.py \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --tp-size 1 \\
      --num-prompts 5 \\
      --max-tokens 100
  
  # Multi-GPU (will capture 0 operations - expected behavior)
  python profile_shapes.py \\
      --model-path Qwen/Qwen2.5-14B-Instruct \\
      --tp-size 8 \\
      --num-prompts 3

Important Limitation:
  This tool ONLY works with TP=1. With TP > 1, tensor operations happen in 
  worker processes that PyTorch dispatch mode cannot intercept, resulting in
  0 captured operations (this is expected behavior, not a bug).

  For multi-GPU profiling, use:
    • NVIDIA Nsight Systems: nsys profile python ...
    • AMD ROCm Profiler: rocprof python ...
    • PyTorch Profiler with distributed support
        """
    )
    
    # Add all ServerArgs
    ServerArgs.add_cli_args(parser)
    
    # Add profiler-specific args
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file (default: model_name_shapes.jsonl)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of prompts to process (default: 3)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens per prompt (default: 50)"
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip analysis after profiling"
    )
    
    args = parser.parse_args()
    
    # Extract profiler args
    output = args.output
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    analyze = not args.no_analyze
    
    delattr(args, "output")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    delattr(args, "no_analyze")
    
    # Parse ServerArgs
    server_args = ServerArgs.from_cli_args(args)
    
    if server_args.model_path is None:
        print("Error: --model-path is required")
        sys.exit(1)
    
    # Default output filename
    if output is None:
        model_name = server_args.model_path.split('/')[-1].replace('-', '_').lower()
        output = f"{model_name}_tp{server_args.tp_size}_shapes.jsonl"
    
    profile_shapes(server_args, output, num_prompts, max_tokens, analyze)


if __name__ == "__main__":
    main()

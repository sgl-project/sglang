"""
Run Qwen 14B inference with kernel shape logging using torch dispatch.

This script demonstrates how to use the ShapeLogger to capture all tensor operation
shapes during model inference.

Usage:
    # Basic usage with Qwen 14B
    python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct
    
    # With custom output file
    python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct --output-file qwen14b_shapes.jsonl
    
    # Verbose mode (prints shapes to console)
    python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct --verbose
    
    # Compact mode (smaller log files)
    python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct --compact
    
    # Control number of prompts and tokens
    python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct --num-prompts 2 --max-tokens 50
"""

import argparse
import dataclasses
import os
import sys

# Add parent directory to path to import torch_shape_logger
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from torch_shape_logger import CompactShapeLogger, ShapeLogger, analyze_shape_log


def main(
    server_args: ServerArgs,
    output_file: str = "qwen_kernel_shapes.jsonl",
    verbose: bool = False,
    compact: bool = False,
    num_prompts: int = 4,
    max_tokens: int = 100,
    analyze_after: bool = True,
):
    """
    Run Qwen model with shape logging.

    Args:
        server_args: Server configuration arguments
        output_file: Path to output JSONL file for shape logs
        verbose: If True, print shapes to console during execution
        compact: If True, use compact logging (shapes only, no dtype/device)
        num_prompts: Number of prompts to process
        max_tokens: Maximum tokens to generate per prompt
        analyze_after: If True, analyze the log file after inference
    """
    print("=" * 80)
    print("Qwen 14B Shape Logger")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"Output file: {output_file}")
    print(f"Compact mode: {compact}")
    print(f"Verbose mode: {verbose}")
    print("=" * 80)

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

    # Create sampling params
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }

    print(f"\nProcessing {len(prompts)} prompts with max {max_tokens} tokens each...\n")

    # Create the LLM engine (this happens outside the shape logger)
    print("Initializing engine...")
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    print("Engine initialized!\n")

    # Choose the appropriate logger
    LoggerClass = CompactShapeLogger if compact else ShapeLogger

    # Run inference with shape logging
    print("Starting inference with shape logging...")
    print("(This may take a while depending on model size and prompt length)\n")

    try:
        with LoggerClass(output_file=output_file, verbose=verbose) as logger:
            outputs = llm.generate(prompts, sampling_params)

        # Print the outputs
        print("\n" + "=" * 80)
        print("Generation Results")
        print("=" * 80)
        for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
            print(f"\n[{i}/{len(prompts)}]")
            print(f"Prompt: {prompt}")
            print(f"Generated: {output['text'][:200]}{'...' if len(output['text']) > 200 else ''}")
            print("-" * 80)

        # Print summary
        summary = logger.get_summary()
        print("\n" + "=" * 80)
        print("Shape Logging Summary")
        print("=" * 80)
        print(f"Total operations captured: {summary['total_operations']}")
        print(f"Unique operations: {summary['unique_operations']}")
        print(f"\nTop 10 most frequent operations:")
        sorted_ops = sorted(summary['operation_counts'].items(), key=lambda x: x[1], reverse=True)
        for op_name, count in sorted_ops[:10]:
            print(f"  {count:8d} : {op_name}")

        # Analyze the log file if requested
        if analyze_after:
            print("\n" + "=" * 80)
            print("Detailed Analysis")
            print("=" * 80)
            analyze_shape_log(output_file)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Logs may be incomplete.")
    except Exception as e:
        print(f"\n\nError during inference: {e}")
        raise
    finally:
        # Cleanup
        llm.shutdown()
        print("\nEngine shutdown complete.")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Qwen inference with kernel shape logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct
  
  # Smaller model for testing
  python qwen_shape_logger.py --model Qwen/Qwen2.5-7B-Instruct
  
  # With custom settings
  python qwen_shape_logger.py --model Qwen/Qwen2.5-14B-Instruct \\
      --output-file my_shapes.jsonl \\
      --num-prompts 2 \\
      --max-tokens 50 \\
      --compact
        """,
    )

    # Add ServerArgs CLI arguments
    ServerArgs.add_cli_args(parser)

    # Add shape logger specific arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default="qwen_kernel_shapes.jsonl",
        help="Output file for shape logs (JSONL format)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print shapes to console during execution",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use compact logging (shapes only, smaller files)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to process (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate per prompt (default: 100)",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip analysis after inference",
    )

    args = parser.parse_args()

    # Extract shape logger args
    output_file = args.output_file
    verbose = args.verbose
    compact = args.compact
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    analyze_after = not args.no_analyze

    # Remove our custom args to avoid conflicts with ServerArgs
    delattr(args, "output_file")
    delattr(args, "verbose")
    delattr(args, "compact")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    delattr(args, "no_analyze")

    # Parse server args
    server_args = ServerArgs.from_cli_args(args)

    # Set default model if not provided
    if server_args.model_path is None:
        print("No model specified, using default: Qwen/Qwen2.5-7B-Instruct")
        server_args.model_path = "Qwen/Qwen2.5-7B-Instruct"

    main(
        server_args=server_args,
        output_file=output_file,
        verbose=verbose,
        compact=compact,
        num_prompts=num_prompts,
        max_tokens=max_tokens,
        analyze_after=analyze_after,
    )

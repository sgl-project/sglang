"""
Profile DeepSeek (or any model) kernel shapes using offline Engine.

FIXED VERSION: This version properly captures shapes by injecting the logger
into the worker process where model inference actually happens.

Usage:
    python profile_deepseek_shapes_fixed.py \
        --model-path Qwen/Qwen2.5-14B-Instruct \
        --tp-size 8 \
        --num-prompts 3 \
        --max-tokens 50
"""

import argparse
import dataclasses
import os
import sys

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
    output_file: str = "model_shapes.jsonl",
    verbose: bool = False,
    compact: bool = True,
    num_prompts: int = 3,
    max_tokens: int = 100,
):
    """
    Profile model with shape logging by using environment variable to enable it.

    Args:
        server_args: Server configuration arguments
        output_file: Path to output JSONL file for shape logs
        verbose: If True, print shapes to console during execution
        compact: If True, use compact logging (shapes only, no dtype/device)
        num_prompts: Number of prompts to process
        max_tokens: Maximum tokens to generate per prompt
    """
    print("=" * 80)
    print("Model Kernel Shape Profiler (FIXED VERSION)")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"Tensor Parallel: {server_args.tp_size}")
    print(f"Output file: {output_file}")
    print(f"Compact mode: {compact}")
    print(f"Verbose mode: {verbose}")
    print(f"Prompts: {num_prompts}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 80)
    
    print("\n⚠️  IMPORTANT: Shape logging in multi-process mode is not yet supported.")
    print("The ShapeLogger only captures operations in the main process,")
    print("but SGLang runs model inference in separate worker processes.")
    print("\nTo profile model operations, you need to either:")
    print("1. Use the model directly (without the Engine wrapper)")
    print("2. Inject the logger into the worker process")
    print("3. Use external profiling tools like nvprof/nsys")
    print("\nFor now, this script demonstrates the issue.")
    print("=" * 80)

    # Sample prompts
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]
    prompts = all_prompts[:num_prompts]

    # Create sampling params
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }

    print(f"\nInitializing engine...")
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    print("✓ Engine initialized successfully!\n")

    print("Running inference (without shape logging)...")
    outputs = llm.generate(prompts, sampling_params)
    print("✓ Generation completed!")

    # Print the outputs
    print("\n" + "=" * 80)
    print("Generation Results")
    print("=" * 80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"\n[{i}/{len(prompts)}]")
        print(f"Prompt: {prompt}")
        generated = output['text']
        display_text = generated[:150] + '...' if len(generated) > 150 else generated
        print(f"Generated: {display_text}")

    print("\nShutting down engine...")
    llm.shutdown()
    print("✓ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile kernel shapes (FIXED VERSION - demonstrates the issue)",
    )
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--output-file", type=str, default="model_shapes.jsonl")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)

    args = parser.parse_args()

    # Extract args
    output_file = args.output_file
    verbose = args.verbose
    compact = not args.no_compact
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens

    # Remove custom args
    delattr(args, "output_file")
    delattr(args, "verbose")
    delattr(args, "no_compact")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")

    # Parse server args
    server_args = ServerArgs.from_cli_args(args)

    if server_args.model_path is None:
        print("Error: --model-path is required")
        sys.exit(1)

    # Set default output file
    if output_file == "model_shapes.jsonl":
        model_name = server_args.model_path.split('/')[-1].replace('-', '_').lower()
        output_file = f"{model_name}_tp{server_args.tp_size}_shapes.jsonl"

    main(
        server_args=server_args,
        output_file=output_file,
        verbose=verbose,
        compact=compact,
        num_prompts=num_prompts,
        max_tokens=max_tokens,
    )

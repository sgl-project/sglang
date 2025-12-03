"""
Profile DeepSeek (or any model) kernel shapes using offline Engine.

This is the RECOMMENDED approach for profiling server-side model inference.
It uses the same Engine that powers the server but runs locally, capturing
all actual model inference operations.

Usage:
    # Profile DeepSeek V3 with TP=8 (no download if model is cached)
    python profile_deepseek_shapes.py \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 \\
        --output-file deepseek_v3_shapes.jsonl
    
    # With custom settings
    python profile_deepseek_shapes.py \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 \\
        --output-file deepseek_shapes.jsonl \\
        --num-prompts 3 \\
        --max-tokens 50 \\
        --compact
    
    # Then analyze
    python analyze_shapes.py deepseek_v3_shapes.jsonl

Benefits over server profiling:
- Captures actual model inference operations (not just HTTP client ops)
- No need to run a separate server
- Full control over profiling parameters
- Works with any model that SGLang supports
"""

import argparse
import dataclasses
import os
import sys

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from torch_shape_logger import CompactShapeLogger, ShapeLogger, analyze_shape_log


def main(
    server_args: ServerArgs,
    output_file: str = "model_shapes.jsonl",
    verbose: bool = False,
    compact: bool = True,
    num_prompts: int = 3,
    max_tokens: int = 100,
    analyze_after: bool = True,
):
    """
    Profile model with shape logging.

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
    print("Model Kernel Shape Profiler")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"Tensor Parallel: {server_args.tp_size}")
    print(f"Output file: {output_file}")
    print(f"Compact mode: {compact}")
    print(f"Verbose mode: {verbose}")
    print(f"Prompts: {num_prompts}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 80)

    # Sample prompts for different scenarios
    all_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about mountains:",
        "What is the meaning of life?",
        "Describe the process of photosynthesis:",
        "Tell me about the history of artificial intelligence.",
        "What are the key principles of machine learning?",
    ]
    prompts = all_prompts[:num_prompts]

    # Create sampling params
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": max_tokens,
    }

    print(f"\nInitializing engine...")
    print("(Model loading may take a few minutes...)")
    
    # Create the LLM engine
    llm = None
    try:
        llm = sgl.Engine(**dataclasses.asdict(server_args))
        print("✓ Engine initialized successfully!\n")
    except Exception as e:
        print(f"\n✗ Error initializing engine: {e}")
        print("\nCommon issues:")
        print("  - Model not found: Check model path")
        print("  - Out of memory: Try smaller TP size or add --mem-fraction-static 0.8")
        print("  - CUDA error: Check GPU availability")
        raise

    # Check if environment variable profiling is enabled
    import os
    env_profiling = os.environ.get("SGLANG_PROFILE_SHAPES", "0") == "1"
    
    if env_profiling:
        print("="*80)
        print("DETECTED: Environment variable profiling enabled")
        print(f"  SGLANG_PROFILE_SHAPES_RANK = {os.environ.get('SGLANG_PROFILE_SHAPES_RANK', '0')}")
        print(f"  SGLANG_PROFILE_SHAPES_FILE = {os.environ.get('SGLANG_PROFILE_SHAPES_FILE', 'shapes.jsonl')}")
        print("Shape logging will happen inside TP workers, not in main process.")
        print("="*80)
        print()
        
        # Don't create main process logger, let workers handle it
        print(f"Starting generation for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        print(f"✓ Generation completed!")
        
        # Check the env var file
        env_file = os.environ.get('SGLANG_PROFILE_SHAPES_FILE', 'shapes.jsonl')
        if os.path.exists(env_file):
            file_size = os.path.getsize(env_file)
            print(f"\n✓ Shape log file created by workers: {env_file} ({file_size:,} bytes)")
        else:
            print(f"\n⚠ Warning: Expected shape log file not found: {env_file}")
        
        logger = None
    else:
        # Original approach (main process only, won't work with TP workers)
        print("="*80)
        print("Starting inference with shape logging (MAIN PROCESS ONLY)...")
        print("WARNING: This will NOT capture shapes in TP worker processes!")
        print("To profile TP workers, use environment variables instead:")
        print("  export SGLANG_PROFILE_SHAPES=1")
        print("  export SGLANG_PROFILE_SHAPES_RANK=0")
        print("  export SGLANG_PROFILE_SHAPES_FILE=gpu0_shapes.jsonl")
        print("="*80)
        print()

        # Choose the appropriate logger
        LoggerClass = CompactShapeLogger if compact else ShapeLogger
        logger = LoggerClass(output_file=output_file, verbose=verbose)
        with logger:
            print(f"Starting generation for {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params)
            print(f"✓ Generation completed!")

    # Print the outputs
    print("\n" + "=" * 80)
    print("Generation Results")
    print("=" * 80)
    for i, (prompt, output) in enumerate(zip(prompts, outputs), 1):
        print(f"\n[{i}/{len(prompts)}]")
        print(f"Prompt: {prompt}")
        generated = output['text']
        # Show first 150 chars
        display_text = generated[:150] + '...' if len(generated) > 150 else generated
        print(f"Generated: {display_text}")
        print(f"Total length: {len(generated)} characters")
        print("-" * 80)

    # Print summary (only if we have a logger from main process)
    if logger:
        summary = logger.get_summary()
        print("\n" + "=" * 80)
        print("Shape Logging Summary")
        print("=" * 80)
        print(f"Total operations captured: {summary['total_operations']:,}")
        print(f"Unique operations: {summary['unique_operations']}")
        
        if summary['operation_counts']:
            print(f"\nTop 10 most frequent operations:")
            sorted_ops = sorted(
                summary['operation_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for op_name, count in sorted_ops[:10]:
                # Shorten operation name if too long
                display_name = op_name if len(op_name) <= 60 else op_name[:57] + "..."
                print(f"  {count:8,d} : {display_name}")

        # Analyze the log file if requested
        if analyze_after and summary['total_operations'] > 0:
            print("\n" + "=" * 80)
            print("Detailed Analysis")
            print("=" * 80)
            analyze_shape_log(output_file)

    try:
        pass  # Cleanup section
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user. Logs may be incomplete.")
        if logger and logger.call_count > 0:
            print(f"Captured {logger.call_count} operations before interrupt.")
    except Exception as e:
        print(f"\n\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        if logger and logger.call_count > 0:
            print(f"Captured {logger.call_count} operations before error.")
        raise
    finally:
        # Cleanup
        if llm:
            print("\nShutting down engine...")
            try:
                llm.shutdown()
                print("✓ Engine shutdown complete.")
            except Exception as e:
                print(f"Warning: Error during engine shutdown: {e}")
        
        # Check if log file has content
        import os
        # Check the correct file based on whether env vars are used
        check_file = os.environ.get('SGLANG_PROFILE_SHAPES_FILE', output_file) if env_profiling else output_file
        
        if os.path.exists(check_file):
            file_size = os.path.getsize(check_file)
            if file_size > 0:
                print(f"\n✓ Log file created: {check_file} ({file_size:,} bytes)")
            else:
                print(f"\n⚠ Warning: Log file is empty: {check_file}")
                print("This may indicate that:")
                print("  1. No tensor operations were captured")
                print("  2. The process was killed before data could be written")
                print("  3. The ShapeLogger was not properly activated")
        else:
            print(f"\n⚠ Warning: Log file was not created: {check_file}")
        
        print("\n" + "=" * 80)
        print("Next Steps")
        print("=" * 80)
        print(f"1. View detailed analysis:")
        print(f"   python analyze_shapes.py {check_file}")
        print(f"\n2. Filter specific operations:")
        print(f"   python analyze_shapes.py {check_file} --filter-op attention")
        print(f"\n3. Find large tensor operations:")
        print(f"   python analyze_shapes.py {check_file} --min-elements 1000000")
        print(f"\n4. Export timeline:")
        print(f"   python analyze_shapes.py {check_file} --timeline timeline.csv")
        print("=" * 80)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile kernel shapes during model inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DeepSeek V3 with TP=8
  python profile_deepseek_shapes.py \\
      --model-path deepseek-ai/DeepSeek-V3 \\
      --tp-size 8
  
  # Qwen with TP=4 and verbose output
  python profile_deepseek_shapes.py \\
      --model-path Qwen/Qwen2.5-72B-Instruct \\
      --tp-size 4 \\
      --verbose
  
  # Quick test with small model
  python profile_deepseek_shapes.py \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --tp-size 1 \\
      --num-prompts 2 \\
      --max-tokens 20
  
  # Profile with existing cached model (no download)
  python profile_deepseek_shapes.py \\
      --model-path /path/to/cached/model \\
      --tp-size 8 \\
      --output-file custom_shapes.jsonl

Note: 
  - Uses the same model path/cache as SGLang server
  - No additional model download if already cached
  - Captures ALL model inference operations
  - TP size must match your GPU configuration
        """,
    )

    # Add ServerArgs CLI arguments
    ServerArgs.add_cli_args(parser)

    # Add shape logger specific arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default="model_shapes.jsonl",
        help="Output file for shape logs (JSONL format)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print shapes to console during execution (very verbose!)",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Disable compact logging (includes dtype/device info, larger files)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="Number of prompts to process (default: 3)",
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
        help="Skip automatic analysis after inference",
    )

    args = parser.parse_args()

    # Extract shape logger args
    output_file = args.output_file
    verbose = args.verbose
    compact = not args.no_compact
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    analyze_after = not args.no_analyze

    # Remove our custom args to avoid conflicts with ServerArgs
    delattr(args, "output_file")
    delattr(args, "verbose")
    delattr(args, "no_compact")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    delattr(args, "no_analyze")

    # Parse server args
    server_args = ServerArgs.from_cli_args(args)

    # Validate model path
    if server_args.model_path is None:
        print("Error: --model-path is required")
        print("\nExamples:")
        print("  --model-path deepseek-ai/DeepSeek-V3")
        print("  --model-path Qwen/Qwen2.5-14B-Instruct")
        print("  --model-path /path/to/local/model")
        sys.exit(1)

    # Set default output file based on model name if not specified
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
        analyze_after=analyze_after,
    )

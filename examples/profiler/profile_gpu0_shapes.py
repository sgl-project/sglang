"""
Profile GPU 0 shapes during model inference with Tensor Parallelism.

This script profiles ONLY GPU 0 (rank 0) when running with --tp-size 8,
capturing all tensor shapes on that specific GPU.

Usage:
    # Profile DeepSeek V3 with TP=8, logging only GPU 0 shapes
    python profile_gpu0_shapes.py \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 \\
        --output-file gpu0_shapes.jsonl
    
    # With custom settings
    python profile_gpu0_shapes.py \\
        --model-path deepseek-ai/DeepSeek-V3 \\
        --tp-size 8 \\
        --output-file gpu0_shapes.jsonl \\
        --num-prompts 3 \\
        --max-tokens 50
    
    # Then analyze GPU 0 shapes
    python analyze_shapes.py gpu0_shapes.jsonl --show-shapes

Key Features:
- Only logs shapes on GPU 0 (rank 0)
- Works with any TP size (TP=8, TP=4, etc.)
- Captures actual tensor shapes during inference
- Zero overhead on other GPUs (rank 1-7)
"""

import argparse
import dataclasses
import os
import sys

# Add profiler to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from torch_shape_logger_rank import CompactRankAwareShapeLogger, RankAwareShapeLogger, get_current_rank


def main(
    server_args: ServerArgs,
    output_file: str = "gpu0_shapes.jsonl",
    verbose: bool = False,
    compact: bool = True,
    num_prompts: int = 3,
    max_tokens: int = 100,
    analyze_after: bool = True,
    only_rank: int = 0,
):
    """
    Profile model with rank-aware shape logging.

    Args:
        server_args: Server configuration arguments
        output_file: Path to output JSONL file for shape logs
        verbose: If True, print shapes to console during execution
        compact: If True, use compact logging (shapes only, no dtype/device)
        num_prompts: Number of prompts to process
        max_tokens: Maximum tokens to generate per prompt
        analyze_after: If True, analyze the log file after inference
        only_rank: Only log on this rank (default: 0 for GPU 0)
    """
    current_rank = get_current_rank()
    
    if current_rank == 0:
        print("=" * 80)
        print("GPU 0 (Rank 0) Kernel Shape Profiler")
        print("=" * 80)
        print(f"Model: {server_args.model_path}")
        print(f"Tensor Parallel: {server_args.tp_size}")
        print(f"Target GPU: GPU {only_rank} (Rank {only_rank})")
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

    if current_rank == 0:
        print(f"\nInitializing engine with TP={server_args.tp_size}...")
        print("(Model loading may take a few minutes...)")
    
    # Create the LLM engine
    llm = None
    try:
        llm = sgl.Engine(**dataclasses.asdict(server_args))
        if current_rank == 0:
            print("✓ Engine initialized successfully!\n")
    except Exception as e:
        if current_rank == 0:
            print(f"\n✗ Error initializing engine: {e}")
            print("\nCommon issues:")
            print("  - Model not found: Check model path")
            print("  - Out of memory: Try smaller TP size or add --mem-fraction-static 0.8")
            print("  - CUDA error: Check GPU availability")
        raise

    # Choose the appropriate logger
    LoggerClass = CompactRankAwareShapeLogger if compact else RankAwareShapeLogger

    # Run inference with shape logging (only on specified rank)
    if current_rank == 0:
        print("="*80)
        print(f"Starting inference with shape logging on GPU {only_rank}...")
        print("(This will capture all kernel operations on GPU 0 during model inference)")
        print("="*80)
        print()

    logger = None
    try:
        # Create logger that only logs on the specified rank
        logger = LoggerClass(
            output_file=output_file,
            verbose=verbose,
            only_rank=only_rank,
        )
        
        with logger:
            if current_rank == 0:
                print(f"Starting generation for {len(prompts)} prompts...")
            outputs = llm.generate(prompts, sampling_params)
            if current_rank == 0:
                print(f"✓ Generation completed!")

        # Print the outputs (only on rank 0)
        if current_rank == 0:
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

        # Print summary (only if logging on this rank)
        if logger.should_log:
            summary = logger.get_summary()
            print("\n" + "=" * 80)
            print(f"Shape Logging Summary (Rank {current_rank})")
            print("=" * 80)
            print(f"Total operations captured: {summary['total_operations']:,}")
            print(f"Unique operations: {summary['unique_operations']}")
            
            if summary['operation_counts']:
                print(f"\nTop 10 most frequent operations on GPU {only_rank}:")
                sorted_ops = sorted(
                    summary['operation_counts'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for op_name, count in sorted_ops[:10]:
                    # Shorten operation name if too long
                    display_name = op_name if len(op_name) <= 60 else op_name[:57] + "..."
                    print(f"  {count:8,d} : {display_name}")

            # Analyze the log file if requested (only on rank 0)
            if current_rank == 0 and analyze_after and summary['total_operations'] > 0:
                print("\n" + "=" * 80)
                print(f"Detailed Analysis (GPU {only_rank} shapes)")
                print("=" * 80)
                from torch_shape_logger import analyze_shape_log
                analyze_shape_log(output_file)

    except KeyboardInterrupt:
        if current_rank == 0:
            print("\n\n✗ Interrupted by user. Logs may be incomplete.")
        if logger and logger.should_log and logger.call_count > 0:
            print(f"[Rank {current_rank}] Captured {logger.call_count} operations before interrupt.")
    except Exception as e:
        if current_rank == 0:
            print(f"\n\n✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
        if logger and logger.should_log and logger.call_count > 0:
            print(f"[Rank {current_rank}] Captured {logger.call_count} operations before error.")
        raise
    finally:
        # Cleanup
        if llm:
            if current_rank == 0:
                print("\nShutting down engine...")
            try:
                llm.shutdown()
                if current_rank == 0:
                    print("✓ Engine shutdown complete.")
            except Exception as e:
                if current_rank == 0:
                    print(f"Warning: Error during engine shutdown: {e}")
        
        # Check if log file has content (only on logging rank)
        if logger and logger.should_log:
            import os
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size > 0:
                    print(f"\n✓ Log file created: {output_file} ({file_size:,} bytes)")
                else:
                    print(f"\n⚠ Warning: Log file is empty: {output_file}")
                    print("This may indicate that:")
                    print("  1. No tensor operations were captured")
                    print("  2. The process was killed before data could be written")
                    print("  3. The ShapeLogger was not properly activated")
            else:
                print(f"\n⚠ Warning: Log file was not created: {output_file}")
        
        if current_rank == 0:
            print("\n" + "=" * 80)
            print("Next Steps")
            print("=" * 80)
            print(f"1. View detailed analysis of GPU {only_rank} shapes:")
            print(f"   python analyze_shapes.py {output_file}")
            print(f"\n2. Filter specific operations on GPU {only_rank}:")
            print(f"   python analyze_shapes.py {output_file} --filter-op attention --show-shapes")
            print(f"\n3. Find large tensor operations on GPU {only_rank}:")
            print(f"   python analyze_shapes.py {output_file} --min-elements 1000000 --show-shapes")
            print(f"\n4. Export timeline:")
            print(f"   python analyze_shapes.py {output_file} --timeline gpu0_timeline.csv")
            print("=" * 80)


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile GPU 0 kernel shapes during model inference with Tensor Parallelism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DeepSeek V3 with TP=8, profile GPU 0
  python profile_gpu0_shapes.py \\
      --model-path deepseek-ai/DeepSeek-V3 \\
      --tp-size 8
  
  # Qwen with TP=4, profile GPU 0 with verbose output
  python profile_gpu0_shapes.py \\
      --model-path Qwen/Qwen2.5-72B-Instruct \\
      --tp-size 4 \\
      --verbose
  
  # Quick test with small model
  python profile_gpu0_shapes.py \\
      --model-path Qwen/Qwen2.5-7B-Instruct \\
      --tp-size 2 \\
      --num-prompts 2 \\
      --max-tokens 20

Key Notes:
  - Only GPU 0 (rank 0) will log shapes by default
  - Other GPUs (ranks 1-7) have zero logging overhead
  - Use --only-rank to profile a different GPU
  - TP size must match your GPU configuration
        """,
    )

    # Add ServerArgs CLI arguments
    ServerArgs.add_cli_args(parser)

    # Add shape logger specific arguments
    parser.add_argument(
        "--output-file",
        type=str,
        default="gpu0_shapes.jsonl",
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
    parser.add_argument(
        "--only-rank",
        type=int,
        default=0,
        help="Only log on this rank/GPU (default: 0 for GPU 0)",
    )

    args = parser.parse_args()

    # Extract shape logger args
    output_file = args.output_file
    verbose = args.verbose
    compact = not args.no_compact
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    analyze_after = not args.no_analyze
    only_rank = args.only_rank

    # Remove our custom args to avoid conflicts with ServerArgs
    delattr(args, "output_file")
    delattr(args, "verbose")
    delattr(args, "no_compact")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    delattr(args, "no_analyze")
    delattr(args, "only_rank")

    # Parse server args
    server_args = ServerArgs.from_cli_args(args)

    # Validate model path
    if server_args.model_path is None:
        if get_current_rank() == 0:
            print("Error: --model-path is required")
            print("\nExamples:")
            print("  --model-path deepseek-ai/DeepSeek-V3")
            print("  --model-path Qwen/Qwen2.5-14B-Instruct")
            print("  --model-path /path/to/local/model")
        sys.exit(1)

    # Set default output file based on model name and rank if not specified
    if output_file == "gpu0_shapes.jsonl":
        model_name = server_args.model_path.split('/')[-1].replace('-', '_').lower()
        output_file = f"{model_name}_tp{server_args.tp_size}_gpu{only_rank}_shapes.jsonl"

    main(
        server_args=server_args,
        output_file=output_file,
        verbose=verbose,
        compact=compact,
        num_prompts=num_prompts,
        max_tokens=max_tokens,
        analyze_after=analyze_after,
        only_rank=only_rank,
    )

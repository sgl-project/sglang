"""
Monkey-patch SGLang to enable shape logging in TP workers.

This patches ModelRunner.forward to wrap it with shape logging.

Usage:
    python profile_sglang_tp_patched.py \\
        --model-path Qwen/Qwen2.5-14B-Instruct \\
        --tp-size 8 \\
        --profile-rank 0 \\
        --output-file gpu0_shapes.jsonl
"""

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sglang as sgl
from sglang.srt.server_args import ServerArgs
from torch_shape_logger_rank import CompactRankAwareShapeLogger


# Global logger for each rank
_rank_loggers = {}


def patch_model_runner():
    """Patch ModelRunner to enable shape logging."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    
    original_forward = ModelRunner.forward
    
    def forward_with_logging(self, *args, **kwargs):
        # Get rank
        rank = self.tp_rank if hasattr(self, 'tp_rank') else 0
        
        # Check if we have a logger for this rank
        if rank in _rank_loggers:
            logger = _rank_loggers[rank]
            if not logger._shape_logger_active:
                logger.__enter__()
                logger._shape_logger_active = True
        
        # Call original forward
        return original_forward(self, *args, **kwargs)
    
    ModelRunner.forward = forward_with_logging
    print("[Patch] ModelRunner.forward patched for shape logging")


def main(server_args, profile_rank=0, output_file="gpu0_shapes.jsonl", num_prompts=3, max_tokens=100):
    """Run with patched shape logging."""
    
    print("=" * 80)
    print("SGLang TP Shape Profiler (Patched)")
    print("=" * 80)
    print(f"Model: {server_args.model_path}")
    print(f"TP Size: {server_args.tp_size}")
    print(f"Profile Rank: {profile_rank}")
    print(f"Output: {output_file}")
    print("=" * 80)
    
    # Apply patch
    patch_model_runner()
    
    # Initialize loggers for target rank
    # Note: This runs in main process, actual logging happens in workers
    _rank_loggers[profile_rank] = CompactRankAwareShapeLogger(
        output_file=output_file,
        verbose=False,
        only_rank=profile_rank,
    )
    _rank_loggers[profile_rank]._shape_logger_active = False
    
    print("\n✓ Patch applied")
    print("\nInitializing engine...")
    
    # Create engine
    llm = sgl.Engine(**dataclasses.asdict(server_args))
    print("✓ Engine initialized")
    
    # Run inference
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
    
    print(f"\nGenerating {num_prompts} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    print("✓ Generation completed")
    
    # Cleanup logger
    if profile_rank in _rank_loggers:
        logger = _rank_loggers[profile_rank]
        if logger._shape_logger_active:
            logger.__exit__(None, None, None)
            logger._shape_logger_active = False
    
    # Cleanup
    llm.shutdown()
    print("\n✓ Engine shutdown")
    
    # Check output
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"\n✓ Output file: {output_file} ({size:,} bytes)")
        
        if size == 0:
            print("\n⚠ WARNING: Output file is empty!")
            print("The patch may not have propagated to worker processes.")
            print("\nThis is because:")
            print("1. Workers are spawned as separate processes")
            print("2. Monkey patches in main process don't affect workers")
            print("3. Need to patch BEFORE workers are spawned")
    else:
        print(f"\n⚠ Output file not created: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--profile-rank", type=int, default=0)
    parser.add_argument("--output-file", type=str, default="gpu0_shapes.jsonl")
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=100)
    
    args = parser.parse_args()
    
    profile_rank = args.profile_rank
    output_file = args.output_file
    num_prompts = args.num_prompts
    max_tokens = args.max_tokens
    
    delattr(args, "profile_rank")
    delattr(args, "output_file")
    delattr(args, "num_prompts")
    delattr(args, "max_tokens")
    
    server_args = ServerArgs.from_cli_args(args)
    
    if server_args.model_path is None:
        print("Error: --model-path required")
        sys.exit(1)
    
    main(server_args, profile_rank, output_file, num_prompts, max_tokens)

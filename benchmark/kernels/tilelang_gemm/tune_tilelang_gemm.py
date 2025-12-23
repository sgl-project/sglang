#!/usr/bin/env python3
"""
TileLang GEMM Tuning Script

This script tunes TileLang FP8 blockwise GEMM kernels for optimal performance.
It supports multi-GPU parallel tuning using Ray.

Usage:
    # Tune for specific (N, K) dimensions
    python tune_tilelang_gemm.py --N 4096 --K 8192
    
    # Multi-GPU parallel tuning
    python tune_tilelang_gemm.py --N 4096 --K 8192 --num-gpus 4
    
    # Tune for a model
    python tune_tilelang_gemm.py --model deepseek-ai/DeepSeek-V3 --tp 8
    
    # Tune specific kernel types only
    python tune_tilelang_gemm.py --N 4096 --K 8192 --kernel-types base splitK
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Set, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add sglang to path
SCRIPT_DIR = Path(__file__).parent
SGLANG_ROOT = SCRIPT_DIR.parent.parent.parent

# Default configuration directory
DEFAULT_CONFIG_DIR = str(
    SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / 
    "tilelang_gemm_wrapper" / "core" / "config"
)

# Default M values (common batch sizes)
DEFAULT_M_VALUES = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 
    256, 512, 1024, 1536, 2048, 3072, 4096
]


def get_model_weight_shapes(
    model_path: str,
    tp_size: int = 1,
    trust_remote_code: bool = False,
) -> Set[Tuple[int, int]]:
    """Extract (N, K) weight shapes from a model configuration."""
    try:
        from transformers import AutoConfig
    except ImportError:
        logger.error("transformers required. Install: pip install transformers")
        sys.exit(1)
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    shapes = set()
    
    hidden_size = getattr(config, "hidden_size", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", None)
    vocab_size = getattr(config, "vocab_size", None)
    moe_intermediate_size = getattr(config, "moe_intermediate_size", intermediate_size)
    n_routed_experts = getattr(config, "n_routed_experts", None)
    
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
    
    def add_shape(n: int, k: int):
        # Round to block size (128)
        n_aligned = ((n + 127) // 128) * 128
        k_aligned = ((k + 127) // 128) * 128
        if n_aligned > 0 and k_aligned > 0:
            shapes.add((n_aligned, k_aligned))
    
    if hidden_size:
        if num_attention_heads and head_dim:
            q_size = num_attention_heads * head_dim // tp_size
            kv_size = num_key_value_heads * head_dim // tp_size
            add_shape(q_size, hidden_size)
            add_shape(kv_size, hidden_size)
            add_shape(hidden_size, num_attention_heads * head_dim // tp_size)
        
        if intermediate_size:
            mlp_size = intermediate_size // tp_size
            add_shape(mlp_size, hidden_size)
            add_shape(hidden_size, mlp_size)
        
        if n_routed_experts and moe_intermediate_size:
            moe_size = moe_intermediate_size // tp_size
            add_shape(moe_size, hidden_size)
            add_shape(hidden_size, moe_size)
        
        if vocab_size:
            add_shape(vocab_size // tp_size, hidden_size)
    
    logger.info(f"Extracted {len(shapes)} unique (N, K) shapes from {model_path}")
    return shapes


def main():
    parser = argparse.ArgumentParser(
        description="TileLang GEMM Tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Target specification (either --model or --N/--K)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--model", type=str,
        help="HuggingFace model path to extract shapes from"
    )
    target_group.add_argument(
        "--N", type=int,
        help="N dimension (weight rows)"
    )
    
    parser.add_argument(
        "--K", type=int,
        help="K dimension (required if --N is specified)"
    )
    parser.add_argument(
        "--tp", "--tensor-parallel-size", type=int, default=1, dest="tp_size",
        help="Tensor parallelism size (for --model mode)"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code from HuggingFace"
    )
    
    # Tuning parameters
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
        help=f"M values to tune (default: {DEFAULT_M_VALUES})"
    )
    parser.add_argument(
        "--kernel-types", type=str, nargs="+",
        choices=["base", "swapAB", "splitK", "splitK_swapAB"],
        help="Kernel types to tune (default: all)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--bench-rep", type=int, default=20,
        help="Benchmark repetitions (default: 20)"
    )
    
    # Output
    parser.add_argument(
        "--config-dir", type=str, default=DEFAULT_CONFIG_DIR,
        help=f"Output config directory (default: {DEFAULT_CONFIG_DIR})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.N is not None and args.K is None:
        parser.error("--K is required when --N is specified")
    
    # Import tuner from sglang (after arg parsing to fail fast on arg errors)
    try:
        from sglang.srt.layers.tilelang_gemm_wrapper.core.tuner import GEMMTuner
    except ImportError as e:
        logger.error(f"Failed to import GEMMTuner: {e}")
        logger.error("Make sure sglang is installed and tilelang is available")
        sys.exit(1)
    
    # Determine shapes to tune
    if args.model:
        shapes = get_model_weight_shapes(
            args.model,
            tp_size=args.tp_size,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        shapes = {(args.N, args.K)}
    
    # Create config directory
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Tune each shape
    logger.info(f"\n{'#'*60}")
    logger.info(f"TileLang GEMM Tuning")
    logger.info(f"{'#'*60}")
    logger.info(f"Shapes to tune: {len(shapes)}")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"{'#'*60}\n")
    
    start_time = time.time()
    results = {}
    
    tuner = None
    try:
        for i, (N, K) in enumerate(sorted(shapes)):
            logger.info(f"\n[{i+1}/{len(shapes)}] Tuning N={N}, K={K}")
            
            try:
                # Create tuner (lazy initialization)
                if tuner is None:
                    tuner = GEMMTuner(
                        config_dir=args.config_dir,
                        m_values=args.m_values,
                        num_gpus=args.num_gpus or torch.cuda.device_count(),
                        bench_rep=args.bench_rep,
                    )
                
                configs = tuner.tune_for_nk(
                    N=N,
                    K=K,
                    kernel_types=args.kernel_types,
                    verbose=True,
                )
                results[(N, K)] = len(configs)
            except Exception as e:
                logger.error(f"Failed to tune N={N}, K={K}: {e}")
                results[(N, K)] = 0
    finally:
        if tuner is not None:
            tuner.shutdown()
    
    elapsed = time.time() - start_time
    
    # Summary
    logger.info(f"\n{'#'*60}")
    logger.info(f"Tuning Complete")
    logger.info(f"{'#'*60}")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    logger.info(f"Results:")
    for (N, K), count in sorted(results.items()):
        status = "✓" if count > 0 else "✗"
        logger.info(f"  {status} N={N}, K={K}: {count} M values tuned")
    
    success = sum(1 for v in results.values() if v > 0)
    logger.info(f"\nSuccess: {success}/{len(results)} shapes")
    
    if success < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()

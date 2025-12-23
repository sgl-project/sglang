"""
Compile TileLang GEMM Kernels for a model

This script extracts model weight shapes and pre-compiles TileLang GEMM kernels.
Unlike DeepGEMM which uses JIT compilation, TileLang requires pre-tuned configurations.

Usage:
    # Basic usage
    python3 -m sglang.compile_tilelang_gemm --model deepseek-ai/DeepSeek-V3 --tp 8

    # Specify config directory
    python3 -m sglang.compile_tilelang_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 \
        --config-dir /path/to/tilelang_config

    # Only warmup specific M values
    python3 -m sglang.compile_tilelang_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 \
        --m-values 1 2 4 8 16 32 64 128

Prerequisites:
    - TileLang must be installed
    - Configuration files must exist in the config directory
    - Run the tuner first if configurations don't exist:
      python -m tilelang_gemm.tuner --N <N> --K <K>
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_weight_shapes(
    model_path: str,
    tp_size: int = 1,
    trust_remote_code: bool = False,
) -> Set[Tuple[int, int]]:
    """
    Extract (N, K) weight shapes from a model configuration.
    
    Args:
        model_path: HuggingFace model path or local path
        tp_size: Tensor parallelism size
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Set of (N, K) tuples representing weight shapes
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        logger.error("transformers is required. Install with: pip install transformers")
        sys.exit(1)
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    
    shapes = set()
    
    # Common model dimensions
    hidden_size = getattr(config, "hidden_size", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", None)
    
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads
    
    logger.info(f"Model config: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    logger.info(f"  num_attention_heads={num_attention_heads}, num_key_value_heads={num_key_value_heads}, head_dim={head_dim}")
    
    # Calculate shapes with TP
    def add_shape(n: int, k: int, name: str = ""):
        # Round to block size (128)
        n_aligned = ((n + 127) // 128) * 128
        k_aligned = ((k + 127) // 128) * 128
        if n_aligned > 0 and k_aligned > 0:
            shapes.add((n_aligned, k_aligned))
            logger.debug(f"  {name}: N={n_aligned}, K={k_aligned}")
    
    if hidden_size:
        # Attention projections (Q, K, V, O)
        if num_attention_heads and head_dim:
            # QKV projection
            q_size = num_attention_heads * head_dim // tp_size
            kv_size = num_key_value_heads * head_dim // tp_size
            
            add_shape(q_size + 2 * kv_size, hidden_size, "qkv_proj")
            
            # O projection
            add_shape(hidden_size, num_attention_heads * head_dim // tp_size, "o_proj")
        
        # MLP projections
        if intermediate_size:
            mlp_size = intermediate_size // tp_size
            # gate_proj, up_proj
            add_shape(mlp_size * 2, hidden_size, "gate_up_proj")
            # down_proj
            add_shape(hidden_size, mlp_size, "down_proj")
        
        #TODO: add MoE projections
    
    logger.info(f"Extracted {len(shapes)} unique (N, K) shapes: {shapes}")
    return shapes


def compile_tilelang_kernels(
    shapes: Set[Tuple[int, int]],
    m_values: List[int],
    config_dir: Optional[str] = None,
    num_workers: int = 16,
) -> Dict[Tuple[int, int], bool]:
    """
    Pre-compile TileLang kernels for given shapes.
    
    Args:
        shapes: Set of (N, K) weight shapes
        m_values: List of M values to compile for
        config_dir: Configuration directory
        num_workers: Number of parallel compilation workers
    
    Returns:
        Dictionary mapping (N, K) to success status
    """
    try:
        from sglang.srt.layers.tilelang_gemm_wrapper.core import TileLangGEMMWrapper
    except ImportError as e:
        logger.error(f"Failed to import TileLangGEMMWrapper: {e}")
        return {}
    
    wrapper = TileLangGEMMWrapper(config_dir=config_dir)
    
    # Check available configurations
    available_configs = set(wrapper.list_available_configs())
    logger.info(f"Available configurations: {len(available_configs)}")
    
    results = {}
    total_kernels = 0
    compiled_kernels = 0
    
    for N, K in sorted(shapes):
        if (N, K) not in available_configs:
            logger.warning(f"No config found for N={N}, K={K}. Run tuner first.")
            results[(N, K)] = False
            continue
        
        logger.info(f"\nCompiling kernels for N={N}, K={K}...")
        
        # Generate warmup shapes
        warmup_shapes = [(M, N, K) for M in m_values]
        total_kernels += len(warmup_shapes)
        
        try:
            start_time = time.time()
            wrapper.warmup(warmup_shapes)
            elapsed = time.time() - start_time
            
            compiled_kernels += len(warmup_shapes)
            results[(N, K)] = True
            logger.info(f"  Compiled {len(warmup_shapes)} kernels in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"  Failed to compile: {e}")
            results[(N, K)] = False
    
    logger.info(f"\nCompilation summary: {compiled_kernels}/{total_kernels} kernels compiled")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compile TileLang GEMM Kernels for a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model path or local path"
    )
    parser.add_argument(
        "--tp", "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp_size",
        help="Tensor parallelism size (default: 1)"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace"
    )
    
    # TileLang arguments
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="TileLang configuration directory (default: built-in config)"
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096],
        help="M values to compile (default: common batch sizes)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel compilation workers (default: 16)"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("TileLang GEMM Kernel Compilation")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"TP size: {args.tp_size}")
    logger.info(f"M values: {args.m_values}")
    logger.info(f"Config dir: {args.config_dir or 'default'}")
    logger.info("=" * 60)
    
    # Step 1: Extract model weight shapes
    logger.info("\nStep 1: Extracting model weight shapes...")
    shapes = get_model_weight_shapes(
        args.model,
        tp_size=args.tp_size,
        trust_remote_code=args.trust_remote_code,
    )
    
    if not shapes:
        logger.error("No weight shapes extracted. Check model configuration.")
        sys.exit(1)
    
    # Step 2: Compile kernels
    logger.info("\nStep 2: Compiling TileLang kernels...")
    results = compile_tilelang_kernels(
        shapes,
        m_values=args.m_values,
        config_dir=args.config_dir,
        num_workers=args.num_workers,
    )
    
    # Summary
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("Compilation Complete")
    logger.info("=" * 60)
    logger.info(f"Success: {success_count}/{total_count} shapes")
    
    if success_count < total_count:
        logger.warning("\nSome shapes failed. Please run the tuner for missing configurations:")
        for (N, K), success in sorted(results.items()):
            if not success:
                logger.warning(f"  python -m tilelang_gemm.tuner --N {N} --K {K}")
        sys.exit(1)
    else:
        logger.info("\nAll kernels compiled successfully!")


if __name__ == "__main__":
    main()


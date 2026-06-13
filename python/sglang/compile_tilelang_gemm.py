"""Compile TileLang FP8 GEMM kernels for model weight shapes.

This utility warms TileLang's normal cache and can export the selected configs
used by SGLang for reproducible CI benchmarks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Set, Tuple

from sglang.srt.layers.tilelang_gemm_wrapper.configs import (
    AUTOTUNE_SEARCH_POLICIES,
    DEFAULT_M_VALUES,
    KERNEL_TYPES,
)
from sglang.srt.layers.tilelang_gemm_wrapper.tuning import warmup_tilelang_shapes

logger = logging.getLogger(__name__)


def _ceil_to_block(value: int, block_size: int = 128) -> int:
    return ((value + block_size - 1) // block_size) * block_size


def get_model_weight_shapes(
    model_path: str,
    tp_size: int = 1,
    trust_remote_code: bool = False,
) -> Set[Tuple[int, int]]:
    try:
        from transformers import AutoConfig
    except ImportError:
        logger.error("transformers is required to inspect model configs.")
        raise

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    shapes: Set[Tuple[int, int]] = set()

    hidden_size = getattr(config, "hidden_size", None)
    intermediate_size = getattr(config, "intermediate_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(config, "head_dim", None)

    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    def add_shape(n: int, k: int) -> None:
        if n > 0 and k > 0:
            shapes.add((_ceil_to_block(n), _ceil_to_block(k)))

    if hidden_size:
        if num_attention_heads and num_key_value_heads and head_dim:
            q_size = num_attention_heads * head_dim // tp_size
            kv_size = num_key_value_heads * head_dim // tp_size
            add_shape(q_size + 2 * kv_size, hidden_size)
            add_shape(hidden_size, num_attention_heads * head_dim // tp_size)

        if intermediate_size:
            mlp_size = intermediate_size // tp_size
            add_shape(mlp_size * 2, hidden_size)
            add_shape(hidden_size, mlp_size)

    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="HuggingFace model path or local model path")
    target.add_argument("--shape", action="append", help="Explicit shape as N,K")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=DEFAULT_M_VALUES,
        help="M values to warm up",
    )
    parser.add_argument(
        "--config-path",
        help="Optional selected-config JSON file or directory to load before warmup",
    )
    parser.add_argument(
        "--export-config-path",
        help="Optional path to export selected configs after warmup",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Tune candidate configs instead of only compiling selected configs",
    )
    parser.add_argument(
        "--autotune-backend",
        default="cudagraph",
        choices=("event", "cupti", "cudagraph"),
        help="TileLang profiler backend to use while autotuning",
    )
    parser.add_argument(
        "--autotune-policy",
        default="family_pruned",
        choices=AUTOTUNE_SEARCH_POLICIES,
        help="Candidate search policy to use while autotuning",
    )
    parser.add_argument(
        "--autotune-warmup",
        type=int,
        default=25,
        help="Warmup value passed to TileLang profiler during autotuning",
    )
    parser.add_argument(
        "--autotune-rep",
        type=int,
        default=100,
        help="Repetition count passed to TileLang profiler during autotuning",
    )
    parser.add_argument(
        "--autotune-max-configs",
        type=int,
        help="Optional cap on candidate configs per shape for smoke runs",
    )
    parser.add_argument(
        "--kernel-type",
        action="append",
        choices=KERNEL_TYPES,
        help="Restrict autotuning to one or more kernel families",
    )
    parser.add_argument(
        "--checkpoint-config-path",
        help="Incrementally write selected configs after each tuned shape",
    )
    parser.add_argument(
        "--resume-config-path",
        help="Load selected configs and skip exact shapes that are already tuned",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.model:
        nk_shapes = get_model_weight_shapes(
            args.model, tp_size=args.tp, trust_remote_code=args.trust_remote_code
        )
    else:
        nk_shapes = set()
        for item in args.shape:
            try:
                N, K = (int(part) for part in item.split(",", 1))
            except Exception:
                logger.error("--shape must use N,K format, got %s", item)
                sys.exit(2)
            nk_shapes.add((N, K))

    logger.info("TileLang FP8 GEMM N,K shapes: %s", sorted(nk_shapes))
    action = "Autotuning" if args.autotune else "Warming"
    logger.info("%s TileLang FP8 GEMM shapes for M values: %s", action, args.m_values)
    warmup_tilelang_shapes(
        nk_shapes,
        args.m_values,
        config_path=args.config_path,
        export_config_path=args.export_config_path,
        autotune=args.autotune,
        autotune_backend=args.autotune_backend,
        autotune_policy=args.autotune_policy,
        autotune_warmup=args.autotune_warmup,
        autotune_rep=args.autotune_rep,
        autotune_max_configs=args.autotune_max_configs,
        kernel_types=args.kernel_type,
        checkpoint_config_path=args.checkpoint_config_path,
        resume_config_path=args.resume_config_path,
    )


if __name__ == "__main__":
    main()

"""Compile TileLang FP8 GEMM kernels for model weight shapes.

This utility warms TileLang's normal cache and can export the selected configs
used by SGLang for reproducible CI benchmarks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, Set, Tuple

from sglang.srt.layers.tilelang_gemm_wrapper.configs import DEFAULT_M_VALUES

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


def warmup_tilelang_shapes(
    nk_shapes: Iterable[Tuple[int, int]],
    m_values: Iterable[int],
    config_path: str | None = None,
    export_config_path: str | None = None,
) -> None:
    from sglang.srt.layers import tilelang_gemm_wrapper

    if config_path:
        tilelang_gemm_wrapper.load_selected_configs(config_path)

    shapes = [(M, N, K) for N, K in sorted(nk_shapes) for M in m_values]
    if not shapes:
        raise RuntimeError("No TileLang GEMM shapes to warm up.")

    logger.info("Warming %s TileLang FP8 GEMM shapes", len(shapes))
    tilelang_gemm_wrapper.warmup_or_autotune_shapes(shapes)

    if export_config_path:
        tilelang_gemm_wrapper.export_selected_configs(export_config_path)
        logger.info("Exported selected TileLang configs to %s", export_config_path)


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
    warmup_tilelang_shapes(
        nk_shapes,
        args.m_values,
        config_path=args.config_path,
        export_config_path=args.export_config_path,
    )


if __name__ == "__main__":
    main()

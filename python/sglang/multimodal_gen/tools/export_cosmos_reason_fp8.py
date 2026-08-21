#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export W8A8 FP8 quantized Cosmos-Reason1-7B for OmniDreams text encoder.

Uses llmcompressor PTQ (SmoothQuantModifier + QuantizationModifier) to produce
a compressed-tensors format W8A8 FP8 model that transformers auto-detects via
``config.json:quantization_config``.

Reference:
    https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/scripts/quantize_fp8.py

Usage:
    python -m sglang.multimodal_gen.tools.export_cosmos_reason_fp8 \\
        --model-id nvidia/Cosmos-Reason1-7B \\
        --save-dir ./Cosmos-Reason1-7B-W8A8-FP8

Requirements:
    pip install llmcompressor>=0.6.0 vllm>=0.9.2
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export W8A8 FP8 quantized Cosmos-Reason1-7B"
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/Cosmos-Reason1-7B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        required=True,
        help="Output directory for W8A8 FP8 model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of calibration samples",
    )
    args = parser.parse_args()

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.transformers import oneshot
    except ImportError:
        raise ImportError(
            "llmcompressor not installed. Run: pip install llmcompressor>=0.6.0"
        )

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(
            targets="Linear",
            scheme="FP8",
            ignore=["lm_head"],
        ),
    ]

    oneshot(
        model=args.model_id,
        dataset="open_platypus",
        recipe=recipe,
        output_dir=str(args.save_dir),
        max_seq_length=2048,
        num_calibration_samples=args.num_samples,
    )
    print(f"W8A8 FP8 model saved to {args.save_dir}")


if __name__ == "__main__":
    main()

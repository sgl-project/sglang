#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export offline FP8-quantized DiT weights for OmniDreams.

This tool mirrors the lazy quantization that ``optimized_dit.py`` does on the
GPU inference hot path, but runs it **offline** once so every inference process
starts with pre-quantized weights on disk (skipping the 5-14s cold-start).

The export goes through the same model-load + ``post_load_weights`` chain as
the inference pipeline (padding-mask fuse 72→68, Cosmos channel-shuffle fuse),
so the quantized output is byte-identical to the lazy path.

Usage:
    python -m sglang.multimodal_gen.tools.export_omnidreams_fp8_dit_weights \\
        --checkpoint /path/to/single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt \\
        --output /path/to/single_view/omnidreams_fp8_dit.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export offline FP8-quantized OmniDreams DiT weights"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the raw BF16 DiT checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for FP8-quantized weights (.pt)",
    )
    parser.add_argument(
        "--export-tool-version",
        type=int,
        default=1,
        help="Version of the export tool (stored in meta for compatibility checks)",
    )
    return parser.parse_args()


def _load_and_fuse_dit(checkpoint_path: Path) -> Any:
    """Instantiate OmniDreamsDiT and run post_load_weights, mirroring
    ``OmniDreamsPipeline._load_flat_dit``.

    The key constraint (spec §3.1): raw checkpoint has PRE-FUSION shapes
    (x_embedder 72-in, final_layer weight in Cosmos patch-shuffle order).
    ``post_load_weights()`` fuses both, producing the same state_dict that
    the lazy quant path snapshots.
    """
    import torch

    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.loader.fsdp_load import (
        load_model_from_full_model_state_dict,
        set_default_torch_dtype,
    )
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT

    dit_config = OmniDreamsDiTConfig()
    with set_default_torch_dtype(torch.bfloat16), torch.device("meta"):
        model = OmniDreamsDiT(config=dit_config, hf_config={})

    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight_iterator = ((k, v) for k, v in weights.items())
    mapping_fn = get_param_names_mapping(model.param_names_mapping)

    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        torch.device("cpu"),
        torch.bfloat16,
        strict=True,
        param_names_mapping=mapping_fn,
    )

    model.post_load_weights()
    return model


def _get_checkpoint_fingerprint(checkpoint_path: Path) -> dict[str, Any]:
    """Lightweight fingerprint: (file_size, mtime) — avoids SHA256 on 3.9GB."""
    st = os.stat(checkpoint_path)
    return {"file_size": st.st_size, "mtime": st.st_mtime}


def main() -> None:
    args = _parse_args()
    checkpoint_path: Path = args.checkpoint
    output_path: Path = args.output
    export_tool_version: int = args.export_tool_version

    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading DiT from {checkpoint_path} ...")
    model = _load_and_fuse_dit(checkpoint_path)
    print("DiT loaded and fused (post_load_weights complete).")

    num_blocks = int(model.arch.num_blocks)  # type: ignore[union-attr]
    print(f"Arch: {num_blocks} blocks")

    print("Extracting fused state_dict (CPU) ...")
    state_dict = model.state_dict()
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}

    # Free the model before quantizing — the 3.9GB BF16 DiT and the FP8
    # temporaries would otherwise co-reside on the same CPU.
    del model, state_dict

    print("Running prepare_fp8_dit_weights (QKV fuse + per-out-channel FP8 quant) ...")
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        prepare_fp8_dit_weights,
    )

    fp8_weights = prepare_fp8_dit_weights(
        cpu_state,
        num_blocks=num_blocks,
        linear_policy="all",
    )
    del cpu_state

    fingerprint = _get_checkpoint_fingerprint(checkpoint_path)
    meta = {
        "export_tool_version": export_tool_version,
        "checkpoint_fingerprint": fingerprint,
        "num_blocks": num_blocks,
    }

    num_keys = len(fp8_weights)
    print(f"Saving {num_keys} keys to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": fp8_weights, "meta": meta}, output_path)
    print(f"Done. Export saved to {output_path}")
    print(f"  Fingerprint: size={fingerprint['file_size']}, mtime={fingerprint['mtime']}")


if __name__ == "__main__":
    main()

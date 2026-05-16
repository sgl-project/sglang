### Based on https://github.com/huggingface/diffusers/blob/main/scripts/convert_wan_to_diffusers.py

import argparse
import json
import pathlib
import shutil
from typing import Any, Dict, List

from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # For the I2V model
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # for the FLF2V model
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Add attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}

SUPPORTED_MODEL_TYPES = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B", "Wan2.2-TI2V-5B"]

# Cascade models have two transformers (high_noise + low_noise)
CASCADE_MODEL_TYPES = {"Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"}


def get_transformer_config(model_type: str) -> Dict[str, Any]:
    if model_type in SUPPORTED_MODEL_TYPES:
        return TRANSFORMER_KEYS_RENAME_DICT
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}"
        )


def get_transformer_dirs(model_type: str) -> List[str]:
    """Return the list of transformer directory names for a given model type."""
    if model_type in CASCADE_MODEL_TYPES:
        return ["transformer", "transformer_2"]
    return ["transformer"]


def get_quant_subpath(
    model_type: str, quant_path: pathlib.Path, transformer_dir: str
) -> pathlib.Path:
    """Return the quant weights subdirectory for a given transformer."""
    if model_type in CASCADE_MODEL_TYPES:
        sub = (
            "high_noise_model"
            if transformer_dir == "transformer"
            else "low_noise_model"
        )
        return quant_path / sub
    return quant_path


def update_dict_(d: Dict[str, Any], old_key: str, new_key: str) -> None:
    d[new_key] = d.pop(old_key)


def load_sharded_safetensors(directory: pathlib.Path, pattern: str) -> dict:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file matching '{pattern}' found in {directory}")
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple files matching '{pattern}' found in {directory}: {candidates}"
        )

    state_dict = {}
    state_dict.update(load_file(candidates[0]))
    return state_dict


def convert_transformer(
    model_type: str, model_dir: pathlib.Path, output_dir: pathlib.Path
) -> None:
    """Convert a single quantized transformer directory into Diffusers format."""
    model_path = pathlib.Path(model_dir)
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    RENAME_DICT = get_transformer_config(model_type)

    state_dict = load_sharded_safetensors(model_path, "quant_model_weight*.safetensors")

    json_candidates = sorted(model_path.glob("quant_model_description*.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"No quant_model_description*.json found in {model_path}"
        )
    with open(json_candidates[0]) as f:
        quant_config = json.load(f)

    for key in list(state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        if new_key != key:
            update_dict_(state_dict, key, new_key)
            # The quant JSON only covers quantized layers, not all model keys
            if key in quant_config:
                update_dict_(quant_config, key, new_key)

    save_file(state_dict, out_path / "diffusion_pytorch_model.safetensors")

    with open(out_path / "quant_model_description.json", "w") as f:
        json.dump(quant_config, f, indent=2)


def repack(
    model_type: str,
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """
    Full one-step repack workflow:
      1. Copy the original HF Diffusers model to output_path, excluding transformer dir(s).
      2. For each transformer: convert quant weights and copy config.json from original.
    """
    transformer_dirs = get_transformer_dirs(model_type)

    # Step 1: Copy original model, skipping transformer dirs (they will be replaced)
    logger.debug(f"Step 1: Copying original model to {output_path}")
    logger.debug(f"        (skipping: {transformer_dirs})")
    shutil.copytree(
        str(original_model_path),
        str(output_path),
        ignore=shutil.ignore_patterns(*transformer_dirs),
    )

    # Step 2+: Convert each transformer
    for i, tdir in enumerate(transformer_dirs):
        q_path = get_quant_subpath(model_type, quant_path, tdir)
        out_tdir = output_path / tdir
        logger.debug(
            f"\nStep {i + 2}: Converting {tdir} (quant source: {q_path.name})..."
        )
        convert_transformer(model_type, q_path, out_tdir)

        # Copy config.json from the original transformer dir
        src_config = original_model_path / tdir / "config.json"
        if src_config.is_file():
            shutil.copy2(str(src_config), str(out_tdir / "config.json"))
            logger.debug(f"  Copied config.json from original {tdir}/")

    logger.info(f"\nDone! Repacked model saved to: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Repack msmodelslim quantized Wan2.2 weights into HF Diffusers format"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help="Model type to convert",
    )
    parser.add_argument(
        "--original-model-path",
        type=str,
        required=True,
        help="Path to the original HF Diffusers model (e.g., /weights/Wan2.2-TI2V-5B-Diffusers)",
    )
    parser.add_argument(
        "--quant-path",
        type=str,
        required=True,
        help="Path to msmodelslim quantized weights directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the repacked model (e.g., /weights/Wan2.2-TI2V-5B-Diffusers-MXFP8)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    repack(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model_path),
        quant_path=pathlib.Path(args.quant_path),
        output_path=pathlib.Path(args.output_path),
    )

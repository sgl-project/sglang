### Based on https://github.com/huggingface/diffusers/blob/main/scripts/convert_wan_to_diffusers.py

import argparse
import json
import pathlib
from typing import Any, Dict, Tuple

from safetensors.torch import load_file, save_file

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


def get_transformer_config(model_type: str) -> Tuple[Dict[str, Any], ...]:
    if model_type == "Wan-T2V-14B":
        RENAME_DICT = TRANSFORMER_KEYS_RENAME_DICT
    return RENAME_DICT


def update_dict_(dict: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
    dict[new_key] = dict.pop(old_key)


def load_sharded_safetensors(path: pathlib.Path):
    file_path = path
    state_dict = {}
    state_dict.update(load_file(file_path))
    return state_dict


def convert_transformer(model_type: str, model_dir: str, output_dir: str):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    RENAME_DICT = get_transformer_config(model_type)

    original_state_dict = load_sharded_safetensors(
        pathlib.Path(model_dir, "quant_model_weight_w8a8_dynamic.safetensors")
    )
    with open(
        pathlib.Path(model_dir, "quant_model_description_w8a8_dynamic.json")
    ) as f:
        original_quant_config = json.load(f)

    for key in list(original_state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_dict_(original_state_dict, key, new_key)
        update_dict_(original_quant_config, key, new_key)

    save_file(
        original_state_dict,
        pathlib.Path(output_dir, "diffusion_pytorch_model.safetensors"),
    )

    with open(pathlib.Path(output_dir, "quant_model_description.json"), "w") as f:
        json.dump(original_quant_config, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    convert_transformer(
        "Wan-T2V-14B",
        model_dir=pathlib.Path(args.input_path, "high_noise_model"),
        output_dir=pathlib.Path(args.output_path, "transformer"),
    )
    convert_transformer(
        "Wan-T2V-14B",
        model_dir=pathlib.Path(args.input_path, "low_noise_model"),
        output_dir=pathlib.Path(args.output_path, "transformer_2"),
    )

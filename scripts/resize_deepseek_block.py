#!/bin/env python3
"""
Resize the FP8 weights quantization block size of original DeepSeek V3/R1 to a smaller one.

Usage: python3 resize_block_size.py DeepSeek-R1/

Default new block size is 64, default ouput_dir is 'DeepSeek-R1-Block64x64/'
"""
import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def resize_deepseek_block(input_dir, output_dir, block_size):
    assert (128 % block_size) == 0, f"128 must be divisible by block_size {block_size}"
    if output_dir == None:
        output_dir = input_dir.rstrip("/") + f"-Block{block_size}x{block_size}"
    os.makedirs(output_dir, exist_ok=True)
    model_index_file = os.path.join(output_dir, "model.safetensors.index.json")
    config_file = os.path.join(output_dir, "config.json")

    if not os.path.exists(model_index_file) or not os.path.exists(config_file):
        for f in os.listdir(input_dir):
            if not f.endswith(".safetensors") and os.path.isfile(
                os.path.join(input_dir, f)
            ):
                shutil.copy(os.path.join(input_dir, f), output_dir)
        print(f"Copy model index file and config files to {output_dir=}")
        # modify config.json and save it
        config = json.load(open(config_file))
        # modify block size from 128x128 to b x b
        quant_config = config["quantization_config"]
        quant_config["weight_block_size"] = [block_size, block_size]
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"config.json modified and saved to {config_file=}")

    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    shape_dict = {}
    resize_factor = 128 // block_size
    safetensor_files = list(glob(os.path.join(input_dir, "*.safetensors")))
    safetensor_files.sort()
    quant_count = 0
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            shape_dict[weight_name] = weight.shape
            if weight_name.endswith("_scale_inv"):
                orig_name = weight_name[:-10]
                scale_shape = torch.Size(
                    [
                        (dim + block_size - 1) // block_size
                        for dim in shape_dict[orig_name]
                    ]
                )
                assert weight.element_size() == 4
                assert weight.dim() == 2
                quant_count += 1
                expand_weight = torch.repeat_interleave(
                    weight, repeats=resize_factor, dim=0
                )
                expand_weight = torch.repeat_interleave(
                    expand_weight, repeats=resize_factor, dim=1
                )
                if scale_shape != expand_weight.size():
                    # print(f'{weight_name=}, {scale_shape=}, {expand_weight.shape=}, ', end='')
                    expand_weight = expand_weight[: scale_shape[0], : scale_shape[1]]
                    # print(f'after: {expand_weight.shape=}', flush=True)
                new_state_dict[weight_name] = expand_weight
            else:
                new_state_dict[weight_name] = weight
        new_safetensor_file = os.path.join(output_dir, file_name)
        save_file(new_state_dict, new_safetensor_file)
    print(f"{quant_count} weights are expanded to block {block_size}x{block_size}.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Resize the FP8 weights quantization block size of original DeepSeek V3/R1 to a smaller one."
    )
    parser.add_argument(
        "input_dir", type=str, help="Input HF model directory of DeepSeek V3/R1."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: ${input_dir}-Block${b}x${b}).",
    )
    parser.add_argument(
        "--new-block-size",
        "-b",
        type=int,
        default=64,
        help="A smaller block size, which is diversible by 128 (default: 64).",
    )
    args = parser.parse_args()
    resize_deepseek_block(args.input_dir, args.output_dir, args.new_block_size)
    print("All done.")

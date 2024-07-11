"""
Convert llava config into a format useable with SGLang

Usage: python3 scripts/convert_llava.py --model-path <path-to-model>
"""

import argparse
import json
import os

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, LlavaConfig, AutoImageProcessor

def add_image_token(model_path: str, hub_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(
        ["<image>"],
        special_tokens=True
    )

    print(tokenizer)
    # tokenizer.save_pretrained(model_path)
    tokenizer.push_to_hub(hub_path, private=True)
    return tokenizer.convert_tokens_to_ids("<image>")

def edit_model_config(model_path, image_token_index, hub_path):
    config = LlavaConfig.from_pretrained(model_path)

    # setattr(config, "architectures", ["LlavaLlamaForCausalLM"])
    setattr(config, "_name_or_path", config._name_or_path.split("/")[-1])
    setattr(config, "image_token_index", image_token_index)

    # config.save_pretrained(model_path)
    processor_config = AutoConfig.from_pretrained(config.mm_vision_tower)
    processor = AutoImageProcessor.from_pretrained(config.mm_vision_tower)
    setattr(config, "vision_config", processor_config)
    print(config)

    processor.push_to_hub(hub_path, private=True)
    config.push_to_hub(hub_path, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--hub-path", type=str)
    args = parser.parse_args()

    image_token_index = add_image_token(args.model_path, args.hub_path)
    edit_model_config(args.model_path, image_token_index, args.hub_path)
"""
Convert Yi-VL config into a format useable with SGLang

Usage: python3 scripts/convert_yi_vl.py --model-path <path-to-model>
"""

import argparse
import json
import os

from transformers import AutoConfig, AutoTokenizer

def add_image_token(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(
        ["<image_placeholder>"],
        special_tokens=True
    )

    print(tokenizer)
    tokenizer.save_pretrained(model_path)

def edit_model_config(model_path):
    config = AutoConfig.from_pretrained(model_path)

    setattr(config, "architectures", ["YiVLForCausalLM"])
    setattr(config, "image_token_index", 64002)

    print(config)
    config.save_pretrained(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    add_image_token(args.model_path)
    edit_model_config(args.model_path)
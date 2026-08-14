#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the Nemotron Duplex SGLang stage without loading the 44 GB file.

Each retained tensor is written as its own safetensors shard. This is less
compact than repacking large shards, but bounds host RAM to the largest tensor
and produces a standard Hugging Face weight index that SGLang can stream.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

PREFIXES = (
    "stt_model.llm.",
    "stt_model.lm_head.",
    "stt_model.function_head.",
    "stt_model.embed_tokens.",
)


def _model_file(path: Path) -> Path:
    return path / "model.safetensors" if path.is_dir() else path


def _resolve_base_model(config_path: Path, base_model: str | None) -> str:
    if base_model is not None:
        return base_model
    config = json.loads(config_path.read_text())
    try:
        return config["model"]["stt"]["model"]["pretrained_llm"]
    except KeyError as error:
        raise ValueError(
            "Could not find model.stt.model.pretrained_llm in --config; "
            "pass --base-model explicitly."
        ) from error


def _stt_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    try:
        return config["model"]["stt"]["model"]
    except KeyError as error:
        raise ValueError("Could not find model.stt.model in --config.") from error


def _configure_duplex_config(config, stt_config: dict):
    config.architectures = ["NemotronDuplexHForCausalLM"]
    config.predict_user_text = False
    config.use_function_head = True
    # Temporal Mamba state is sensitive to accumulated low-precision drift.
    # Keep it in fp32 even when model weights run in bf16 for online latency.
    config.mamba_ssm_dtype = "float32"
    config.duplex_text_channel_weight = float(
        stt_config.get("duplex_text_channel_weight", 1.0)
    )
    config.duplex_user_channel_weight = float(
        stt_config.get("duplex_user_channel_weight", 1.0)
    )
    config.duplex_function_channel_weight = float(
        stt_config.get("duplex_function_channel_weight", 1.0)
    )
    config.fuse_method = "add"
    return config


def convert(
    checkpoint: Path,
    output: Path,
    base_model: str | None,
    config_path: Path,
):
    source = _model_file(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    base_model = _resolve_base_model(config_path, base_model)
    stt_config = _stt_config(config_path)
    if bool(stt_config.get("predict_user_text", False)):
        raise ValueError("The converter does not support predict_user_text=True.")
    if not bool(stt_config.get("use_function_head", False)):
        raise ValueError("The published checkpoint requires use_function_head=True.")
    if (stt_config.get("fuse_method", "add") or "add") != "add":
        raise ValueError("The converter only supports fuse_method='add'.")

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=False)
    config = _configure_duplex_config(config, stt_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    pad_token = stt_config.get("pad_token", "<SPECIAL_12>")
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    if pad_token_id is None or pad_token_id < 0:
        raise ValueError(f"Could not resolve VoiceChat pad token {pad_token!r}.")
    config.pad_token_id = pad_token_id
    config.save_pretrained(output)
    tokenizer.save_pretrained(output)

    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if key.startswith(PREFIXES)]
        if not keys:
            raise ValueError(f"No Duplex tensors found in {source}")
        if "stt_model.function_head.weight" not in keys:
            raise ValueError(
                f"Checkpoint {source} is missing stt_model.function_head.weight"
            )
        weight_map, total_size = {}, 0
        total = len(keys)
        for index, key in enumerate(keys, 1):
            tensor = handle.get_tensor(key)
            filename = f"model-{index:05d}-of-{total:05d}.safetensors"
            save_file({key: tensor}, output / filename)
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
            del tensor

    with (output / "model.safetensors.index.json").open("w") as file:
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            file,
            indent=2,
        )
    with (output / "voicechat_source.json").open("w") as file:
        json.dump(
            {
                "source_checkpoint": str(checkpoint),
                "source_config": str(config_path),
                "base_model": base_model,
                "stage": "nemotron_duplex_h",
            },
            file,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-model")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.base_model, args.config)


if __name__ == "__main__":
    main()

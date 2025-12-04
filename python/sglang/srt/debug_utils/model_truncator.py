# This file also references Slime :: fp8_cast_bf16.py
import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def main(args):
    dir_input = Path(_maybe_snapshot_download(args.input))
    dir_output = Path(args.output)
    print(f"{dir_input=} {dir_output=}")

    dir_output.mkdir(parents=True, exist_ok=True)

    for pattern in ["generation_config.json", "*.py", "tokenizer*"]:
        os.system(f"cp -rf {dir_input}/{pattern} {dir_output}")

    _transform_json(
        dir_input,
        dir_output,
        "config.json",
        lambda data: _transform_config(args, data),
    )

    safetensors_index = _transform_json(
        dir_input,
        dir_output,
        "model.safetensors.index.json",
        lambda data: _transform_safetensors_index(args, data),
    )

    for path_input_safetensors in sorted(list(dir_input.glob("*.safetensors"))):
        path_output_safetensors = dir_output / path_input_safetensors.relative_to(
            dir_input
        )

        state_dict = load_file(path_input_safetensors)
        _transform_safetensors_file(
            state_dict, safetensors_index, debug_name=str(path_output_safetensors)
        )
        if len(state_dict) > 0:
            print(f"Save {len(state_dict)} tensors to {path_output_safetensors}")
            save_file(state_dict, path_output_safetensors)
        else:
            print(f"Skip saving {path_output_safetensors} since it is empty")


def _maybe_snapshot_download(path):
    if Path(path).exists():
        return path
    return snapshot_download(path)


def _transform_json(dir_input, dir_output, filename, fn):
    data = json.loads((dir_input / filename).read_text())
    fn(data)
    (dir_output / filename).write_text(json.dumps(data, indent=4))
    return data


def _transform_config(args, config_json):
    config_json["num_hidden_layers"] = args.keep_num_layers


def _transform_safetensors_index(args, safetensors_index):
    weight_map = safetensors_index["weight_map"]
    weight_map = {
        name: loc for name, loc in weight_map.items() if _filter_tensor_name(args, name)
    }
    safetensors_index["weight_map"] = weight_map


def _transform_safetensors_file(
    state_dict: Dict[str, torch.Tensor], safetensors_index, debug_name: str
):
    names_to_remove = set(state_dict) - set(safetensors_index["weight_map"])
    print(f"Remove {list(names_to_remove)} in {debug_name}")
    for name in names_to_remove:
        del state_dict[name]


def _filter_tensor_name(args, tensor_name: str):
    # We focus on DeepSeek-like names currently, but can be easily extended to more kinds of models
    m = re.match(r"^model.layers.(\d+).*", tensor_name)
    if m is None:
        return True

    layer_id = int(m.group(1))
    return layer_id < args.keep_num_layers


if __name__ == "__main__":
    """
    Example:
    python -m sglang.srt.debug_utils.model_truncator --input deepseek-ai/DeepSeek-V3-0324 --output /tmp/DeepSeek-V3-0324-5layer
    hf upload my_name/DeepSeek-V3-0324-5layer /tmp/DeepSeek-V3-0324-5layer

    Alternatively, the following may be used on-the-fly.
    But this may not be useful to test RL frameworks, and sometimes it may have issues.
        --json-model-override-args '{"num_hidden_layers": 5}'
    """
    parser = ArgumentParser(description="Create truncated model for fast debugging.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--keep-num-layers", type=int, default=5)
    main(parser.parse_args())

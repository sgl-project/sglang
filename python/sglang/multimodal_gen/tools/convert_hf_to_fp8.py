# copied and adapted from Slime
"""
Convert HuggingFace safetensors model to FP8 format for efficient inference.

Example usage:
    # convert FLUX.1-dev transformer to FP8
    python -m sglang.multimodal_gen.tools.convert_hf_to_fp8 \
        --model-dir /path/to/FLUX.1-dev/transformer \
        --save-dir /path/to/FLUX.1-dev/transformer-FP8 \
        --strategy block \
        --block-size 128 128

Options:
    --model-dir MODEL_DIR
                        path to the directory of the HF safetensors model (e.g., transformer subfolder)
    --save-dir SAVE_DIR
                        path to the directory to save the converted FP8 model
    --strategy {block,channel,tensor}
                        quantization strategy (default: block)
    --block-size [BLOCK_SIZE ...]
                        block size for block quantization, e.g., --block-size 128 128
    --max-workers MAX_WORKERS
                        number of worker threads for parallel processing (default: 1)
"""

import argparse
import gc
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm

FP8_INFO = torch.finfo(torch.float8_e4m3fn)
FP8_MAX, FP8_MIN = FP8_INFO.max, FP8_INFO.min


def ceildiv(a, b):
    return -(-a // b)


def block_fp8(weight, block_size):

    # per block quant
    block_n, block_k = block_size[0], block_size[1]

    shape_0, shape_1 = weight.shape

    n_tiles = ceildiv(shape_0, block_n)
    k_tiles = ceildiv(shape_1, block_k)

    q_weight = F.pad(
        weight,
        (0, k_tiles * block_k - shape_1, 0, n_tiles * block_n - shape_0),
        mode="constant",
        value=0.0,
    )

    qweight = q_weight.reshape(n_tiles, block_n, k_tiles, block_k)
    block_max = torch.max(torch.abs(qweight), dim=1, keepdim=True)[0]
    block_max = torch.max(block_max, dim=3, keepdim=True)[0]

    scale = block_max.to(torch.float32) / FP8_MAX
    qweight = (
        (qweight / scale)
        .clamp(min=FP8_MIN, max=FP8_MAX)
        .reshape((n_tiles * block_n, k_tiles * block_k))
        .to(torch.float8_e4m3fn)
    )
    qweight = qweight[:shape_0, :shape_1].clone().detach()
    scale = scale.squeeze()

    return qweight, scale


def channel_fp8(weight):
    channel_max = torch.max(weight.abs(), dim=-1, keepdim=True)[0]
    scale = channel_max.clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    return qweight, scale


def tensor_fp8(weight):
    scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
    qweight = (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX)
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.view(1)
    return qweight, scale


def quant_fp8(weight, strategy, block_size=None):
    if strategy == "tensor":
        return tensor_fp8(weight)
    elif strategy == "channel":
        return channel_fp8(weight)
    else:
        return block_fp8(weight, block_size)


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []

    def add_result(self, filename, q_weights, module_names):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += v.numel()
            self.modules_to_not_convert.extend(module_names)


def process_file(
    input_path, output_path, filename, strategy, block_size, result_collector
):
    if not filename.endswith(".safetensors"):
        return

    print(f"Processing {filename}, memory usage: {torch.cuda.memory_allocated()}")
    weights = {}
    q_weights = {}

    with safetensors.safe_open(
        os.path.join(input_path, filename), framework="pt", device="cuda"
    ) as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    modules_to_not_convert = []
    for key in weights.keys():
        if (
            "weight" in key
            and "layernorm" not in key
            and "embed" not in key
            and "router" not in key
            and "mlp.gate." not in key
            and "norm" not in key
            and "lm_head" not in key
            and "eh_proj" not in key
            and "net" not in key
            and "txt_mod" not in key
            and "img_mod" not in key
            and "img_in" not in key
            and "txt_in" not in key
            and "time_in" not in key
            and "vector_in" not in key
            and "adaLN_modulation" not in key
            and "all_final_layer" not in key
            and "feed_forward" not in key
            and "proj_out.weight" != key
        ):
            qw, s = quant_fp8(weights[key], strategy, block_size)
            q_weights[key] = qw
            if block_size:
                scale_name = key.replace(".weight", ".weight_scale_inv")
            else:
                scale_name = key.replace(".weight", ".weight_scale")
            q_weights[scale_name] = s
        else:
            modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = weights[key]

    safetensors.torch.save_file(
        q_weights, os.path.join(output_path, filename), metadata={"format": "pt"}
    )

    result_collector.add_result(filename, q_weights, modules_to_not_convert)


def convert_fp8(input_path, output_path, strategy, block_size=None, max_workers=4):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(
            os.path.join(input_path, filename)
        ):
            shutil.copyfile(
                os.path.join(input_path, filename), os.path.join(output_path, filename)
            )

    safetensors_files = [
        f for f in os.listdir(input_path) if f.endswith(".safetensors")
    ]

    result_collector = ConversionResult()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in safetensors_files:
            future = executor.submit(
                process_file,
                input_path,
                output_path,
                filename,
                strategy,
                block_size,
                result_collector,
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    if strategy == "block" or strategy == "tensor":
        quantization_config = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
        }
        if block_size:
            quantization_config["weight_block_size"] = block_size
        if len(result_collector.modules_to_not_convert) > 0:
            quantization_config["modules_to_not_convert"] = list(
                set(result_collector.modules_to_not_convert)
            )
    else:
        quant_group = {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "float",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": strategy,
                    "symmetric": True,
                    "type": "float",
                },
            },
        }
        quantization_config = {
            "config_groups": quant_group,
            "format": "float-quantized",
            "ignore": list(set(result_collector.modules_to_not_convert)),
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        }

    config_path = os.path.join(input_path, "config.json")
    if os.path.exists(config_path):
        cfg = json.load(open(config_path))
        cfg["quantization_config"] = quantization_config
        json.dump(cfg, open(os.path.join(output_path, "config.json"), "w"), indent=2)

    index_dict = {
        "weight_map": result_collector.weight_map,
        "metadata": {"total_size": result_collector.param_count},
    }
    json.dump(
        index_dict,
        open(os.path.join(output_path, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to the directory of the HF safetensors model.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Path to the directory to save the converted model.",
    )
    parser.add_argument(
        "--strategy", type=str, default="block", choices=["block", "channel", "tensor"]
    )
    parser.add_argument(
        "--block-size", type=int, nargs="*", default=None, help="eg. --block-size 32 32"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)
    elif not os.path.isdir(args.save_dir):
        raise ValueError("The save_dir should be a directory.")

    convert_fp8(
        args.model_dir, args.save_dir, args.strategy, args.block_size, args.max_workers
    )

#!/usr/bin/env python3
"""
Convert Z-Image-Turbo DiT weights to FP8 (block 128x128) for H20 inference.

This is a MODIFIED version of sglang.multimodal_gen.tools.convert_hf_to_fp8
with the "feed_forward" exclusion REMOVED, because:
  - FFN SwiGLU accounts for 56.96% of denoising GEMM time
  - The original exclusion was inherited from FLUX, not designed for Z-Image
  - FP8 block 128x128 with dynamic activation scaling is safe for SwiGLU

Usage:
    # Full FP8 (recommended: includes FFN)
    python convert_zimage_to_fp8.py \
        --model-dir /mnt/geminihzceph/rhyshen/models/Z-Image-Turbo/transformer \
        --save-dir /mnt/geminihzceph/rhyshen/models/Z-Image-Turbo/transformer-FP8-block128 \
        --strategy block --block-size 128 128

    # Compare: FP8 without FFN (match original script behavior)
    python convert_zimage_to_fp8.py \
        --model-dir /mnt/geminihzceph/rhyshen/models/Z-Image-Turbo/transformer \
        --save-dir /mnt/geminihzceph/rhyshen/models/Z-Image-Turbo/transformer-FP8-block128-no-ffn \
        --strategy block --block-size 128 128 --exclude-feed-forward
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


def quant_fp8(weight, strategy, block_size=None):
    if strategy == "block":
        return block_fp8(weight, block_size)
    else:
        raise ValueError(
            f"Only 'block' strategy supported for Z-Image. Got: {strategy}"
        )


# Z-Image-Turbo specific exclusion list
# These layers are either:
# 1. Normalization/embedding layers (sensitive to quantization, tiny compute)
# 2. Modulation layers (adaLN, sensitive, small)
# 3. Final output layers (sensitive)
ZIMAGE_ALWAYS_EXCLUDE = {
    "norm",  # RMSNorm, LayerNorm weights
    "embed",  # patch/position embeddings
    "modulation",  # adaLN modulation (includes adaLN_modulation)
    "all_final_layer",  # final projection layers
    "time_in",  # timestep embedding
    "proj_out.weight",  # output projection
    "layernorm",  # any layernorm
}

# FFN-specific exclusion (disabled by default for Z-Image)
FFN_EXCLUDE = {
    "feed_forward",  # SwiGLU FFN layers (w1, w2, w3)
}

# FLUX-specific exclusions (not needed for Z-Image, listed for reference)
FLUX_ONLY_EXCLUDE = {
    "router",  # MoE router (FLUX-specific)
    "mlp.gate.",  # MoE gate (FLUX-specific)
    "net",  # FLUX FFN (net.0.proj pattern)
    "lm_head",  # Language model head (not in DiT)
    "eh_proj",  # FLUX-specific
    "txt_mod",  # FLUX double-stream text modulation
    "img_mod",  # FLUX double-stream image modulation
    "txt_in",  # FLUX text input projection
    "img_in",  # FLUX image input projection
    "vector_in",  # FLUX vector input
}


def should_quantize(key: str, exclude_feed_forward: bool = False) -> bool:
    """Determine if a weight key should be FP8 quantized for Z-Image-Turbo."""
    if "weight" not in key:
        return False

    # Always exclude these
    for pattern in ZIMAGE_ALWAYS_EXCLUDE:
        if pattern in key:
            return False

    # Optionally exclude FFN (for A/B comparison)
    if exclude_feed_forward:
        for pattern in FFN_EXCLUDE:
            if pattern in key:
                return False

    return True


class ConversionResult:
    def __init__(self):
        self.lock = threading.Lock()
        self.weight_map = {}
        self.param_count = 0
        self.modules_to_not_convert = []
        self.quantized_keys = []
        self.skipped_keys = []

    def add_result(self, filename, q_weights, module_names, quantized, skipped):
        with self.lock:
            for k, v in q_weights.items():
                self.weight_map[k] = filename
                self.param_count += v.numel()
            self.modules_to_not_convert.extend(module_names)
            self.quantized_keys.extend(quantized)
            self.skipped_keys.extend(skipped)


def process_file(
    input_path,
    output_path,
    filename,
    strategy,
    block_size,
    exclude_feed_forward,
    result_collector,
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
    quantized_keys = []
    skipped_keys = []

    for key in weights.keys():
        if should_quantize(key, exclude_feed_forward):
            qw, s = quant_fp8(weights[key], strategy, block_size)
            q_weights[key] = qw
            scale_name = key.replace(".weight", ".weight_scale_inv")
            q_weights[scale_name] = s
            quantized_keys.append(key)

            # Print shape info for verification
            print(f"  [FP8] {key}: {list(weights[key].shape)} -> scale {list(s.shape)}")
        else:
            modules_to_not_convert.append(key.replace(".weight", ""))
            q_weights[key] = weights[key]
            skipped_keys.append(key)

    safetensors.torch.save_file(
        q_weights, os.path.join(output_path, filename), metadata={"format": "pt"}
    )

    result_collector.add_result(
        filename, q_weights, modules_to_not_convert, quantized_keys, skipped_keys
    )


def convert_fp8(
    input_path,
    output_path,
    strategy,
    block_size=None,
    max_workers=4,
    exclude_feed_forward=False,
):
    input_path = os.path.abspath(input_path)
    os.makedirs(output_path, exist_ok=True)

    # Copy non-safetensors files
    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(
            os.path.join(input_path, filename)
        ):
            shutil.copyfile(
                os.path.join(input_path, filename),
                os.path.join(output_path, filename),
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
                exclude_feed_forward,
                result_collector,
            )
            futures.append(future)

        for future in tqdm(futures, desc="Processing files"):
            future.result()

    # Write quantization config
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

    # Print summary
    print("\n" + "=" * 60)
    print("FP8 Conversion Summary for Z-Image-Turbo")
    print("=" * 60)
    print(f"Strategy: {strategy}, Block size: {block_size}")
    print(f"Exclude feed_forward: {exclude_feed_forward}")
    print(f"\nQuantized layers ({len(result_collector.quantized_keys)}):")
    for k in sorted(result_collector.quantized_keys):
        print(f"  [FP8] {k}")
    print(f"\nSkipped layers ({len(result_collector.skipped_keys)}):")
    for k in sorted(result_collector.skipped_keys):
        print(f"  [SKIP] {k}")
    print(f"\nTotal parameters: {result_collector.param_count:,}")
    print(f"Output: {output_path}")
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Z-Image-Turbo DiT to FP8 (with FFN quantization enabled by default)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the transformer subdirectory of Z-Image-Turbo",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Path to save the FP8 converted model",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="block",
        choices=["block"],
        help="Quantization strategy (only 'block' supported for Z-Image on H20)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Block size for block quantization (default: 128 128)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--exclude-feed-forward",
        action="store_true",
        help="Exclude feed_forward (FFN) layers from FP8 (for A/B comparison)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print(f"Creating directory {args.save_dir}")
        os.makedirs(args.save_dir)

    convert_fp8(
        args.model_dir,
        args.save_dir,
        args.strategy,
        args.block_size,
        args.max_workers,
        args.exclude_feed_forward,
    )

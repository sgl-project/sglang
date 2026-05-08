#!/usr/bin/env python3
"""Generate golden outputs for vision processor testing.

This script generates reference outputs from HuggingFace transformers
that are used to verify the Rust image preprocessors produce identical results.

Usage:
    # Generate all golden outputs
    python scripts/generate_vision_golden.py

    # Generate for specific model
    python scripts/generate_vision_golden.py --model llava

    # Use specific image
    python scripts/generate_vision_golden.py --image tests/fixtures/images/square.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Model configurations
MODELS = {
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "processor_class": "CLIPImageProcessor",
        "description": "Standard CLIP processing (no expand-to-square)",
    },
    "llava_pad": {
        "model_id": "liuhaotian/llava-v1.5-7b",
        "processor_class": "CLIPImageProcessor",
        "description": "With expand-to-square (image_aspect_ratio=pad)",
    },
    "llava_next": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "processor_class": "LlavaNextImageProcessor",
        "description": "Multi-crop anyres processing",
    },
    "qwen2_vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "processor_class": "Qwen2VLImageProcessor",
        "description": "Dynamic resolution with smart resize",
    },
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "processor_class": "Qwen2VLImageProcessorFast",
        "description": "Dynamic resolution with patch_size=16 and [0.5,0.5,0.5] normalization",
    },
    "phi3_vision": {
        "model_id": "microsoft/Phi-3-vision-128k-instruct",
        "processor_class": "Phi3VImageProcessor",
        "description": "Dynamic HD transform with 336x336 tiles",
    },
    "phi4_vision": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "processor_class": "Phi4MMImageProcessor",
        "description": "Dynamic HD transform with 448x448 tiles and SiGLIP encoder",
    },
    "llama4_vision": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "processor_class": "Llama4ImageProcessorFast",
        "description": "Tile-based processing with 336x336 tiles and global tile",
    },
    "pixtral": {
        "model_id": "mistralai/Pixtral-12B-2409",
        "processor_class": "PixtralImageProcessor",
        "description": "Dynamic resolution with CLIP normalization and bicubic resize",
    },
}

# Default test images
DEFAULT_IMAGES = [
    "tests/fixtures/images/square.jpg",
    "tests/fixtures/images/tall.jpg",
    "tests/fixtures/images/wide.jpg",
    "tests/fixtures/images/small.jpg",
    "tests/fixtures/images/tiny.jpg",
    "tests/fixtures/images/very_tall.jpg",
    "tests/fixtures/images/very_wide.jpg",
    "tests/fixtures/images/large.jpg",
    "tests/fixtures/images/odd_dims.jpg",
    "tests/fixtures/images/grayscale.jpg",
]


def expand_to_square(image: Image.Image, background_color: tuple) -> Image.Image:
    """Expand image to square by padding with background color.

    This matches the LLaVA preprocessing pipeline where images are
    first expanded to square before being processed by CLIP.
    """
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        # Pad vertically
        new_image = Image.new("RGB", (width, width), background_color)
        paste_y = (width - height) // 2
        new_image.paste(image, (0, paste_y))
        return new_image
    else:
        # Pad horizontally
        new_image = Image.new("RGB", (height, height), background_color)
        paste_x = (height - width) // 2
        new_image.paste(image, (paste_x, 0))
        return new_image


def generate_golden_llava(image_path: str, output_dir: str) -> dict:
    """Generate golden output for LLaVA 1.5 (standard CLIP processing).

    This uses standard CLIP processing WITHOUT expand-to-square.
    Matches behavior of llava-hf/* models where image_aspect_ratio is not set.

    LLaVA 1.5 preprocessing pipeline:
    1. Resize so shortest edge = 336 (preserving aspect ratio)
    2. Center crop to 336x336
    3. Normalize with CLIP mean/std
    """
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Standard CLIP processing (no expand-to-square)
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]

    # Calculate expected token count
    # LLaVA 1.5: (336 / 14)^2 = 576 tokens
    patch_size = 14
    image_size = 336
    num_tokens = (image_size // patch_size) ** 2

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "processor_config": processor.to_dict(),
    }


def generate_golden_llava_pad(image_path: str, output_dir: str) -> dict:
    """Generate golden output for LLaVA 1.5 with expand-to-square (pad mode).

    This uses expand-to-square preprocessing.
    Matches behavior of liuhaotian/llava-* models where image_aspect_ratio = "pad".

    LLaVA 1.5 pad mode preprocessing pipeline:
    1. Expand image to square by padding with mean color
    2. Resize to 336x336
    3. Normalize with CLIP mean/std
    """
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # LLaVA-specific: expand to square with mean color padding
    # CLIP mean values converted to 0-255 range
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    mean_color = tuple(int(m * 255) for m in clip_mean)
    image = expand_to_square(image, mean_color)

    # Process image with CLIP processor
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]

    # Calculate expected token count
    # LLaVA 1.5: (336 / 14)^2 = 576 tokens
    patch_size = 14
    image_size = 336
    num_tokens = (image_size // patch_size) ** 2

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "processor_config": processor.to_dict(),
    }


def generate_golden_llava_next(image_path: str, output_dir: str) -> dict:
    """Generate golden output for LLaVA-NeXT (anyres)."""
    try:
        from transformers import LlavaNextImageProcessor
    except ImportError:
        print("LlavaNextImageProcessor not available, skipping llava_next")
        return None

    processor = LlavaNextImageProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]

    # Get additional outputs if available
    image_sizes = outputs.get("image_sizes")

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.to_dict(),
    }

    if image_sizes is not None:
        result["image_sizes"] = np.array(image_sizes)

    return result


def generate_golden_qwen2_vl(image_path: str, output_dir: str) -> dict:
    """Generate golden output for Qwen2-VL.

    Qwen2-VL uses dynamic resolution with smart resize:
    1. Smart resize to fit within min/max pixel bounds
    2. Align dimensions to (patch_size * merge_size) boundary
    3. Normalize with CLIP mean/std
    4. Returns image_grid_thw for position encoding

    Default parameters:
    - patch_size: 14
    - merge_size: 2
    - min_pixels: 256 * 28 * 28 = 200,704
    - max_pixels: 1280 * 28 * 28 = 1,003,520
    - temporal_patch_size: 2
    """
    try:
        from transformers import Qwen2VLImageProcessor
    except ImportError:
        print("Qwen2VLImageProcessor not available, skipping qwen2_vl")
        return None

    processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]
    image_grid_thw = outputs.get("image_grid_thw")

    # Get config values for token calculation
    patch_size = processor.patch_size
    merge_size = processor.merge_size
    temporal_patch_size = getattr(processor, "temporal_patch_size", 2)
    min_pixels = processor.min_pixels
    max_pixels = processor.max_pixels

    # Calculate number of tokens
    # tokens = (T * H * W) / merge_size²
    if image_grid_thw is not None:
        # image_grid_thw has shape [batch, 3] with [T, H, W]
        grid_thw = image_grid_thw[0]  # First (and only) image
        num_tokens = int(np.prod(grid_thw) / (merge_size**2))
    else:
        num_tokens = None

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.to_dict(),
    }

    if image_grid_thw is not None:
        result["image_grid_thw"] = np.array(image_grid_thw)

    if num_tokens is not None:
        result["num_tokens"] = num_tokens

    # Add debug info
    result["config_info"] = {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

    return result


def save_golden(model_key: str, image_name: str, data: dict, output_dir: str):
    """Save golden output to files."""
    model_dir = Path(output_dir) / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy data
    npz_data = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    npz_data["original_size"] = np.array(data["original_size"])
    if "num_tokens" in data:
        npz_data["num_tokens"] = np.array([data["num_tokens"]])

    npz_path = model_dir / f"golden_{image_name}.npz"
    np.savez(npz_path, **npz_data)
    print(f"  Saved: {npz_path}")

    # Save processor config (only once per model)
    config_path = model_dir / "preprocessor_config.json"
    if not config_path.exists() and "processor_config" in data:
        with open(config_path, "w") as f:
            json.dump(data["processor_config"], f, indent=2)
        print(f"  Saved: {config_path}")


def generate_golden_qwen3_vl(image_path: str, output_dir: str) -> dict:
    """Generate golden output for Qwen3-VL.

    Qwen3-VL uses dynamic resolution with smart resize similar to Qwen2-VL
    but with different parameters:
    - patch_size: 16 (vs 14 in Qwen2-VL)
    - factor: 32 (vs 28 in Qwen2-VL)
    - normalization: [0.5, 0.5, 0.5] mean/std (vs CLIP values in Qwen2-VL)

    Default parameters:
    - patch_size: 16
    - merge_size: 2
    - temporal_patch_size: 2
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image using the image processor directly
    outputs = processor.image_processor(images=image, return_tensors="pt")

    # Convert to numpy for saving
    pixel_values = outputs["pixel_values"].numpy()
    image_grid_thw = outputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.numpy()

    # Get config values
    img_processor = processor.image_processor
    patch_size = getattr(img_processor, "patch_size", 16)
    merge_size = getattr(img_processor, "merge_size", 2)
    temporal_patch_size = getattr(img_processor, "temporal_patch_size", 2)

    # Calculate number of tokens
    if image_grid_thw is not None:
        grid_thw = image_grid_thw[0]
        num_tokens = int(np.prod(grid_thw) / (merge_size**2))
    else:
        num_tokens = None

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": img_processor.to_dict(),
    }

    if image_grid_thw is not None:
        result["image_grid_thw"] = image_grid_thw

    if num_tokens is not None:
        result["num_tokens"] = num_tokens

    # Add debug info
    result["config_info"] = {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
    }

    return result


def generate_golden_phi3_vision(image_path: str, output_dir: str) -> dict:
    """Generate golden output for Phi3-Vision.

    Phi3-Vision uses Dynamic HD transform:
    1. If width < height, transpose image
    2. Calculate scale: while scale * ceil(scale/ratio) <= hd_num: scale++
    3. Resize to new_w = scale * 336, new_h = new_w / ratio
    4. Pad height to multiple of 336 (centered, white padding)
    5. If transposed, transpose back
    6. Normalize with CLIP mean/std
    7. Create global image (336x336 via bicubic)
    8. Reshape into tiles [num_tiles, 3, 336, 336]
    9. Concatenate [global, tiles] and pad to [num_crops+1, 3, 336, 336]

    Default parameters:
    - num_crops: 16
    - num_img_tokens: 144 (per tile)
    - normalization: CLIP mean/std
    """
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]
    image_sizes = outputs.get("image_sizes")
    num_img_tokens = outputs.get("num_img_tokens")

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.to_dict(),
    }

    if image_sizes is not None:
        result["image_sizes"] = np.array(image_sizes)

    if num_img_tokens is not None:
        result["num_img_tokens"] = np.array(num_img_tokens)

    # Add debug info
    result["config_info"] = {
        "num_crops": processor.num_crops,
        "num_img_tokens": processor.num_img_tokens,
    }

    return result


def generate_golden_phi4_vision(image_path: str, output_dir: str) -> dict:
    """Generate golden output for Phi4-Vision (Phi-4-multimodal).

    Phi4-Vision uses Dynamic HD transform similar to Phi3 but with:
    - Base resolution: 448 (vs 336 in Phi3)
    - Normalization: [0.5, 0.5, 0.5] mean/std (vs CLIP in Phi3)
    - Default dynamic_hd: 36 (vs 16 num_crops in Phi3)
    - Uses SiGLIP vision encoder (vs CLIP in Phi3)
    - Has per-crop attention masks

    Token count formula:
    256 + 1 + mask_sum + mask_col0_sum + 16

    Note: Phi4 uses 'input_image_embeds' key instead of 'pixel_values'
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image using the image processor directly
    outputs = processor.image_processor(images=image, return_tensors="np")

    # Phi4 uses 'input_image_embeds' instead of 'pixel_values'
    pixel_values = outputs.get("input_image_embeds")
    pixel_attention_mask = outputs.get("image_attention_mask")
    image_sizes = outputs.get("image_sizes")
    num_img_tokens = outputs.get("num_img_tokens")

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.image_processor.to_dict(),
    }

    if pixel_attention_mask is not None:
        result["pixel_attention_mask"] = np.array(pixel_attention_mask)

    if image_sizes is not None:
        result["image_sizes"] = np.array(image_sizes)

    if num_img_tokens is not None:
        result["num_img_tokens"] = np.array(num_img_tokens)

    # Add debug info
    result["config_info"] = {
        "dynamic_hd": getattr(processor.image_processor, "dynamic_hd", 36),
        "base_resolution": 448,
    }

    return result


def generate_golden_llama4_vision(image_path: str, output_dir: str) -> dict:
    """Generate golden output for LLaMA 4 Vision.

    LLaMA 4 Vision uses tile-based processing:
    1. Find supported resolutions based on max_patches (default 16)
    2. Get best fit resolution for the image (minimize upscaling)
    3. Resize preserving aspect ratio
    4. Pad with black (0) to target dimensions
    5. Normalize with [0.5, 0.5, 0.5] mean/std
    6. Split into tiles of 336x336
    7. If multiple tiles, add global tile at the end

    Output:
    - pixel_values: [1, num_tiles, 3, 336, 336]
    - aspect_ratios: [1, 2] with [h_tiles, w_tiles]

    Token count: num_tiles * (336 / 14)² = num_tiles * 576
    """
    from transformers.models.llama4 import Llama4ImageProcessorFast

    processor = Llama4ImageProcessorFast()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image - Llama4 only supports PyTorch tensors
    outputs = processor(images=image, return_tensors="pt")
    # Convert to numpy (need to convert from bfloat16 to float32 first)
    pixel_values = outputs["pixel_values"].float().numpy()
    aspect_ratios = outputs.get("aspect_ratios")
    if aspect_ratios is not None:
        aspect_ratios = aspect_ratios.numpy()

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.to_dict(),
    }

    if aspect_ratios is not None:
        result["aspect_ratios"] = aspect_ratios

    # Calculate num_tokens from aspect_ratios
    if aspect_ratios is not None:
        h_tiles = int(aspect_ratios[0][0])
        w_tiles = int(aspect_ratios[0][1])
        num_tiles = h_tiles * w_tiles
        # Add 1 for global tile if num_tiles > 1
        total_tiles = num_tiles + 1 if num_tiles > 1 else num_tiles
        tokens_per_tile = (336 // 14) ** 2  # 576
        num_tokens = total_tiles * tokens_per_tile
        result["num_tokens"] = num_tokens

    # Add debug info
    result["config_info"] = {
        "tile_size": 336,
        "max_patches": processor.max_patches,
        "resize_to_max_canvas": processor.resize_to_max_canvas,
    }

    return result


def generate_golden_pixtral(image_path: str, output_dir: str) -> dict:
    """Generate golden output for Pixtral/Mistral3 Vision.

    Pixtral uses dynamic resolution processing:
    1. If image exceeds longest_edge (default 1024), scale down proportionally
    2. Resize to dimensions that are multiples of patch_size (default 16)
    3. Use bicubic interpolation for resize
    4. Normalize with CLIP mean/std

    Output:
    - pixel_values: [1, 3, H, W] where H, W are multiples of patch_size
    - image_sizes: [(H, W)]

    Token count: (H / patch_size) * (W / patch_size)
    """
    from transformers import PixtralImageProcessor

    processor = PixtralImageProcessor.from_pretrained("mistral-community/pixtral-12b")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Process image
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]
    image_sizes = outputs.get("image_sizes")

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "processor_config": processor.to_dict(),
    }

    if image_sizes is not None:
        result["image_sizes"] = np.array(image_sizes)

    # Calculate num_tokens from image_sizes
    if image_sizes is not None:
        h, w = image_sizes[0]
        patch_size = getattr(processor, "patch_size", {"height": 16, "width": 16})
        if isinstance(patch_size, dict):
            patch_h = patch_size.get("height", 16)
            patch_w = patch_size.get("width", 16)
        else:
            patch_h = patch_w = patch_size
        num_tokens = (h // patch_h) * (w // patch_w)
        result["num_tokens"] = num_tokens

    # Add debug info
    result["config_info"] = {
        "longest_edge": processor.size.get("longest_edge", 1024),
        "patch_size": processor.patch_size,
        "image_mean": processor.image_mean,
        "image_std": processor.image_std,
    }

    return result


def generate_for_model(model_key: str, image_paths: list, output_dir: str):
    """Generate golden outputs for a specific model."""
    print(f"\nGenerating golden outputs for {model_key}...")

    generator_fn = {
        "llava": generate_golden_llava,
        "llava_pad": generate_golden_llava_pad,
        "llava_next": generate_golden_llava_next,
        "qwen2_vl": generate_golden_qwen2_vl,
        "qwen3_vl": generate_golden_qwen3_vl,
        "phi3_vision": generate_golden_phi3_vision,
        "phi4_vision": generate_golden_phi4_vision,
        "llama4_vision": generate_golden_llama4_vision,
        "pixtral": generate_golden_pixtral,
    }.get(model_key)

    if generator_fn is None:
        print(f"  No generator for {model_key}, skipping")
        return

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}, skipping")
            continue

        image_name = Path(image_path).stem
        print(f"  Processing {image_name}...")

        try:
            data = generator_fn(image_path, output_dir)
            if data is not None:
                save_golden(model_key, image_name, data, output_dir)
                print(f"    pixel_values shape: {data['pixel_values'].shape}")
                print(
                    f"    pixel_values range: [{data['pixel_values'].min():.4f}, {data['pixel_values'].max():.4f}]"
                )
        except Exception as e:
            print(f"    Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden outputs for vision processor testing"
    )
    parser.add_argument(
        "--model", "-m", help="Specific model to generate (default: all)"
    )
    parser.add_argument("--image", "-i", action="append", help="Specific image path(s)")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="tests/fixtures/golden",
        help="Output directory for golden files",
    )
    args = parser.parse_args()

    # Determine which images to use
    image_paths = args.image if args.image else DEFAULT_IMAGES

    # Determine which models to generate
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_generate = [args.model]
    else:
        models_to_generate = list(MODELS.keys())

    print(f"Output directory: {args.output_dir}")
    print(f"Images: {image_paths}")
    print(f"Models: {models_to_generate}")

    # Generate golden outputs
    for model_key in models_to_generate:
        generate_for_model(model_key, image_paths, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()

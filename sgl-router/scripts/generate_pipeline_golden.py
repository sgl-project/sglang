#!/usr/bin/env python3
"""Generate golden outputs for multimodal pipeline testing using SGLang's processing logic.

This script generates reference outputs by replicating SGLang's image processing pipeline,
which is what the Rust MultiModalPipeline must match.

The key difference from generate_vision_golden.py:
- This uses SGLang's full processing logic (expand2square, smart_resize, etc.)
- Tests the complete pipeline including model-specific preprocessing

Usage:
    # Generate all pipeline golden outputs
    python scripts/generate_pipeline_golden.py

    # Generate for specific model
    python scripts/generate_pipeline_golden.py --model qwen2_vl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Add sglang to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

# Pipeline test configurations
PIPELINE_MODELS = {
    "qwen2_vl": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "description": "Qwen2-VL dynamic resolution pipeline",
    },
    "qwen3_vl": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "description": "Qwen3-VL with patch_size=16",
    },
    "llava": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "description": "LLaVA 1.5 CLIP preprocessing",
    },
    "llava_pad": {
        "model_id": "liuhaotian/llava-v1.5-7b",
        "description": "LLaVA 1.5 with expand-to-square",
    },
    "pixtral": {
        "model_id": "mistralai/Pixtral-12B-2409",
        "description": "Pixtral dynamic resolution",
    },
    "phi3_vision": {
        "model_id": "microsoft/Phi-3-vision-128k-instruct",
        "description": "Phi-3 Vision dynamic HD",
    },
    "phi4_vision": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "description": "Phi-4 Vision with SiGLIP",
    },
    "llama4_vision": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "description": "LLaMA 4 tile-based processing",
    },
}

# Test scenarios
TEST_SCENARIOS = {
    "single_square": {
        "images": ["tests/fixtures/images/square.jpg"],
        "description": "Single square image",
    },
    "single_wide": {
        "images": ["tests/fixtures/images/wide.jpg"],
        "description": "Single wide image",
    },
    "single_tall": {
        "images": ["tests/fixtures/images/tall.jpg"],
        "description": "Single tall image",
    },
}


# ============================================================================
# SGLang-style processing functions (copied/adapted from SGLang source)
# ============================================================================

def expand2square(pil_img: Image.Image, background_color: Tuple[int, int, int]) -> Image.Image:
    """Expand image to square by padding with background color.

    Matches sglang.srt.multimodal.mm_utils.expand2square
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# ============================================================================
# Model-specific golden generators
# ============================================================================

def generate_llava_golden(image_path: str, output_dir: Path, with_pad: bool = False) -> Dict:
    """Generate golden output for LLaVA.

    Uses SGLang's processing logic:
    - If with_pad=True: expand2square first, then CLIP processing
    - If with_pad=False: direct CLIP processing
    """
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    if with_pad:
        # SGLang expand2square logic
        background_color = tuple(int(x * 255) for x in processor.image_mean)
        image = expand2square(image, background_color)

    # Standard CLIP processing
    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]

    # LLaVA: 576 tokens (24x24 patches from 336x336)
    num_tokens = 576

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
    }


def generate_qwen_vl_golden(image_path: str, output_dir: Path, model_id: str) -> Dict:
    """Generate golden output for Qwen2-VL / Qwen3-VL.

    Uses HuggingFace's Qwen2VLImageProcessor which implements SGLang's smart_resize logic.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Get the image processor
    image_processor = processor.image_processor

    # Process the image - Qwen processors return pixel_values and image_grid_thw
    # Use return_tensors="pt" and convert to numpy (the fast processor requires pt)
    outputs = image_processor(images=[image], return_tensors="pt")
    pixel_values = outputs["pixel_values"].numpy()
    image_grid_thw = outputs["image_grid_thw"].numpy() if "image_grid_thw" in outputs else None

    # Calculate tokens from grid_thw
    # For Qwen2-VL: merge_size=2, so actual tokens = t * h * w / 4
    # For Qwen3-VL: spatial_merge_size=4, so actual tokens = t * h * w / 4
    # (Qwen3 merges spatial only, not temporal)
    merge_factor = 4  # merge_size^2 for both Qwen2 and Qwen3
    if image_grid_thw is not None:
        t, h, w = image_grid_thw[0]
        num_tokens = int(t * h * w) // merge_factor
    else:
        num_tokens = 0

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "image_grid_thw": image_grid_thw,
    }


def generate_pixtral_golden(image_path: str, output_dir: Path) -> Dict:
    """Generate golden output for Pixtral.

    Uses HuggingFace's PixtralImageProcessor.
    Note: Use mistral-community/pixtral-12b for the processor config.
    """
    from transformers import PixtralImageProcessor

    # mistral-community/pixtral-12b has the proper preprocessor_config.json
    processor = PixtralImageProcessor.from_pretrained("mistral-community/pixtral-12b")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    outputs = processor(images=image, return_tensors="np")
    pixel_values = outputs["pixel_values"]
    image_sizes = outputs.get("image_sizes")

    # Pixtral returns pixel_values and image_sizes
    # Calculate tokens from image_sizes
    patch_size = 16
    if image_sizes is not None and len(image_sizes) > 0:
        h, w = image_sizes[0]
        num_tokens = (h // patch_size) * (w // patch_size)
    elif isinstance(pixel_values, list) and len(pixel_values) > 0:
        # Fallback: get from pixel_values shape
        pv = pixel_values[0]
        if isinstance(pv, np.ndarray):
            # Shape: [C, H, W]
            h, w = pv.shape[1], pv.shape[2]
            num_tokens = (h // patch_size) * (w // patch_size)
        else:
            num_tokens = 0
    elif isinstance(pixel_values, np.ndarray) and len(pixel_values.shape) == 4:
        # Shape: [B, C, H, W]
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        num_tokens = (h // patch_size) * (w // patch_size)
    else:
        num_tokens = 0

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
    }
    if image_sizes is not None:
        result["image_sizes"] = np.array(image_sizes)

    return result


def generate_phi3_vision_golden(image_path: str, output_dir: Path) -> Dict:
    """Generate golden output for Phi-3 Vision.

    Uses HuggingFace's Phi3VImageProcessor.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-3-vision-128k-instruct",
        trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_processor = processor.image_processor
    outputs = image_processor(images=[image], return_tensors="np")
    pixel_values = outputs["pixel_values"]
    image_sizes = outputs.get("image_sizes")

    # Phi-3: tokens = num_tiles * (336/14)^2 = num_tiles * 576
    if len(pixel_values.shape) == 5:
        num_tiles = pixel_values.shape[1]
        num_tokens = num_tiles * 576
    else:
        num_tokens = 576

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "image_sizes": image_sizes,
    }


def generate_phi4_vision_golden(image_path: str, output_dir: Path) -> Dict:
    """Generate golden output for Phi-4 Vision.

    Uses HuggingFace's Phi4MMImageProcessor.
    Note: Phi4 uses 'input_image_embeds' key instead of 'pixel_values'.
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        trust_remote_code=True
    )
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    image_processor = processor.image_processor
    outputs = image_processor(images=image, return_tensors="np")

    # Phi4 uses 'input_image_embeds' instead of 'pixel_values'
    pixel_values = outputs.get("input_image_embeds")
    image_sizes = outputs.get("image_sizes")
    image_attention_mask = outputs.get("image_attention_mask")

    # Phi-4: tokens = num_tiles * (448/14)^2 = num_tiles * 1024
    if pixel_values is not None and len(pixel_values.shape) == 5:
        num_tiles = pixel_values.shape[1]
        num_tokens = num_tiles * 1024
    else:
        num_tokens = 1024

    result = {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "image_sizes": image_sizes,
    }
    if image_attention_mask is not None:
        result["image_attention_mask"] = image_attention_mask

    return result


def generate_llama4_vision_golden(image_path: str, output_dir: Path) -> Dict:
    """Generate golden output for LLaMA 4 Vision.

    Uses HuggingFace's Llama4ImageProcessorFast with default config (no pretrained download).
    """
    from transformers.models.llama4 import Llama4ImageProcessorFast

    # Use default config to avoid needing to access gated repo
    processor = Llama4ImageProcessorFast()
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # LLaMA 4 fast processor requires pt tensors
    outputs = processor(images=image, return_tensors="pt")
    # Convert from bfloat16 to float32 then to numpy
    pixel_values = outputs["pixel_values"].float().numpy()
    aspect_ratios = outputs.get("aspect_ratios")
    if aspect_ratios is not None and hasattr(aspect_ratios, "numpy"):
        aspect_ratios = aspect_ratios.numpy()

    # LLaMA 4: tokens based on tiles
    if aspect_ratios is not None:
        h_tiles = int(aspect_ratios[0][0])
        w_tiles = int(aspect_ratios[0][1])
        num_tiles = h_tiles * w_tiles
        # Add 1 for global tile if num_tiles > 1
        total_tiles = num_tiles + 1 if num_tiles > 1 else num_tiles
        num_tokens = total_tiles * 576  # 336/14 = 24, 24^2 = 576
    elif len(pixel_values.shape) == 5:
        num_tiles = pixel_values.shape[1]
        num_tokens = num_tiles * 576
    else:
        num_tokens = 576

    return {
        "pixel_values": pixel_values,
        "original_size": original_size,
        "num_tokens": num_tokens,
        "aspect_ratios": aspect_ratios,
    }


def generate_golden(model_key: str, scenario_key: str, output_dir: Path) -> Optional[Dict]:
    """Generate golden output for a model/scenario combination."""
    model_config = PIPELINE_MODELS[model_key]
    scenario = TEST_SCENARIOS[scenario_key]

    model_id = model_config["model_id"]
    image_paths = scenario["images"]

    # For now, we only support single image scenarios
    if len(image_paths) != 1:
        print(f"    WARNING: Multi-image scenarios not yet supported")
        return None

    image_path = image_paths[0]
    if not Path(image_path).exists():
        print(f"    WARNING: Image not found: {image_path}")
        return None

    print(f"  Processing {image_path}...")

    try:
        if model_key == "llava":
            result = generate_llava_golden(image_path, output_dir, with_pad=False)
        elif model_key == "llava_pad":
            result = generate_llava_golden(image_path, output_dir, with_pad=True)
        elif model_key in ("qwen2_vl", "qwen3_vl"):
            result = generate_qwen_vl_golden(image_path, output_dir, model_id)
        elif model_key == "pixtral":
            result = generate_pixtral_golden(image_path, output_dir)
        elif model_key == "phi3_vision":
            result = generate_phi3_vision_golden(image_path, output_dir)
        elif model_key == "phi4_vision":
            result = generate_phi4_vision_golden(image_path, output_dir)
        elif model_key == "llama4_vision":
            result = generate_llama4_vision_golden(image_path, output_dir)
        else:
            print(f"    WARNING: Unknown model: {model_key}")
            return None
    except Exception as e:
        print(f"    WARNING: Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Create output directory
    scenario_dir = output_dir / model_key / scenario_key
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Save pixel_values as npz
    npz_path = scenario_dir / "golden.npz"
    save_dict = {"pixel_values": result["pixel_values"]}

    # Add num_tokens
    if "num_tokens" in result:
        save_dict["num_tokens"] = np.array([result["num_tokens"]], dtype=np.int64)

    # Add model-specific arrays
    for key in ["image_grid_thw", "image_sizes", "image_attention_mask", "aspect_ratios"]:
        if key in result and result[key] is not None:
            save_dict[key] = np.array(result[key])

    np.savez(npz_path, **save_dict)

    # Save metadata as JSON
    pixel_values = result["pixel_values"]
    if isinstance(pixel_values, list):
        shape = [pv.shape for pv in pixel_values]
        dtype = str(pixel_values[0].dtype) if pixel_values else "unknown"
    else:
        shape = list(pixel_values.shape)
        dtype = str(pixel_values.dtype)

    metadata = {
        "model_id": model_id,
        "model_key": model_key,
        "scenario": scenario_key,
        "description": f"{model_config['description']} - {scenario['description']}",
        "image_paths": image_paths,
        "num_tokens": int(result.get("num_tokens", 0)),
        "original_size": list(result.get("original_size", (0, 0))),
        "pixel_values_shape": shape,
        "pixel_values_dtype": dtype,
    }

    meta_path = scenario_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved to {scenario_dir}")
    print(f"    pixel_values shape: {shape}")
    print(f"    num_tokens: {result.get('num_tokens', 0)}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate pipeline golden outputs")
    parser.add_argument("--model", type=str, help="Generate for specific model only")
    parser.add_argument("--scenario", type=str, help="Generate for specific scenario only")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/fixtures/golden_pipeline",
        help="Output directory for golden files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else list(PIPELINE_MODELS.keys())
    scenarios = [args.scenario] if args.scenario else list(TEST_SCENARIOS.keys())

    results = {}
    for model_key in models:
        if model_key not in PIPELINE_MODELS:
            print(f"Unknown model: {model_key}")
            continue

        print(f"\n=== {model_key} ===")
        model_results = {}

        for scenario_key in scenarios:
            if scenario_key not in TEST_SCENARIOS:
                print(f"Unknown scenario: {scenario_key}")
                continue

            print(f"\n  Scenario: {scenario_key}")
            metadata = generate_golden(model_key, scenario_key, output_dir)
            if metadata:
                model_results[scenario_key] = metadata

        if model_results:
            results[model_key] = model_results

    # Write summary
    if results:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n\nSummary written to {summary_path}")

    print("\nGolden outputs generated!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert LongCat weights to Diffusion native format.

This script performs a complete conversion from original LongCat weights
to Diffusion native implementation in a single step:

1. Converts transformer weights (with QKV/KV splitting)
2. Copies other components (VAE, text encoder, tokenizer, scheduler)
3. Converts LoRA weights (cfg_step_lora, refinement_lora)
4. Updates config files to point to native model

Usage:
    python scripts/checkpoint_conversion/longcat_to_Diffusion.py \
        --source /path/to/LongCat-Video/weights/LongCat-Video \
        --output weights/longcat-native \
        --validate
"""

import argparse
import glob
import json
import re
import shutil
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


def split_qkv(qkv_weight: torch.Tensor, qkv_bias: torch.Tensor | None = None):
    """Split fused QKV projection into separate Q, K, V."""
    dim = qkv_weight.shape[0] // 3
    q, k, v = torch.chunk(qkv_weight, 3, dim=0)

    if qkv_bias is not None:
        q_bias, k_bias, v_bias = torch.chunk(qkv_bias, 3, dim=0)
    else:
        q_bias = k_bias = v_bias = None

    return (q, k, v), (q_bias, k_bias, v_bias)


def split_kv(kv_weight: torch.Tensor, kv_bias: torch.Tensor | None = None):
    """Split fused KV projection into separate K, V."""
    dim = kv_weight.shape[0] // 2
    k, v = torch.chunk(kv_weight, 2, dim=0)

    if kv_bias is not None:
        k_bias, v_bias = torch.chunk(kv_bias, 2, dim=0)
    else:
        k_bias = v_bias = None

    return (k, v), (k_bias, v_bias)


def convert_transformer_weights(source_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert LongCat transformer weights to native SGLang Diffusion format.

    Main transformations:
    1. Split fused QKV projections (self-attention)
    2. Split fused KV projections (cross-attention)
    3. Rename parameters according to mapping
    """
    converted = OrderedDict()
    processed_keys = set()

    print("  Converting transformer weights...")

    for key, value in tqdm(source_weights.items(), desc="  Processing parameters"):
        if key in processed_keys:
            continue

        # === Embedders ===
        if key.startswith("x_embedder."):
            new_key = key.replace("x_embedder.", "patch_embed.")
            converted[new_key] = value

        elif key.startswith("t_embedder.mlp.0."):
            new_key = key.replace("t_embedder.mlp.0.", "time_embedder.linear_1.")
            converted[new_key] = value

        elif key.startswith("t_embedder.mlp.2."):
            new_key = key.replace("t_embedder.mlp.2.", "time_embedder.linear_2.")
            converted[new_key] = value

        elif key.startswith("y_embedder.y_proj.0."):
            new_key = key.replace("y_embedder.y_proj.0.", "caption_embedder.linear_1.")
            converted[new_key] = value

        elif key.startswith("y_embedder.y_proj.2."):
            new_key = key.replace("y_embedder.y_proj.2.", "caption_embedder.linear_2.")
            converted[new_key] = value

        # === Self-Attention QKV Splitting ===
        elif ".attn.qkv.weight" in key:
            block_idx = key.split(".")[1]
            qkv_weight = value
            qkv_bias_key = key.replace(".weight", ".bias")
            qkv_bias = source_weights.get(qkv_bias_key)

            (q, k, v), (q_bias, k_bias, v_bias) = split_qkv(qkv_weight, qkv_bias)

            converted[f"blocks.{block_idx}.self_attn.to_q.weight"] = q
            converted[f"blocks.{block_idx}.self_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.self_attn.to_v.weight"] = v

            if q_bias is not None:
                converted[f"blocks.{block_idx}.self_attn.to_q.bias"] = q_bias
                converted[f"blocks.{block_idx}.self_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.self_attn.to_v.bias"] = v_bias

            processed_keys.add(key)
            if qkv_bias is not None:
                processed_keys.add(qkv_bias_key)

        elif ".attn.qkv.bias" in key:
            continue

        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".self_attn.to_out.")
            converted[new_key] = value

        elif ".attn.q_norm." in key or ".attn.k_norm." in key:
            new_key = key.replace(".attn.", ".self_attn.")
            converted[new_key] = value

        # === Cross-Attention ===
        elif ".cross_attn.q_linear." in key:
            new_key = key.replace(".cross_attn.q_linear.", ".cross_attn.to_q.")
            converted[new_key] = value

        elif ".cross_attn.kv_linear.weight" in key:
            block_idx = key.split(".")[1]
            kv_weight = value
            kv_bias_key = key.replace(".weight", ".bias")
            kv_bias = source_weights.get(kv_bias_key)

            (k, v), (k_bias, v_bias) = split_kv(kv_weight, kv_bias)

            converted[f"blocks.{block_idx}.cross_attn.to_k.weight"] = k
            converted[f"blocks.{block_idx}.cross_attn.to_v.weight"] = v

            if k_bias is not None:
                converted[f"blocks.{block_idx}.cross_attn.to_k.bias"] = k_bias
                converted[f"blocks.{block_idx}.cross_attn.to_v.bias"] = v_bias

            processed_keys.add(key)
            if kv_bias is not None:
                processed_keys.add(kv_bias_key)

        elif ".cross_attn.kv_linear.bias" in key:
            continue

        elif ".cross_attn.proj." in key:
            new_key = key.replace(".cross_attn.proj.", ".cross_attn.to_out.")
            converted[new_key] = value

        elif ".cross_attn.q_norm." in key or ".cross_attn.k_norm." in key:
            converted[key] = value

        # === Final Layer (must come BEFORE general transformer block patterns) ===
        elif key.startswith("final_layer.adaLN_modulation.1."):
            new_key = key.replace("final_layer.adaLN_modulation.1.", "final_layer.adaln_linear.")
            converted[new_key] = value

        # === Transformer Block AdaLN ===
        elif ".adaLN_modulation.1." in key:
            new_key = key.replace(".adaLN_modulation.1.", ".adaln_linear_1.")
            converted[new_key] = value

        # === Transformer Block Normalization ===
        elif ".mod_norm_attn." in key or ".mod_norm_ffn." in key:
            continue

        elif ".pre_crs_attn_norm.weight" in key:
            new_key = key.replace(".pre_crs_attn_norm.", ".norm_cross.")
            converted[new_key] = value

        elif ".pre_crs_attn_norm.bias" in key:
            new_key = key.replace(".pre_crs_attn_norm.", ".norm_cross.")
            converted[new_key] = value

        # === FFN (SwiGLU) ===
        elif ".ffn.w1." in key or ".ffn.w2." in key or ".ffn.w3." in key:
            converted[key] = value

        elif key.startswith("final_layer.norm_final."):
            continue

        elif key.startswith("final_layer.linear."):
            new_key = key.replace("final_layer.linear.", "final_layer.proj.")
            converted[new_key] = value

        else:
            print(f"    ⚠️  Unknown key: {key}")
            converted[key] = value

    return converted


def validate_conversion(original: dict, converted: dict) -> bool:
    """Validate that conversion preserved all parameters correctly."""
    print("\n  Validating conversion...")

    orig_count = sum(p.numel() for p in original.values())
    conv_count = sum(p.numel() for p in converted.values())

    dropped_count = 0
    for key, value in original.items():
        if ".mod_norm_attn." in key or ".mod_norm_ffn." in key:
            dropped_count += value.numel()
        elif "final_layer.norm_final." in key:
            dropped_count += value.numel()

    expected_conv_count = orig_count - dropped_count

    print(f"    Original parameters: {orig_count:,}")
    print(f"    Converted parameters: {conv_count:,}")
    print(f"    Dropped parameters (norms without params): {dropped_count:,}")

    if conv_count != expected_conv_count:
        print(f"    ⚠️  Parameter count mismatch!")
        return False

    print(f"    ✓ Parameter count matches")

    # Verify QKV/KV splits
    print("\n  Verifying QKV/KV splits...")
    num_blocks = 48

    for i in range(num_blocks):
        orig_qkv_weight = original.get(f"blocks.{i}.attn.qkv.weight")
        if orig_qkv_weight is not None:
            conv_q = converted[f"blocks.{i}.self_attn.to_q.weight"]
            conv_k = converted[f"blocks.{i}.self_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.self_attn.to_v.weight"]
            reconstructed = torch.cat([conv_q, conv_k, conv_v], dim=0)
            if not torch.allclose(orig_qkv_weight, reconstructed):
                print(f"    ❌ QKV weight mismatch in block {i}")
                return False

        orig_kv_weight = original.get(f"blocks.{i}.cross_attn.kv_linear.weight")
        if orig_kv_weight is not None:
            conv_k = converted[f"blocks.{i}.cross_attn.to_k.weight"]
            conv_v = converted[f"blocks.{i}.cross_attn.to_v.weight"]
            reconstructed = torch.cat([conv_k, conv_v], dim=0)
            if not torch.allclose(orig_kv_weight, reconstructed):
                print(f"    ❌ KV weight mismatch in block {i}")
                return False

    print(f"    ✓ All splits verified successfully")
    return True


def copy_component(source_dir: Path, output_dir: Path, component: str, mapping: dict = None) -> bool:
    """Copy a component directory, optionally with name mapping."""
    source_name = mapping.get(component, component) if mapping else component
    source_path = source_dir / source_name

    if source_path.exists():
        output_path = output_dir / component
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_path, output_path)
        print(f"  ✓ {component} copied")
        return True
    else:
        print(f"  ⚠️  {component} not found, skipping")
        return False


def create_model_index():
    """Create model_index.json for Diffusion native model."""
    return {
        "_class_name": "LongCatPipeline",
        "_diffusers_version": "0.32.0",
        "workload_type": "video-generation",
        "tokenizer": ["transformers", "AutoTokenizer"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "LongCatTransformer3DModel"]  # Native model
    }


def update_transformer_config(transformer_dir: Path):
    """Update transformer config.json to point to native model."""
    config_path = transformer_dir / "config.json"
    if not config_path.exists():
        print("  ⚠️  Transformer config not found, skipping")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    if '_class_name' in config:
        old_class = config['_class_name']
        config['_class_name'] = 'LongCatTransformer3DModel'
        print(f"  Updated _class_name: {old_class} → LongCatTransformer3DModel")
    else:
        config['_class_name'] = 'LongCatTransformer3DModel'
        print(f"  Added _class_name: LongCatTransformer3DModel")

    # Fix num_heads -> num_attention_heads for Diffusion compatibility
    if 'num_heads' in config and 'num_attention_heads' not in config:
        config['num_attention_heads'] = config.pop('num_heads')
        print(f"  Updated num_heads → num_attention_heads")

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("  ✓ Transformer config updated")


# ============================================================================
# LoRA Conversion Functions
# ============================================================================

def parse_lora_key(key: str) -> tuple[str, str]:
    """Parse LongCat LoRA key into module path and weight type."""
    if key.startswith("lora___lorahyphen___"):
        key = key[len("lora___lorahyphen___"):]

    key = key.replace("___lorahyphen___", ".")

    if ".lora_down.weight" in key:
        return key.replace(".lora_down.weight", ""), "lora_down.weight"
    elif ".lora_up.weight" in key:
        return key.replace(".lora_up.weight", ""), "lora_up.weight"
    elif ".lora_up.blocks." in key:
        match = re.match(r"(.+)\.lora_up\.blocks\.(\d+)\.weight", key)
        if match:
            return match.group(1), f"lora_up.blocks.{match.group(2)}.weight"
    elif ".alpha_scale" in key:
        return key.replace(".alpha_scale", ""), "alpha_scale"
    elif ".lora_alpha" in key:
        return key.replace(".lora_alpha", ""), "lora_alpha"

    raise ValueError(f"Unknown LoRA key format: {key}")


def map_lora_module(module_path: str) -> list[tuple[str, str]]:
    """Map LongCat module path to Diffusion paths. Returns [(path, component)]."""
    # Self-attention QKV → Q, K, V
    match = re.match(r"blocks\.(\d+)\.attn\.qkv", module_path)
    if match:
        b = match.group(1)
        return [(f"blocks.{b}.self_attn.to_q", "q"),
                (f"blocks.{b}.self_attn.to_k", "k"),
                (f"blocks.{b}.self_attn.to_v", "v")]

    # Self-attention output
    match = re.match(r"blocks\.(\d+)\.attn\.proj", module_path)
    if match:
        return [(f"blocks.{match.group(1)}.self_attn.to_out", "single")]

    # Cross-attention Q
    match = re.match(r"blocks\.(\d+)\.cross_attn\.q_linear", module_path)
    if match:
        return [(f"blocks.{match.group(1)}.cross_attn.to_q", "single")]

    # Cross-attention KV → K, V
    match = re.match(r"blocks\.(\d+)\.cross_attn\.kv_linear", module_path)
    if match:
        b = match.group(1)
        return [(f"blocks.{b}.cross_attn.to_k", "k"),
                (f"blocks.{b}.cross_attn.to_v", "v")]

    # FFN
    match = re.match(r"blocks\.(\d+)\.ffn\.(w[123])", module_path)
    if match:
        return [(f"blocks.{match.group(1)}.ffn.{match.group(2)}", "single")]

    # AdaLN modulation
    match = re.match(r"blocks\.(\d+)\.adaLN_modulation\.1", module_path)
    if match:
        return [(f"blocks.{match.group(1)}.adaln_linear_1", "single")]

    # Final layer
    if module_path == "final_layer.adaLN_modulation.1":
        return [("final_layer.adaln_linear", "single")]
    if module_path == "final_layer.linear":
        return [("final_layer.proj", "single")]

    raise ValueError(f"Unknown LoRA module: {module_path}")


def convert_lora_weights(source_weights: dict[str, torch.Tensor], lora_name: str) -> dict[str, torch.Tensor]:
    """Convert LongCat LoRA to Diffusion format."""
    print(f"  Converting {lora_name}...")
    print(f"    Source keys: {len(source_weights)}")

    converted = OrderedDict()

    # Group by module
    modules = {}
    for key in source_weights.keys():
        try:
            module_path, weight_type = parse_lora_key(key)
            if module_path not in modules:
                modules[module_path] = {}
            modules[module_path][weight_type] = key
        except ValueError:
            continue

    # Process each module
    for module_path, weight_keys in modules.items():
        try:
            targets = map_lora_module(module_path)
        except ValueError:
            continue

        # Get alpha_scale if present (defaults to 1.0 if missing)
        alpha_scale = 1.0
        if "alpha_scale" in weight_keys:
            alpha_scale_tensor = source_weights[weight_keys["alpha_scale"]]
            alpha_scale = alpha_scale_tensor.item() if alpha_scale_tensor.numel() == 1 else float(alpha_scale_tensor.mean())

        # Handle lora_down (lora_A)
        if "lora_down.weight" in weight_keys:
            lora_down = source_weights[weight_keys["lora_down.weight"]]

            if len(targets) == 1:
                converted[f"{targets[0][0]}.lora_A"] = lora_down
                # Compute alpha from alpha_scale and rank
                rank = lora_down.shape[0]
                alpha = alpha_scale * rank
                converted[f"{targets[0][0]}.lora_alpha"] = torch.tensor(alpha, dtype=torch.float32)
            else:
                # Split for fused projections
                n = len(targets)
                rank = lora_down.shape[0] // n
                for i, (path, _) in enumerate(targets):
                    converted[f"{path}.lora_A"] = lora_down[i*rank:(i+1)*rank, :]
                    # Compute alpha from alpha_scale and rank for each split
                    alpha = alpha_scale * rank
                    converted[f"{path}.lora_alpha"] = torch.tensor(alpha, dtype=torch.float32)

        # Handle lora_up (lora_B) - may have multiple blocks
        lora_up_blocks = []
        i = 0
        while f"lora_up.blocks.{i}.weight" in weight_keys:
            lora_up_blocks.append(source_weights[weight_keys[f"lora_up.blocks.{i}.weight"]])
            i += 1

        if lora_up_blocks:
            # Multi-block LoRA: construct block-diagonal lora_B
            # This is equivalent to the multi-block computation without modifying Diffusion
            n_blocks = len(lora_up_blocks)
            out_per_block, rank_per_block = lora_up_blocks[0].shape  # e.g., [4096, 128]

            if len(targets) == 1:
                # Single layer with multi-block: create block-diagonal matrix
                total_out = out_per_block * n_blocks
                total_rank = rank_per_block * n_blocks
                lora_B_blockdiag = torch.zeros(total_out, total_rank, dtype=lora_up_blocks[0].dtype)

                for i in range(n_blocks):
                    lora_B_blockdiag[i*out_per_block:(i+1)*out_per_block,
                                     i*rank_per_block:(i+1)*rank_per_block] = lora_up_blocks[i]

                converted[f"{targets[0][0]}.lora_B"] = lora_B_blockdiag
                # Note: rank for alpha calculation should be total_rank (will be computed from lora_A.shape[0])
            else:
                # Multi-block with split targets (e.g., QKV split)
                # Each target gets one block
                for i, (path, _) in enumerate(targets):
                    if i < n_blocks:
                        converted[f"{path}.lora_B"] = lora_up_blocks[i]
        elif "lora_up.weight" in weight_keys:
            lora_up = source_weights[weight_keys["lora_up.weight"]]
            # Split if needed
            if len(targets) == 1:
                converted[f"{targets[0][0]}.lora_B"] = lora_up
            else:
                n = len(targets)
                out_dim = lora_up.shape[0] // n
                for i, (path, _) in enumerate(targets):
                    converted[f"{path}.lora_B"] = lora_up[i*out_dim:(i+1)*out_dim, :]
        else:
            continue

    print(f"    Output keys: {len(converted)} (including lora_alpha)")
    # Count how many lora_alpha values were added
    alpha_count = sum(1 for k in converted.keys() if "lora_alpha" in k)
    print(f"    Alpha values saved: {alpha_count}")
    return converted


def convert_loras(source_dir: Path, output_dir: Path) -> bool:
    """Convert all LoRA files in source directory."""
    lora_source = source_dir / "lora"
    if not lora_source.exists():
        print("  No LoRA directory found, skipping")
        return False

    lora_files = list(lora_source.glob("*.safetensors"))
    if not lora_files:
        print("  No LoRA files found, skipping")
        return False

    print(f"  Found {len(lora_files)} LoRA file(s)")

    # Map LoRA filenames to subdirectory names for Diffusion compatibility
    # Each LoRA gets its own directory under lora/
    lora_subdir_mapping = {
        "cfg_step_lora.safetensors": "distilled",
        "refinement_lora.safetensors": "refinement",
    }

    for lora_file in lora_files:
        try:
            # Determine output subdirectory - use mapping if available, otherwise generic name
            if lora_file.name in lora_subdir_mapping:
                lora_subdir_name = lora_subdir_mapping[lora_file.name]
            else:
                # For unknown LoRAs, create subdirectory based on filename
                lora_subdir_name = lora_file.stem

            lora_output = output_dir / "lora" / lora_subdir_name
            lora_output.mkdir(parents=True, exist_ok=True)

            # Load
            source_weights = load_file(str(lora_file))

            # Convert
            converted = convert_lora_weights(source_weights, lora_file.stem)

            # Save
            output_file = lora_output / lora_file.name
            save_file(converted, str(output_file))

            size_mb = output_file.stat().st_size / (1024**2)
            print(f"    ✓ {lora_file.name} → lora/{lora_subdir_name}/ ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"    ❌ Failed to convert {lora_file.name}: {e}")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LongCat weights to Diffusion native format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to original LongCat weights (LongCat-Video/weights/LongCat-Video/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for native weights",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion",
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Check source directory
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return 1

    # Check for dit/transformer directory (original uses 'dit', we output to 'transformer')
    transformer_source = source_dir / "dit"
    if not transformer_source.exists():
        print(f"❌ Error: DiT directory not found in source")
        return 1

    print("=" * 60)
    print("LongCat → Diffusion Native Conversion")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Convert transformer weights
    print("[Step 1/4] Converting transformer weights...")

    # Load source weights
    shard_files = sorted(glob.glob(str(transformer_source / "*.safetensors")))
    if not shard_files:
        print(f"❌ Error: No safetensors files found in {transformer_source}")
        return 1

    print(f"  Found {len(shard_files)} shard(s)")
    source_weights = {}
    for shard_file in shard_files:
        print(f"  Loading {Path(shard_file).name}...")
        source_weights.update(load_file(shard_file))

    print(f"  Loaded {len(source_weights)} parameters")

    # Convert
    converted_weights = convert_transformer_weights(source_weights)
    print(f"  Converted to {len(converted_weights)} parameters")

    # Validate if requested
    if args.validate:
        if not validate_conversion(source_weights, converted_weights):
            print("\n❌ Validation failed!")
            return 1
        print("\n✓ Validation passed!")

    # Save
    transformer_output = output_dir / "transformer"
    transformer_output.mkdir(parents=True, exist_ok=True)
    output_file = transformer_output / "model.safetensors"
    print(f"\n  Saving to {output_file}...")
    save_file(converted_weights, str(output_file))
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"  ✓ Saved ({size_gb:.2f} GB)")
    print()

    # Step 2: Copy other components
    print("[Step 2/5] Copying other components...")
    output_dir.mkdir(parents=True, exist_ok=True)

    components = ["vae", "text_encoder", "tokenizer", "scheduler"]
    for component in components:
        copy_component(source_dir, output_dir, component)

    print()

    # Step 3: Convert LoRA weights
    print("[Step 3/5] Converting LoRA weights...")
    convert_loras(source_dir, output_dir)
    print()

    # Step 4: Update transformer config
    print("[Step 4/5] Updating transformer config...")

    # Copy config.json from source if exists
    source_config = transformer_source / "config.json"
    output_config = transformer_output / "config.json"
    if source_config.exists():
        shutil.copy(source_config, output_config)
        print(f"  Copied config.json")

    update_transformer_config(transformer_output)
    print()

    # Step 5: Create model_index.json
    print("[Step 5/5] Creating model_index.json...")
    model_index_path = output_dir / "model_index.json"
    with open(model_index_path, 'w') as f:
        json.dump(create_model_index(), f, indent=2)
    print(f"  ✓ Created {model_index_path}")
    print()

    print("=" * 60)
    print("✓ Conversion Complete!")
    print("=" * 60)
    print(f"Native weights ready at: {output_dir}")
    print()
    print("Converted components:")
    print("  ✓ Transformer (native Diffusion implementation)")
    print("  ✓ VAE, text encoder, tokenizer, scheduler")
    if (output_dir / "lora").exists():
        lora_dirs = [d for d in (output_dir / "lora").iterdir() if d.is_dir()]
        if lora_dirs:
            print(f"  ✓ LoRA weights ({len(lora_dirs)} adapters)")
            for lora_dir in sorted(lora_dirs):
                print(f"    - lora/{lora_dir.name}/")
    print()
    print("Next steps:")
    print()
    print("  1. Test basic generation:")
    print("     from Diffusion import VideoGenerator")
    print(f"     generator = VideoGenerator.from_pretrained('{output_dir}')")
    print("     video = generator.generate_video(")
    print("         prompt='A cat playing piano',")
    print("         num_inference_steps=50")
    print("     )")
    print()
    if (output_dir / "lora" / "distilled").exists():
        print("  2. Test distilled generation (16 steps with LoRA):")
        print(f"     generator = VideoGenerator.from_pretrained('{output_dir}',")
        print(f"         lora_path='{output_dir}/lora/distilled',")
        print("         lora_nickname='distilled')")
        print("     video = generator.generate_video(")
        print("         prompt='A cat playing piano',")
        print("         num_inference_steps=16,")
        print("         guidance_scale=1.0)")
    print()
    print()

    return 0


if __name__ == "__main__":
    exit(main())

# SPDX-License-Identifier: Apache-2.0
"""Convert the requested minWM 5B ``model.pt`` into an SGLang model directory.

The converter only moves weights and donor components; runtime inference does not
import the sibling minWM checkout. Run it inside AWS, next to the 10 GB checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file

GENERATOR_KEYS = ("generator_ema", "ema_student", "generator", "model")
WRAPPER_PREFIXES = (
    "model._fsdp_wrapped_module.",
    "_fsdp_wrapped_module.",
    "module.",
    "model.",
)
DONOR_COMPONENTS = ("text_encoder", "tokenizer", "vae", "scheduler")
DEFAULT_SOURCE_URI = (
    "s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/"
    "Wan21/Action2V/dmd/wan22-5B-stage3-dmd-8-0721-6a531f0e067/"
    "global_step_003200/ema_student/model.pt"
)

TRANSFORMER_CONFIG = {
    "_class_name": "MinWMCausalTransformer3DModel",
    "_diffusers_version": "0.36.0",
    "model_type": "t2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 48,
    "out_dim": 48,
    "dim": 3072,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "in_channels": 48,
    "out_channels": 48,
    "num_heads": 24,
    "num_layers": 30,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "qk_norm": "rms_norm_across_heads",
    "cross_attn_norm": True,
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
    "local_attn_size": -1,
    "sink_size": 0,
    "rope_position_mode": "absolute",
    "rope_max_frame_gap": 1,
    "prompt_first_frame_pin_enabled": False,
    "scene_cut_rope_offset": 0,
    "scene_cut_sink_enabled": False,
    "num_frame_per_block": 4,
    "num_frame_first_block": 1,
    "num_frames_per_block": 4,
    "sliding_window_num_frames": 128,
    "action_type": "primitive_token_residual",
    "action_embed_dim": 256,
    "action_hidden_dim": 512,
    "action_kernel_size": 3,
    "action_history_frames": 4,
}

MODEL_INDEX = {
    "_class_name": "MinWMCausalDMDPipeline",
    "_diffusers_version": "0.36.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "T5TokenizerFast"],
    "transformer": ["diffusers", "MinWMCausalTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}


def _select_generator_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("MinWM checkpoint must contain a state dict")
    for key in GENERATOR_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value, key
    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return checkpoint, "root"
    raise ValueError(
        f"no generator state dict found; checked wrapper keys {GENERATOR_KEYS}"
    )


def _strip_wrapper_prefix(name: str) -> str:
    previous = None
    while name != previous:
        previous = name
        for prefix in WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    return name


def extract_generator_state_dict(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict, selected_key = _select_generator_state_dict(checkpoint)
    cleaned = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"non-tensor checkpoint entry: {name}")
        cleaned[_strip_wrapper_prefix(name)] = tensor.detach().contiguous()
    return cleaned, selected_key


def validate_generator_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    block_indices = {
        int(name.split(".")[1])
        for name in state_dict
        if name.startswith("blocks.") and name.split(".")[1].isdigit()
    }
    required_shapes = {
        "patch_embedding.weight": (3072, 48, 1, 2, 2),
        "patch_embedding.bias": (3072,),
        "action_in.encode_1.conv.weight": (512, 512, 3),
        "action_in.encode_2.conv.weight": (512, 512, 3),
        "action_in.proj.weight": (3072, 512),
        "head.head.weight": (192, 3072),
    }
    has_move_embedding = "action_in.move_embedding.weight" in state_dict
    has_look_embedding = "action_in.look_embedding.weight" in state_dict
    if has_move_embedding != has_look_embedding:
        raise ValueError("incompatible MinWM checkpoint: incomplete action embeddings")
    action_type = (
        "primitive_token_residual"
        if has_move_embedding
        else "primitive_rope_token_residual"
    )
    if has_move_embedding:
        required_shapes.update(
            {
                "action_in.move_embedding.weight": (5, 256),
                "action_in.look_embedding.weight": (5, 256),
            }
        )
    errors = []
    for name, shape in required_shapes.items():
        tensor = state_dict.get(name)
        if tensor is None:
            errors.append(f"missing {name}")
        elif tuple(tensor.shape) != shape:
            errors.append(f"{name}: expected {shape}, got {tuple(tensor.shape)}")
    if block_indices != set(range(30)):
        errors.append(f"expected transformer blocks 0..29, got {sorted(block_indices)}")
    forbidden = [
        name
        for name in state_dict
        if "prope" in name.lower() or "camera" in name.lower()
    ]
    if forbidden:
        errors.append(
            "checkpoint contains old camera/PRoPE tensors: " + ", ".join(forbidden[:8])
        )
    if errors:
        raise ValueError("incompatible MinWM checkpoint:\n- " + "\n- ".join(errors))
    return {
        "tensor_count": len(state_dict),
        "parameter_count": sum(tensor.numel() for tensor in state_dict.values()),
        "block_count": len(block_indices),
        "action_type": action_type,
    }


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor], output_dir: Path, max_shard_bytes: int
) -> dict:
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_bytes = 0
    for name in sorted(state_dict):
        tensor = state_dict[name]
        tensor_bytes = _tensor_nbytes(tensor)
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            shards.append(current)
            current = OrderedDict()
            current_bytes = 0
        current[name] = tensor
        current_bytes += tensor_bytes
    if current:
        shards.append(current)

    weight_map = {}
    total = len(shards)
    for index, shard in enumerate(shards, start=1):
        filename = f"diffusion_pytorch_model-{index:05d}-of-{total:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt"})
        for name in shard:
            weight_map[name] = filename
    index = {
        "metadata": {"total_size": sum(_tensor_nbytes(t) for t in state_dict.values())},
        "weight_map": weight_map,
    }
    with (output_dir / "diffusion_pytorch_model.safetensors.index.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"shard_count": total, **index["metadata"]}


def link_or_copy(source: Path, target: Path, *, link: bool) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing component: {target}")
    if link:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def build_transformer_config(
    *,
    local_attn_size: int,
    sink_size: int,
    sliding_window_num_frames: int,
    action_type: str = "primitive_token_residual",
    rope_position_mode: str = "absolute",
    rope_max_frame_gap: int = 1,
    prompt_first_frame_pin_enabled: bool = False,
    scene_cut_rope_offset: int = 0,
    scene_cut_sink_enabled: bool = False,
) -> dict:
    if local_attn_size != -1 and local_attn_size <= 0:
        raise ValueError("local_attn_size must be -1 or positive")
    if sink_size < 0:
        raise ValueError("sink_size must be non-negative")
    if local_attn_size != -1 and sink_size >= local_attn_size:
        raise ValueError("sink_size must be smaller than finite local_attn_size")
    if sliding_window_num_frames <= 0:
        raise ValueError("sliding_window_num_frames must be positive")
    if local_attn_size != -1 and sliding_window_num_frames != local_attn_size:
        raise ValueError(
            "bounded MinWM requires sliding_window_num_frames to equal "
            "local_attn_size"
        )
    if local_attn_size == -1 and sink_size >= sliding_window_num_frames:
        raise ValueError("sink_size must be smaller than sliding_window_num_frames")
    if action_type not in {
        "primitive_token_residual",
        "primitive_rope_token_residual",
    }:
        raise ValueError(f"unsupported MinWM action_type={action_type!r}")
    if rope_position_mode not in {"absolute", "block_relative"}:
        raise ValueError("rope_position_mode must be absolute or block_relative")
    if rope_max_frame_gap < 1:
        raise ValueError("rope_max_frame_gap must be >= 1")
    if rope_position_mode == "block_relative" and scene_cut_rope_offset:
        raise ValueError(
            "block_relative RoPE does not support nonzero scene_cut_rope_offset"
        )
    config = dict(TRANSFORMER_CONFIG)
    config.update(
        {
            "local_attn_size": local_attn_size,
            "sink_size": sink_size,
            "sliding_window_num_frames": sliding_window_num_frames,
            "action_type": action_type,
            "rope_position_mode": rope_position_mode,
            "rope_max_frame_gap": rope_max_frame_gap,
            "prompt_first_frame_pin_enabled": prompt_first_frame_pin_enabled,
            "scene_cut_rope_offset": scene_cut_rope_offset,
            "scene_cut_sink_enabled": scene_cut_sink_enabled,
        }
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minwm-checkpoint", required=True)
    parser.add_argument("--donor-diffusers-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--link-donor", action="store_true")
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--source-version-id")
    parser.add_argument("--source-etag")
    parser.add_argument("--max-shard-gib", type=float, default=4.0)
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--sink-size", type=int, default=0)
    parser.add_argument("--sliding-window-num-frames", type=int, default=128)
    parser.add_argument(
        "--action-type",
        choices=(
            "auto",
            "primitive_token_residual",
            "primitive_rope_token_residual",
        ),
        default="auto",
    )
    parser.add_argument(
        "--rope-position-mode",
        choices=("absolute", "block_relative"),
        default="absolute",
    )
    parser.add_argument("--rope-max-frame-gap", type=int, default=1)
    parser.add_argument("--prompt-first-frame-pin-enabled", action="store_true")
    parser.add_argument("--scene-cut-rope-offset", type=int, default=0)
    parser.add_argument("--scene-cut-sink-enabled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to write into non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict, selected_key = extract_generator_state_dict(args.minwm_checkpoint)
    summary = validate_generator_state_dict(state_dict)
    action_type = (
        summary["action_type"] if args.action_type == "auto" else args.action_type
    )
    if action_type != summary["action_type"]:
        raise ValueError(
            f"--action-type={action_type} does not match checkpoint tensors "
            f"({summary['action_type']})"
        )
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir()
    shard_summary = save_sharded_state_dict(
        state_dict,
        transformer_dir,
        max_shard_bytes=int(args.max_shard_gib * 1024**3),
    )
    transformer_config = build_transformer_config(
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        sliding_window_num_frames=args.sliding_window_num_frames,
        action_type=action_type,
        rope_position_mode=args.rope_position_mode,
        rope_max_frame_gap=args.rope_max_frame_gap,
        prompt_first_frame_pin_enabled=args.prompt_first_frame_pin_enabled,
        scene_cut_rope_offset=args.scene_cut_rope_offset,
        scene_cut_sink_enabled=args.scene_cut_sink_enabled,
    )
    with (transformer_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(transformer_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    donor = Path(args.donor_diffusers_dir)
    for component in DONOR_COMPONENTS:
        source = donor / component
        if not source.exists():
            raise FileNotFoundError(f"missing donor component: {source}")
        link_or_copy(source, output_dir / component, link=args.link_donor)
    with (output_dir / "model_index.json").open("w", encoding="utf-8") as handle:
        json.dump(MODEL_INDEX, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "format": "sglang-minwm-5b-v1",
        "source_checkpoint": {
            "uri": args.source_uri,
            "version_id": args.source_version_id,
            "etag": args.source_etag,
            "local_size": os.path.getsize(args.minwm_checkpoint),
            "selected_state_dict": selected_key,
        },
        "donor_diffusers_dir": str(donor.resolve()),
        "generator": summary,
        "safetensors": shard_summary,
        "native_geometry": {"width": 832, "height": 480},
        "causal_cache_defaults": {
            "local_attn_size": transformer_config["local_attn_size"],
            "sink_size": transformer_config["sink_size"],
            "sliding_window_num_frames": transformer_config[
                "sliding_window_num_frames"
            ],
            "provenance": "converter arguments, not checkpoint tensors",
        },
    }
    with (output_dir / "minwm_conversion_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        "serve with: sglang serve --model-path "
        f"{output_dir} --pipeline-class-name MinWMCausalDMDPipeline"
    )


if __name__ == "__main__":
    main()

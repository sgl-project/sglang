# SPDX-License-Identifier: Apache-2.0
"""Materialize the flat FastH3 Diffusers export into the base-H3 layout.

The transformer, text encoder, audio VAE, tokenizer and processor are linked
from the source snapshot; only the derived rope buffer and the video VAE in
its fused source form are written.
"""

import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_ROPE_FILE = "sglang_rope_inv_freq.safetensors"
_TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"

# Diffusers export -> fused source names; fused to_qkv and the SwiGLU half
# order are converted separately.
_DECODER_RENAMES = (
    ("decoder.proj_in.", "decoder.x_embedder."),
    (".attn.to_out.0.", ".attn.to_out."),
    (".ff.net.2.", ".ff.w2."),
)


def _write_rope_inv_freq(*, source_dir: str, output_dir: str) -> None:
    """Add the rope.inv_freq buffer the Diffusers export drops.

    The fp32 expression reproduces the base MiniMax-H3 tensor bit for bit.
    """
    with open(os.path.join(source_dir, "transformer", "config.json")) as f:
        rope_freq_dim = int(json.load(f)["rope_freq_dim"])
    exponents = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (
        2 * rope_freq_dim
    )
    inv_freq = 1.0 / (10000.0**exponents)
    transformer_dir = os.path.join(output_dir, "transformer")
    save_file({"rope.inv_freq": inv_freq}, os.path.join(transformer_dir, _ROPE_FILE))

    index_path = os.path.join(transformer_dir, _TRANSFORMER_INDEX)
    with open(index_path) as f:
        index = json.load(f)
    index["weight_map"]["rope.inv_freq"] = _ROPE_FILE
    if os.path.lexists(index_path):
        os.remove(index_path)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _interleave_qkv_rows(
    qkv: torch.Tensor, *, heads: int, dim_head: int
) -> torch.Tensor:
    """Stacked [q; k; v] rows -> the per-head (q, k, v) interleave of to_qkv."""
    rest_shape = qkv.shape[1:]
    return (
        qkv.reshape(3, heads, dim_head, *rest_shape)
        .transpose(0, 1)
        .reshape(3 * heads * dim_head, *rest_shape)
    )


def _rename_encoder(name: str) -> str:
    # encoder.down_blocks.N.resnets.M.* -> encoder.down.N.block.M.*
    # encoder.down_blocks.N.downsamplers.0.* -> encoder.down.N.downsample.*
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "encoder" and parts[1] == "down_blocks":
        block = parts[2]
        if parts[3] == "resnets":
            tail = parts[5:]
            if tail and tail[0] == "conv_shortcut":
                tail[0] = "nin_shortcut"
            return ".".join(["encoder", "down", block, "block", parts[4], *tail])
        if parts[3] == "downsamplers":
            return ".".join(["encoder", "down", block, "downsample", *parts[5:]])
    return name


def _convert_video_vae(*, source_dir: str, output_dir: str) -> None:
    """Re-serialize the video VAE in the fused source form the native decoder
    loads strictly; tensor values are unchanged."""
    vae_dir = os.path.join(source_dir, "vae")
    with open(os.path.join(vae_dir, "config.json")) as f:
        config = json.load(f)
    heads = int(config["decoder_num_attention_heads"])
    dim_head = int(config["decoder_attention_head_dim"])

    with open(
        os.path.join(vae_dir, "diffusion_pytorch_model.safetensors.index.json")
    ) as f:
        weight_map = json.load(f)["weight_map"]

    handles: dict[str, object] = {}

    def load(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(vae_dir, shard), framework="pt", device="cpu"
            )
        return handles[shard].get_tensor(name)

    converted: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for name in weight_map:
        tensor = load(name)
        if ".attn.to_q." in name or ".attn.to_k." in name or ".attn.to_v." in name:
            slot = name.split(".attn.to_")[1][0]
            fused_name = (
                name.replace(".attn.to_q.", ".attn.to_qkv.")
                .replace(".attn.to_k.", ".attn.to_qkv.")
                .replace(".attn.to_v.", ".attn.to_qkv.")
            )
            pending[fused_name][slot] = tensor
            if len(pending[fused_name]) == 3:
                stacked = torch.cat(
                    [pending[fused_name][s] for s in ("q", "k", "v")], dim=0
                )
                converted[fused_name] = _interleave_qkv_rows(
                    stacked, heads=heads, dim_head=dim_head
                )
                del pending[fused_name]
            continue
        if ".ff.net.0.proj." in name:
            # Diffusers SwiGLU stores [value, gate]; the source FF reads [gate, value].
            value, gate = tensor.chunk(2, dim=0)
            tensor = torch.cat((gate, value), dim=0)
            name = name.replace(".ff.net.0.proj.", ".ff.w1.")
        else:
            for old, new in _DECODER_RENAMES:
                name = name.replace(old, new)
            name = _rename_encoder(name)
        converted[name] = tensor

    if pending:
        raise ValueError(
            "Incomplete fused QKV groups in the Diffusers VAE export: "
            + ", ".join(sorted(pending))
        )
    # The export drops decoder.mask_token; the base checkpoint stores zeros
    # and inference never reads it.
    converted["decoder.mask_token"] = torch.zeros(
        (1, 1, heads * dim_head), dtype=torch.float32
    )

    target_dir = os.path.join(output_dir, "video_vae", "source")
    os.makedirs(target_dir, exist_ok=True)
    source_file = os.path.join(target_dir, "model.safetensors")
    save_file(
        {name: tensor.contiguous() for name, tensor in converted.items()},
        source_file,
    )
    # The completeness check globs weights at the component root while the H3
    # loader reads source/model.safetensors; both names share one inode.
    root_file = os.path.join(output_dir, "video_vae", "model.safetensors")
    if os.path.lexists(root_file):
        os.remove(root_file)
    os.link(source_file, root_file)


def _write_patched_component_config(
    *, source_path: str, target_path: str, class_name: str
) -> None:
    """Copy a component config under the SGLang-native ``_class_name``."""
    with open(source_path) as f:
        config = json.load(f)
    config["_class_name"] = class_name
    if os.path.lexists(target_path):
        os.remove(target_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def materialize(
    *, overlay_dir: str, source_dir: str, output_dir: str, manifest: dict
) -> None:
    _write_rope_inv_freq(source_dir=source_dir, output_dir=output_dir)
    _convert_video_vae(source_dir=source_dir, output_dir=output_dir)
    _write_patched_component_config(
        source_path=os.path.join(source_dir, "vae", "config.json"),
        target_path=os.path.join(output_dir, "video_vae", "config.json"),
        class_name="MiniMaxH3VideoVAE",
    )
    _write_patched_component_config(
        source_path=os.path.join(source_dir, "audio_vae", "config.json"),
        target_path=os.path.join(output_dir, "audio_vae", "config.json"),
        class_name="MiniMaxH3AudioVAE",
    )

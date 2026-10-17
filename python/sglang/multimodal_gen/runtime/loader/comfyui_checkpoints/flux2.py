# SPDX-License-Identifier: Apache-2.0
"""ComfyUI FLUX.2 / FLUX.2 Klein checkpoint spec.

ComfyUI stores FLUX.2 in the original BFL layout. FluxConfig's own name
mapping already covers those names (fused qkv, fused single-block linear1),
so the spec only has to read the DiT geometry from tensor shapes and fix the
one tensor whose halves are stored in the other order.
"""

import re

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    WeightIterator,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

FLUX2_PIPELINE = "Flux2ComfyUIPipeline"
FLUX2_KLEIN_PIPELINE = "Flux2KleinComfyUIPipeline"
FLUX2_KLEIN_BASE_PIPELINE = "Flux2KleinBaseComfyUIPipeline"

FLUX2_HEAD_DIM = 128
FLUX2_AXES_DIMS_ROPE = (32, 32, 32, 32)
FLUX2_ROPE_THETA = 2000
# Mistral-3 text features are 3 stacked layers of 5120; Klein stacks Qwen3's.
FLUX2_DEV_JOINT_DIM = 15360

_PREFIX = "model.diffusion_model."
_QUANTIZED_MARKERS = (".comfy_quant", ".weight_scale", ".scale_weight")


def read_safetensors_shapes(path: str) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name.removeprefix(_PREFIX): tuple(handle.get_slice(name).get_shape())
            for name in handle.keys()
        }


def infer_flux2_geometry(shapes: dict[str, tuple[int, ...]]) -> dict:
    """DiT geometry from the tensor shapes of a BFL-layout FLUX.2 file."""
    if any(name.endswith(_QUANTIZED_MARKERS) for name in shapes):
        raise ValueError(
            "Quantized FLUX.2 checkpoints (fp8 / comfy_quant) are not supported "
            "in ComfyUI integrated mode; use the BF16 file"
        )
    try:
        hidden, in_channels = shapes["img_in.weight"]
        joint_dim = shapes["txt_in.weight"][1]
        out_channels = shapes["final_layer.linear.weight"][0]
        mlp_hidden = shapes["double_blocks.0.img_mlp.0.weight"][0] // 2
        timestep_channels = shapes["time_in.in_layer.weight"][1]
        shapes["double_stream_modulation_img.lin.weight"]
    except KeyError as exc:
        raise ValueError(
            f"Not a BFL-layout FLUX.2 checkpoint: missing tensor {exc.args[0]!r}"
        ) from exc
    if hidden % FLUX2_HEAD_DIM != 0:
        raise ValueError(f"Unexpected FLUX.2 hidden size {hidden}")
    return {
        "hidden": hidden,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "joint_dim": joint_dim,
        "mlp_ratio": mlp_hidden / hidden,
        "timestep_channels": timestep_channels,
        "num_layers": _count_blocks(shapes, "double_blocks."),
        "num_single_layers": _count_blocks(shapes, "single_blocks."),
        "guidance_embeds": "guidance_in.in_layer.weight" in shapes,
    }


def _count_blocks(shapes: dict[str, tuple[int, ...]], prefix: str) -> int:
    return 1 + max(
        int(name[len(prefix) :].split(".")[0])
        for name in shapes
        if name.startswith(prefix)
    )


def flux2_pipeline_name(geometry: dict) -> str:
    if geometry["joint_dim"] == FLUX2_DEV_JOINT_DIM:
        return FLUX2_PIPELINE
    if geometry["guidance_embeds"]:
        return FLUX2_KLEIN_BASE_PIPELINE
    return FLUX2_KLEIN_PIPELINE


def _build_dit_config(server_args: ServerArgs) -> FluxConfig:
    geometry = infer_flux2_geometry(read_safetensors_shapes(server_args.model_path))
    dit_config = FluxConfig()
    dit_config.update_model_arch(
        {
            "patch_size": 1,
            "in_channels": geometry["in_channels"],
            "out_channels": geometry["out_channels"],
            "num_layers": geometry["num_layers"],
            "num_single_layers": geometry["num_single_layers"],
            "attention_head_dim": FLUX2_HEAD_DIM,
            "num_attention_heads": geometry["hidden"] // FLUX2_HEAD_DIM,
            "joint_attention_dim": geometry["joint_dim"],
            "timestep_guidance_channels": geometry["timestep_channels"],
            "mlp_ratio": geometry["mlp_ratio"],
            "axes_dims_rope": FLUX2_AXES_DIMS_ROPE,
            "rope_theta": FLUX2_ROPE_THETA,
            "eps": 1e-6,
            "guidance_embeds": geometry["guidance_embeds"],
        }
    )
    server_args.pipeline_config.dit_config = dit_config
    return dit_config


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    half = tensor.shape[0] // 2
    return torch.cat([tensor[half:], tensor[:half]], dim=0)


_FUSED_QKV = re.compile(r"^double_blocks\.(\d+)\.(img|txt)_attn\.qkv\.weight$")
_QKV_NAMES = {
    "img": ("to_q", "to_k", "to_v"),
    "txt": ("add_q_proj", "add_k_proj", "add_v_proj"),
}


def _convert_weights(weights: WeightIterator, dit_config) -> WeightIterator:
    for name, tensor in weights:
        name = name.removeprefix(_PREFIX)
        fused = _FUSED_QKV.match(name)
        if fused:
            # The unquantized model keeps q/k/v separate; BFL fuses them as [q; k; v].
            block, stream = fused.groups()
            for target, part in zip(_QKV_NAMES[stream], tensor.chunk(3, dim=0)):
                yield f"transformer_blocks.{block}.attn.{target}.weight", part
            continue
        if name == "final_layer.adaLN_modulation.1.weight":
            # BFL stores [scale, shift]; AdaLayerNormContinuous reads [shift, scale].
            tensor = _swap_halves(tensor)
        yield name, tensor


for _pipeline_name in (
    FLUX2_PIPELINE,
    FLUX2_KLEIN_PIPELINE,
    FLUX2_KLEIN_BASE_PIPELINE,
):
    register_comfyui_checkpoint(
        _pipeline_name,
        ComfyUICheckpointSpec(
            dit_cls_name="Flux2Transformer2DModel",
            build_dit_config=_build_dit_config,
            convert_weights=_convert_weights,
        ),
    )

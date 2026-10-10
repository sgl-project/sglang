# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Qwen-Image 2.1 checkpoint spec.

Comfy-Org's single-file DiTs (``qwen_image_2.1_bf16.safetensors`` and the
serialized INT8 ConvRot ``qwen_image_2.1_int8_convrot.safetensors``) keep the
native parameter names, except that ComfyUI fuses the SwiGLU input projection
row-wise as ``img_mlp.gate_up = [gate; up]``.
"""

import re
from typing import Any

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import QwenImage21DitConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    WeightIterator,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# INT8 ConvRot files carry a per-row weight_scale next to each weight.
_GATE_UP_RE = re.compile(r"^(.*\.img_mlp)\.gate_up\.(weight|weight_scale)$")


def _build_dit_config(server_args: ServerArgs) -> QwenImage21DitConfig:
    # The Comfy-Org files are the default Qwen-Image 2.1 architecture.
    return server_args.pipeline_config.dit_config


def _convert_weights(weights: WeightIterator, dit_config: Any) -> WeightIterator:
    """Split ``img_mlp.gate_up`` into ``gate_layer`` / ``proj``; drop quant markers."""
    ffn_dim = dit_config.arch_config.hidden_size * dit_config.arch_config.mlp_ratio
    for name, tensor in weights:
        if name.endswith(".comfy_quant"):
            continue  # read by _quant_markers
        match = _GATE_UP_RE.match(name)
        if match is None:
            yield name, tensor
            continue
        if tensor.shape[0] != 2 * ffn_dim:
            raise ValueError(
                f"{name} must have {2 * ffn_dim} rows, got {tuple(tensor.shape)}"
            )
        mlp, param = match.groups()
        # Exact for INT8 too: ConvRot rotates the input axis, the scale is per row.
        yield f"{mlp}.gate_layer.{param}", tensor[:ffn_dim]
        yield f"{mlp}.proj.{param}", tensor[ffn_dim:]


def _quant_markers(safetensors_list: list[str]) -> dict[str, dict[str, Any]]:
    from sglang.multimodal_gen.runtime.utils.quantization_utils import (
        inspect_comfy_quant_markers,
    )

    markers = {}
    for prefix, marker in inspect_comfy_quant_markers(safetensors_list).items():
        if prefix.endswith(".img_mlp.gate_up"):
            mlp = prefix[: -len(".gate_up")]
            markers[f"{mlp}.gate_layer"] = dict(marker)
            markers[f"{mlp}.proj"] = dict(marker)
        else:
            markers[prefix] = marker
    return markers


register_comfyui_checkpoint(
    "QwenImage21Pipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="QwenImage21Transformer2DModel",
        build_dit_config=_build_dit_config,
        convert_weights=_convert_weights,
        quant_markers=_quant_markers,
    ),
)

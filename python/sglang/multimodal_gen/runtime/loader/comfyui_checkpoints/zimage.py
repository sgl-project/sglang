# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Z-Image (lumina2) checkpoint spec."""

import re
from typing import Any

from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoint import (
    ComfyUICheckpointSpec,
    WeightIterator,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _build_dit_config(server_args: ServerArgs) -> ZImageDitConfig:
    dit_config = getattr(server_args.pipeline_config, "dit_config", None)
    if not isinstance(dit_config, ZImageDitConfig):
        dit_config = ZImageDitConfig()
        server_args.pipeline_config.dit_config = dit_config
    return dit_config


def _convert_weights(weights: WeightIterator, dit_config: Any) -> WeightIterator:
    """Split ComfyUI's merged attention qkv into to_q / to_k / to_v."""
    arch_config = dit_config.arch_config
    q_size = arch_config.dim
    k_size = (arch_config.dim // arch_config.num_attention_heads) * arch_config.n_kv_heads

    for name, tensor in weights:
        match = re.match(
            r"(layers|noise_refiner|context_refiner)\.(\d+)\.attention\.qkv\.(weight|bias)$",
            name,
        )
        if not match:
            yield name, tensor
            continue

        module_name, layer_idx, param_type = match.groups()
        prefix = f"{module_name}.{layer_idx}.attention"
        yield f"{prefix}.to_q.{param_type}", tensor[:q_size]
        yield f"{prefix}.to_k.{param_type}", tensor[q_size : q_size + k_size]
        yield f"{prefix}.to_v.{param_type}", tensor[q_size + k_size :]


_PARAM_NAMES_MAPPING = {
    r"(.*)\.attention\.k_norm\.weight$": (r"\1.attention.norm_k.weight", None, None),
    r"(.*)\.attention\.q_norm\.weight$": (r"\1.attention.norm_q.weight", None, None),
    r"(.*)\.attention\.out\.weight$": (r"\1.attention.to_out.0.weight", None, None),
    r"^final_layer\.(.*)$": (r"all_final_layer.2-1.\1", None, None),
    r"^x_embedder\.(.*)$": (r"all_x_embedder.2-1.\1", None, None),
}


register_comfyui_checkpoint(
    "ZImagePipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="ZImageTransformer2DModel",
        build_dit_config=_build_dit_config,
        param_names_mapping=_PARAM_NAMES_MAPPING,
        convert_weights=_convert_weights,
    ),
)

# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Flux checkpoint spec."""

import re
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoint import (
    ComfyUICheckpointSpec,
    WeightIterator,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _build_dit_config(server_args: ServerArgs) -> FluxConfig:
    dit_config = getattr(server_args.pipeline_config, "dit_config", None)
    if not isinstance(dit_config, FluxConfig):
        dit_config = FluxConfig()
        server_args.pipeline_config.dit_config = dit_config

    # ComfyUI Flux checkpoints always carry guidance_in weights.
    dit_config.arch_config.guidance_embeds = True
    return dit_config


def _split_sizes(dit_config: FluxConfig) -> tuple[int, int]:
    arch_config = dit_config.arch_config
    hidden_size = arch_config.num_attention_heads * arch_config.attention_head_dim
    mlp_hidden_dim = int(hidden_size * getattr(arch_config, "mlp_ratio", 4.0))
    return 3 * hidden_size, mlp_hidden_dim


def _split_qkv(
    tensor: torch.Tensor, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        tensor[:hidden_size],
        tensor[hidden_size : 2 * hidden_size],
        tensor[2 * hidden_size : 3 * hidden_size],
    )


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    """ComfyUI emits [shift, scale]; AdaLayerNormContinuous expects [scale, shift]."""
    half = tensor.shape[0] // 2
    return torch.cat([tensor[half:], tensor[:half]], dim=0)


def _convert_weights(weights: WeightIterator, dit_config: Any) -> WeightIterator:
    qkv_size, mlp_hidden_dim = _split_sizes(dit_config)
    has_guidance_embeds = dit_config.arch_config.guidance_embeds
    hidden_size = qkv_size // 3

    for name, tensor in weights:
        if not has_guidance_embeds and name.startswith("guidance_in."):
            continue

        match = re.match(
            r"double_blocks\.(\d+)\.(img_attn|txt_attn)\.qkv\.(weight|bias)$", name
        )
        if match:
            block_idx, attn_type, param_type = match.groups()
            if tensor.shape[0] < qkv_size:
                logger.warning(
                    "%s shape %s smaller than expected qkv size %d, skipping",
                    name,
                    tensor.shape,
                    qkv_size,
                )
                continue

            q, k, v = _split_qkv(tensor, hidden_size)
            prefix = f"transformer_blocks.{block_idx}.attn"
            if attn_type == "img_attn":
                yield f"{prefix}.to_q.{param_type}", q
                yield f"{prefix}.to_k.{param_type}", k
                yield f"{prefix}.to_v.{param_type}", v
            else:
                yield f"{prefix}.add_q_proj.{param_type}", q
                yield f"{prefix}.add_k_proj.{param_type}", k
                yield f"{prefix}.add_v_proj.{param_type}", v
            continue

        match = re.match(r"single_blocks\.(\d+)\.linear1\.(weight|bias)$", name)
        if match:
            block_idx, param_type = match.groups()
            expected_size = qkv_size + mlp_hidden_dim
            if tensor.shape[0] < expected_size:
                logger.warning(
                    "linear1.%s shape %s doesn't match expected size %d, skipping",
                    param_type,
                    tensor.shape,
                    expected_size,
                )
                continue

            q, k, v = _split_qkv(tensor[:qkv_size], hidden_size)
            prefix = f"single_transformer_blocks.{block_idx}"
            yield f"{prefix}.attn.to_q.{param_type}", q
            yield f"{prefix}.attn.to_k.{param_type}", k
            yield f"{prefix}.attn.to_v.{param_type}", v
            yield f"{prefix}.proj_mlp.{param_type}", tensor[qkv_size:]
            continue

        if name in (
            "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.1.bias",
        ):
            yield name, _swap_halves(tensor)
            continue

        yield name, tensor


# ComfyUI names differ from SGLang's diffusers-style names. Fused tensors that
# _convert_weights already split are emitted under their SGLang names, so only
# the untouched ones need an entry here.
_PARAM_NAMES_MAPPING = {
    r"double_blocks\.(\d+)\.img_attn\.proj\.(weight|bias)$": (
        r"transformer_blocks.\1.attn.to_out.0.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_attn\.proj\.(weight|bias)$": (
        r"transformer_blocks.\1.attn.to_add_out.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.img_attn\.norm\.query_norm\.scale$": (
        r"transformer_blocks.\1.attn.norm_q.weight",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.img_attn\.norm\.key_norm\.scale$": (
        r"transformer_blocks.\1.attn.norm_k.weight",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_attn\.norm\.query_norm\.scale$": (
        r"transformer_blocks.\1.attn.norm_added_q.weight",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_attn\.norm\.key_norm\.scale$": (
        r"transformer_blocks.\1.attn.norm_added_k.weight",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.img_mlp\.0\.(weight|bias)$": (
        r"transformer_blocks.\1.ff.net.0.proj.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.img_mlp\.2\.(weight|bias)$": (
        r"transformer_blocks.\1.ff.net.2.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_mlp\.0\.(weight|bias)$": (
        r"transformer_blocks.\1.ff_context.net.0.proj.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_mlp\.2\.(weight|bias)$": (
        r"transformer_blocks.\1.ff_context.net.2.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.img_mod\.lin\.(weight|bias)$": (
        r"transformer_blocks.\1.norm1.linear.\2",
        None,
        None,
    ),
    r"double_blocks\.(\d+)\.txt_mod\.lin\.(weight|bias)$": (
        r"transformer_blocks.\1.norm1_context.linear.\2",
        None,
        None,
    ),
    r"single_blocks\.(\d+)\.linear2\.(weight|bias)$": (
        r"single_transformer_blocks.\1.proj_out.\2",
        None,
        None,
    ),
    r"single_blocks\.(\d+)\.norm\.query_norm\.scale$": (
        r"single_transformer_blocks.\1.attn.norm_q.weight",
        None,
        None,
    ),
    r"single_blocks\.(\d+)\.norm\.key_norm\.scale$": (
        r"single_transformer_blocks.\1.attn.norm_k.weight",
        None,
        None,
    ),
    r"single_blocks\.(\d+)\.modulation\.lin\.(weight|bias)$": (
        r"single_transformer_blocks.\1.norm.linear.\2",
        None,
        None,
    ),
    r"^time_in\.in_layer\.(weight|bias)$": (
        r"time_text_embed.timestep_embedder.linear_1.\1",
        None,
        None,
    ),
    r"^time_in\.out_layer\.(weight|bias)$": (
        r"time_text_embed.timestep_embedder.linear_2.\1",
        None,
        None,
    ),
    r"^txt_in\.(weight|bias)$": (r"context_embedder.\1", None, None),
    r"^vector_in\.in_layer\.(weight|bias)$": (
        r"time_text_embed.text_embedder.linear_1.\1",
        None,
        None,
    ),
    r"^vector_in\.out_layer\.(weight|bias)$": (
        r"time_text_embed.text_embedder.linear_2.\1",
        None,
        None,
    ),
    r"^final_layer\.linear\.(weight|bias)$": (r"proj_out.\1", None, None),
    r"^final_layer\.norm_final\.(weight|bias)$": (r"norm_out.\1", None, None),
    r"^final_layer\.adaLN_modulation\.1\.(weight|bias)$": (
        r"norm_out.linear.\1",
        None,
        None,
    ),
    r"^img_in\.(weight|bias)$": (r"x_embedder.\1", None, None),
    r"^guidance_in\.in_layer\.(weight|bias)$": (
        r"time_text_embed.guidance_embedder.linear_1.\1",
        None,
        None,
    ),
    r"^guidance_in\.out_layer\.(weight|bias)$": (
        r"time_text_embed.guidance_embedder.linear_2.\1",
        None,
        None,
    ),
}


register_comfyui_checkpoint(
    "FluxPipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="FluxTransformer2DModel",
        build_dit_config=_build_dit_config,
        param_names_mapping=_PARAM_NAMES_MAPPING,
        convert_weights=_convert_weights,
        # ComfyUI Flux checkpoints ship without the optional attention biases.
        strict=False,
        # FluxConfig's own mapping reads the same BFL names this spec does, but
        # targets a different parameter layout (fused to_qkv, ff.linear_in,
        # time_guidance_embed) than FluxTransformer2DModel exposes. Layering the
        # two rewrites correctly resolved names into ones the model lacks.
        inherit_config_mapping=False,
    ),
)

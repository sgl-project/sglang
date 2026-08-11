# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import (
    is_embed_tokens,
    is_final_norm,
    is_layer,
)

# LTX-2.5 saved its text encoder with a transformers build that names the vision
# embedder differently from the installed one. Only the vision path is affected --
# text-to-video never runs it -- but the loader is strict about uninitialized
# weights, so map the names rather than let them go missing.
GEMMA_4_UNIFIED_PARAM_NAMES_MAPPING: dict[str, str] = {
    r"^model\.embed_vision\.embedding_projection\.(.*)$": r"model.embed_vision.multimodal_embedder.embedding_projection.\1",
    r"^model\.vision_embedder\.(.*)$": r"model.embed_vision.\1",
}


@dataclass
class Gemma4UnifiedArchConfig(TextEncoderArchConfig):
    """Gemma-4-Unified text encoder used by LTX-2.5.

    Like `Gemma3ArchConfig`, the actual module is instantiated by transformers
    from the repo's `text_encoder/config.json`
    (`Gemma4UnifiedForConditionalGeneration`); this config carries tokenization
    and sharding metadata.

    LTX-2.5 consumes all 48 hidden layers plus the embedding output, which is why
    the connector's `text_proj_in_factor` is 49 and `caption_channels` is 3840.
    """

    param_names_mapping: dict = field(
        default_factory=lambda: dict(GEMMA_4_UNIFIED_PARAM_NAMES_MAPPING)
    )

    hidden_size: int = 3840
    num_hidden_layers: int = 48
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 262144
    hidden_state_skip_layer: int = 2
    text_len: int = 1024

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "0"),  # type: ignore
            (".gate_up_proj", ".up_proj", "1"),  # type: ignore
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_layer, is_embed_tokens, is_final_norm]
    )


@dataclass
class Gemma4UnifiedConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(default_factory=Gemma4UnifiedArchConfig)

    prefix: str = "gemma_4_unified"

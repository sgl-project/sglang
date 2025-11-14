# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_transformer_blocks(n, m):
    return "transformer_blocks" in n and n.split(".")[-1].isdigit()


@dataclass
class StepVideoArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [is_transformer_blocks]
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # transformer block
            r"^transformer_blocks\.(\d+)\.norm1\.(weight|bias)$": r"transformer_blocks.\1.norm1.norm.\2",
            r"^transformer_blocks\.(\d+)\.norm2\.(weight|bias)$": r"transformer_blocks.\1.norm2.norm.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.weight$": r"transformer_blocks.\1.ff.fc_in.weight",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.weight$": r"transformer_blocks.\1.ff.fc_out.weight",
            # adanorm block
            r"^adaln_single\.emb\.timestep_embedder\.linear_1\.(weight|bias)$": r"adaln_single.emb.mlp.fc_in.\1",
            r"^adaln_single\.emb\.timestep_embedder\.linear_2\.(weight|bias)$": r"adaln_single.emb.mlp.fc_out.\1",
            # caption projection
            r"^caption_projection\.linear_1\.(weight|bias)$": r"caption_projection.fc_in.\1",
            r"^caption_projection\.linear_2\.(weight|bias)$": r"caption_projection.fc_out.\1",
        }
    )

    num_attention_heads: int = 48
    attention_head_dim: int = 128
    in_channels: int = 64
    out_channels: int | None = 64
    num_layers: int = 48
    dropout: float = 0.0
    patch_size: int = 1
    norm_type: str = "ada_norm_single"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6
    caption_channels: int | list[int] | tuple[int, ...] | None = field(
        default_factory=lambda: [6144, 1024]
    )
    attention_type: str | None = "torch"
    use_additional_conditions: bool | None = False
    exclude_lora_layers: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.out_channels = (
            self.in_channels if self.out_channels is None else self.out_channels
        )
        self.num_channels_latents = self.out_channels


@dataclass
class StepVideoConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=StepVideoArchConfig)

    prefix: str = "StepVideo"

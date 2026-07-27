# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


def is_blocks(name: str, module) -> bool:
    return "blocks" in name and str.isdigit(name.split(".")[-1])


@dataclass
class LingBotVideoMoEArchConfig(DiTArchConfig):
    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_blocks])

    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)

    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16

    hidden_size: int = 2048
    num_attention_heads: int = 16
    depth: int = 48
    intermediate_size: int = 6144
    text_dim: int = 2560
    freq_dim: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 256.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (4096, 512, 512)

    qkv_bias: bool = False
    out_bias: bool = True
    patch_embed_bias: bool = True
    timestep_mlp_bias: bool = True

    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    n_shared_experts: int = 1
    score_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 4
    topk_group: int = 2
    routed_scaling_factor: float = 2.5

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = self.out_channels


@dataclass
class LingBotVideoMoEConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=LingBotVideoMoEArchConfig)
    prefix: str = "LingBotVideo"

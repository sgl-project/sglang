# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

GLOBAL_ORDER = ["register", "anchor", "text"]


@dataclass
class PraxisArchConfig(DiTArchConfig):
    dim: int = 3072
    timestep_dim: int = 128
    num_registers: int = 4
    head_dim: int = 128
    num_kv_heads: int = 8
    attention_backend: str = "flash"
    dense_ffn_hidden: int = 11520
    num_routed_experts: int = 128
    top_k: int = 8
    expert_hidden: int = 1280
    routed_scaling_factor: float = 2.5
    rope_base: int = 26_000_000
    rope_effective_dim: int = 64
    rope_table_size: int = 1_000_000
    text_dim: int = 2048
    latent_channels: int = 192
    encoder_depth: int = 14
    encoder_dense_blocks: list[int] = field(default_factory=lambda: [0, 1])
    tap_blocks: list[int] = field(default_factory=lambda: [0, 2, 4, 6, 7, 9, 11, 13])
    decoder_depth: int = 6
    decoder_dense_blocks: list[int] = field(default_factory=lambda: [0])
    decoder_patch: list[int] = field(default_factory=lambda: [2, 2])
    decoder_channels: int = 48
    encoder_zero_add: bool = True
    decoder_zero_add: bool = True
    expert_dim: int = 1024
    expert_layers: int = 16
    expert_heads: int = 16
    expert_head_dim: int = 64
    expert_trunk_dim: int = 3072
    action_dim: int = 40
    proprio_dim: int = 10
    num_future_per_chunk: int = 8
    max_chunks: int = 16
    expert_rope_dim: int = 64
    expert_rope_effective_dim: int = 64
    expert_rope_base: int = 26_000_000
    expert_rope_table_size: int = 1_000_000
    expert_tap_norm: bool = True
    clean_only_conditioning: bool = False
    head_plus_zero: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.dim
        self.num_attention_heads = self.dim // self.head_dim
        self.num_channels_latents = self.latent_channels

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any]) -> "PraxisArchConfig":
        backbone, expert = model_config["backbone"], model_config["action_expert"]
        _require_supported(backbone=backbone, expert=expert)
        moe, rope, enc, dec = (
            backbone["moe"],
            backbone["rope"],
            backbone["encoder"],
            backbone["decoder"],
        )
        return cls(
            dim=backbone["dim"],
            timestep_dim=backbone["timestep_dim"],
            num_registers=backbone["num_registers"],
            head_dim=backbone["attention"]["head_dim"],
            num_kv_heads=backbone["attention"]["num_kv_heads"],
            attention_backend=backbone["attention"]["backend"],
            dense_ffn_hidden=backbone["dense_ffn_hidden"],
            num_routed_experts=moe["num_routed_experts"],
            top_k=moe["top_k_routed"],
            expert_hidden=moe["expert_hidden"],
            routed_scaling_factor=moe["routed_scaling_factor"],
            rope_base=rope["base"],
            rope_effective_dim=rope["effective_dim"],
            rope_table_size=rope["efficient_cache_max"],
            text_dim=enc["input_projections"]["TextEncodingTensor"]["c_in"],
            latent_channels=enc["input_projections"]["VideoTensor"]["c_in"],
            encoder_depth=enc["depth"],
            encoder_dense_blocks=list(enc["dense_blocks"]),
            tap_blocks=list(enc["tap_blocks"]),
            decoder_depth=dec["depth"],
            decoder_dense_blocks=list(dec["dense_blocks"]),
            decoder_patch=list(dec["unpatch_factor"][1:]),
            decoder_channels=dec["input_projections"]["VideoTensor"]["c_in"],
            encoder_zero_add=enc["positive_zero_residual_add"],
            decoder_zero_add=dec["positive_zero_residual_add"],
            expert_dim=expert["dim"],
            expert_layers=expert["num_layers"],
            expert_heads=expert["num_heads"],
            expert_head_dim=expert["head_dim"],
            expert_trunk_dim=expert["trunk_dim"],
            action_dim=expert["action_patch_dim"],
            proprio_dim=expert["proprio_patch_dim"],
            num_future_per_chunk=expert["num_future_per_chunk"],
            max_chunks=expert["max_chunks"],
            expert_rope_dim=expert["rope"]["dim"],
            expert_rope_effective_dim=expert["rope"]["effective_dim"],
            expert_rope_base=expert["rope"]["base"],
            expert_rope_table_size=expert["rope"]["table_size"],
            expert_tap_norm=expert["tap_norm"],
            clean_only_conditioning=expert["clean_only_conditioning"],
            head_plus_zero=expert["head_plus_zero"],
        )


def _require_supported(*, backbone: dict[str, Any], expert: dict[str, Any]) -> None:
    checks = {
        "backbone.layout": backbone.get("layout") == "chunk_causal",
        "backbone.modulation_dtype": backbone["modulation_dtype"] == "mixed",
        "backbone.attention.rope_after_norm": not backbone["attention"][
            "rope_after_norm"
        ],
        "backbone.attention.backend": backbone["attention"]["backend"]
        in ("flash", "triton"),
        "backbone.moe.router_activation": backbone["moe"]["router_activation"]
        == "sigmoid",
        "action_expert.rope": expert["rope"]["kind"] == "trunk"
        and expert["rope"]["order"] == "before_norm",
        "action_expert.global_order": list(expert["global_order"]) == GLOBAL_ORDER,
    }
    unsupported = [name for name, ok in checks.items() if not ok]
    if unsupported:
        raise ValueError(f"unsupported Praxis-1 variant: {unsupported}")


@dataclass
class PraxisDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=PraxisArchConfig)
    prefix: str = "praxis"

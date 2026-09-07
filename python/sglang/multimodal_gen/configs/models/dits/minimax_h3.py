# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64
MINIMAX_H3_ADALN_MODALITY_NUM = 3


@dataclass
class MiniMaxH3DiTArchConfig(DiTArchConfig):
    # accept Diffusers/PEFT aliases in the source-to-native model mapping
    # H3 fuses Q/K/V, so split projections are stacked for the fused LoRA layer
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(.*)\.weight_scale$": r"\1.weight_scale_inv",
            r"^(.*\.lora_[AB])\.[^.]+$": r"\1",
            r"^base_model\.model\.(.*\.lora_[AB])$": r"\1",
            r"^transformer\.(.*\.lora_[AB])$": r"\1",
            r"^proj_in\.(.*)$": r"video_patch_proj.\1",
            r"^audio_proj_in\.(.*)$": r"audio_patch_proj.\1",
            r"^context_embedder\.(.*)$": r"condition_proj.\1",
            r"^time_embedder\.linear_1\.(.*)$": r"time_embedder.proj_in.\1",
            r"^time_embedder\.linear_2\.(.*)$": r"time_embedder.proj_out.\1",
            r"^time_embedder\.table$": r"adaln_t_table",
            r"^norm_out\.norm\.(.*)$": r"final_layer.norm.\1",
            r"^norm_out\.folded_bias$": r"final_layer.adaln_proj.linear.bias",
            r"^norm_out\.linear\.(.*)$": r"final_layer.adaln_proj.linear.\1",
            r"^proj_out\.(.*)$": r"final_layer.video_out.\1",
            r"^audio_proj_out\.(.*)$": r"final_layer.audio_out.\1",
            r"^transformer_blocks\.(\d+)\.adaln_proj\.linear\.(.*)$": r"blocks.\1.adaln_proj.linear.\2",
            r"^transformer_blocks\.(\d+)\.adaln_proj\.folded_bias$": r"blocks.\1.adaln_proj.linear.bias",
            r"^transformer_blocks\.(\d+)\.attn\.to_q\.(.*)$": (
                r"blocks.\1.attn.qkv_proj.\2",
                0,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_k\.(.*)$": (
                r"blocks.\1.attn.qkv_proj.\2",
                1,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_v\.(.*)$": (
                r"blocks.\1.attn.qkv_proj.\2",
                2,
                3,
            ),
            r"^transformer_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$": r"blocks.\1.attn.out_proj.\2",
            r"^transformer_blocks\.(\d+)\.attn\.to_gate_compress\.(.*)$": r"blocks.\1.attn.to_gate_compress.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_q\.(.*)$": r"blocks.\1.attn.q_norm.\2",
            r"^transformer_blocks\.(\d+)\.attn\.norm_k\.(.*)$": r"blocks.\1.attn.k_norm.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$": r"blocks.\1.mlp.fc1.\2",
            r"^transformer_blocks\.(\d+)\.ff\.net\.2\.(.*)$": r"blocks.\1.mlp.fc2.\2",
            r"^transformer_blocks\.(\d+)\.norm([12])\.(.*)$": r"blocks.\1.norm\2.\3",
            r"^token_refiner\.final_norm\.(.*)$": r"token_refiner.final_norm.\1",
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_q\.(.*)$": (
                r"token_refiner.blocks.\1.attn.qkv_proj.\2",
                0,
                3,
            ),
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_k\.(.*)$": (
                r"token_refiner.blocks.\1.attn.qkv_proj.\2",
                1,
                3,
            ),
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_v\.(.*)$": (
                r"token_refiner.blocks.\1.attn.qkv_proj.\2",
                2,
                3,
            ),
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.to_out\.0\.(.*)$": r"token_refiner.blocks.\1.attn.out_proj.\2",
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.norm_q\.(.*)$": r"token_refiner.blocks.\1.attn.q_norm.\2",
            r"^token_refiner\.refiner_blocks\.(\d+)\.attn\.norm_k\.(.*)$": r"token_refiner.blocks.\1.attn.k_norm.\2",
            r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.0\.proj\.(.*)$": r"token_refiner.blocks.\1.mlp.fc1.\2",
            r"^token_refiner\.refiner_blocks\.(\d+)\.ff\.net\.2\.(.*)$": r"token_refiner.blocks.\1.mlp.fc2.\2",
            r"^token_refiner\.refiner_blocks\.(\d+)\.norm([12])\.(.*)$": r"token_refiner.blocks.\1.norm\2.\3",
        }
    )

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    # Pruned checkpoints replace the timestep MLP with a sampled AdaLN curve.
    adaln_curve_grid: int | None = None
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    checkpoint_uses_diffusers_layout: bool = False
    adaln_affine_input_dim: int | None = None
    has_gate_compress: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        self.num_channels_latents = self.latents_dim


@dataclass
class MiniMaxH3DiTConfig(DiTConfig):
    arch_config: MiniMaxH3DiTArchConfig = field(default_factory=MiniMaxH3DiTArchConfig)

    def update_model_arch(self, source_model_dict: dict) -> None:
        aliases = {
            "num_refiner_layers": "token_refiner_num_layers",
            "ffn_dim": "ffn_hidden_size",
            "in_channels": "latents_dim",
            "audio_in_channels": "audio_latents_dim",
            "freq_dim": "timestep_input_dim",
            "time_embed_hidden_dim": "time_embed_hidden_size",
            "rope_freq_dim": "rope_inv_freq_len",
        }
        model_dict = {
            aliases.get(key, key): value for key, value in source_model_dict.items()
        }
        if source_model_dict.get("_class_name") == "MiniMaxH3PrunedTransformer3DModel":
            model_dict["adaln_affine_input_dim"] = source_model_dict["time_embed_dim"]
            model_dict["time_embed_dim"] = source_model_dict["adaln_rank"]
            model_dict["adaln_curve_grid"] = source_model_dict["time_table_size"]
        super().update_model_arch(model_dict)


__all__ = [
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
    "MiniMaxH3DiTConfig",
]

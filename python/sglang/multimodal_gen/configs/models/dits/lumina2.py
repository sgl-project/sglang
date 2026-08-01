# SPDX-License-Identifier: Apache-2.0
#
# Architecture and model configuration for Lumina-Image-2.0 DiT (NextDiT).
#
# Defaults correspond to Alpha-VLLM/Lumina-Image-2.0 (2B), read off the published
# transformer/config.json and safetensors headers.
#
# Reference: https://arxiv.org/abs/2503.21758

from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_lumina2_layer


@dataclass
class Lumina2ArchConfig(DiTArchConfig):
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int | None = None  # checkpoint stores null -> defaults to in_channels
    hidden_size: int = 2304
    num_layers: int = 26
    num_refiner_layers: int = 2
    num_attention_heads: int = 24
    num_kv_heads: int = 8  # GQA
    norm_eps: float = 1e-5
    cap_feat_dim: int = 2304  # Gemma-2-2B hidden size
    # SwiGLU inner width = round_up(4 * hidden_size * (ffn_dim_multiplier or 1),
    # multiple_of); a null multiplier gives 9216 = linear_1.weight.shape[0].
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    axes_dim_rope: Tuple[int, int, int] = (32, 32, 32)
    axes_lens: Tuple[int, int, int] = (300, 512, 512)
    sample_size: int = 128
    scaling_factor: float = 1.0  # present in config.json; unused by the model

    qk_norm: bool = True
    # Absent from transformer/config.json; diffusers hardcodes theta=10000 at the
    # Lumina2RotaryPosEmbed construction site.
    rope_theta: float = 10000.0
    # FlowMatchEuler num_train_timesteps; the DiT rescales timesteps by it.
    t_scale: float = 1000.0

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_lumina2_layer])

    # diffusers Lumina2Transformer2DModel checkpoint keys -> runtime module
    # names. ffn_norm1 / ffn_norm2 / x_embedder / caption_embedder need no rule.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*)\.attn\.to_q\.weight$": (r"\1.attention.to_qkv.weight", 0, 3),
            r"(.*)\.attn\.to_k\.weight$": (r"\1.attention.to_qkv.weight", 1, 3),
            r"(.*)\.attn\.to_v\.weight$": (r"\1.attention.to_qkv.weight", 2, 3),
            # NOTE: these resolve, but the fused merge stacks shards into
            # (N, out, r) and GQA makes them unequal, so an attention LoRA
            # raises at load. Kept anyway: dropping them targets to_q/to_k/to_v,
            # which the fused model lacks, and the adapter silently no-ops.
            r"(.*)\.attn\.to_q\.(lora_A|lora_B)$": (r"\1.attention.to_qkv.\2", 0, 3),
            r"(.*)\.attn\.to_k\.(lora_A|lora_B)$": (r"\1.attention.to_qkv.\2", 1, 3),
            r"(.*)\.attn\.to_v\.(lora_A|lora_B)$": (r"\1.attention.to_qkv.\2", 2, 3),
            r"(.*)\.attn\.(norm_q|norm_k|to_out)\.": r"\1.attention.\2.",
            r"(.*)\.feed_forward\.linear_1\.weight$": (
                r"\1.feed_forward.w13.weight",
                0,
                2,
            ),
            r"(.*)\.feed_forward\.linear_3\.weight$": (
                r"\1.feed_forward.w13.weight",
                1,
                2,
            ),
            r"(.*)\.feed_forward\.linear_1\.(lora_A|lora_B)$": (
                r"\1.feed_forward.w13.\2",
                0,
                2,
            ),
            r"(.*)\.feed_forward\.linear_3\.(lora_A|lora_B)$": (
                r"\1.feed_forward.w13.\2",
                1,
                2,
            ),
            r"(.*)\.feed_forward\.linear_2\.": r"\1.feed_forward.w2.",
            r"(.*)\.norm1\.linear\.": r"\1.adaLN_modulation.1.",
            # norm1.weight (bare) is the context_refiner's unmodulated variant.
            r"(.*)\.norm1\.norm\.": r"\1.attention_norm1.",
            r"(.*)\.norm1\.weight$": r"\1.attention_norm1.weight",
            r"(.*)\.norm2\.weight$": r"\1.attention_norm2.weight",
            r"^time_caption_embed\.timestep_embedder\.linear_1\.": r"time_caption_embed.timestep_embedder.mlp.0.",
            r"^time_caption_embed\.timestep_embedder\.linear_2\.": r"time_caption_embed.timestep_embedder.mlp.2.",
            r"^norm_out\.linear_1\.": r"norm_out.adaLN_modulation.1.",
            r"^norm_out\.linear_2\.": r"norm_out.linear.",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.num_channels_latents = self.in_channels

        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim != sum(self.axes_dim_rope):
            raise ValueError(
                f"axes_dim_rope {self.axes_dim_rope} must sum to the attention head "
                f"dim {head_dim} (hidden_size {self.hidden_size} / "
                f"num_attention_heads {self.num_attention_heads})"
            )


@dataclass
class Lumina2Config(DiTConfig):
    arch_config: Lumina2ArchConfig = field(default_factory=Lumina2ArchConfig)
    prefix: str = "Lumina2"

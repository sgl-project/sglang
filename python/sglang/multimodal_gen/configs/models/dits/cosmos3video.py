# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 video DiT architecture configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in


def is_layers(n: str, m) -> bool:
    return is_module_list_entry_in(n, ("layers", "gen_layers"))


def _build_cosmos3_param_names_mapping() -> dict:
    """Map diffusers-format Cosmos3 weights to the sglang model namespace.

    Source keys (diffusers transformer ckpt) → target keys (sglang model):
        embed_tokens.weight                                    -> language_model.embed_tokens.weight
        layers.X.input_layernorm.weight                        -> language_model.layers.X.input_layernorm.weight
        layers.X.input_layernorm_moe_gen.weight                -> gen_layers.X.input_layernorm.weight
        layers.X.self_attn.{to_q,to_k,to_v}.weight            -> language_model.layers.X.self_attn.to_qkv.weight (concat dim 0)
        layers.X.self_attn.{add_q,add_k,add_v}_proj.weight    -> gen_layers.X.cross_attention.to_qkv.weight (concat dim 0)
        layers.X.mlp.{gate,up}_proj.weight                     -> language_model.layers.X.mlp.gate_up_proj.weight (concat dim 0)
        layers.X.mlp_moe_gen.{gate,up}_proj.weight             -> gen_layers.X.mlp.gate_up_proj.weight (concat dim 0)
        norm_moe_gen.weight                                    -> norm_moe_gen.weight
        time_embedder.linear_{1,2}.weight                      -> (pass-through)
        proj_in.weight, proj_out.weight                        -> (pass-through)
        vae2llm.weight                                         -> proj_in.weight  (FP8 ckpt alias)
        llm2vae.weight                                         -> proj_out.weight (FP8 ckpt alias)

    GEN patterns (`*_moe_gen`, `add_*`, `to_add_out`, `norm_added_*`) must
    precede the UND catch-all so the catch-all can't claim GEN keys.
    `norm.weight` and `lm_head.weight` are inherited from Qwen3-VL
    pretraining and not used at inference; both are skipped.
    Audio and action keys (``audio_proj_*``, ``action_proj_*``, modality
    embeds) pass through unchanged.
    """
    return {
        # Inherited from Qwen3-VL pretraining; unused at diffusion inference.
        r"^lm_head\.weight$": "",
        r"^norm\.weight$": "",
        # Top-level norms / embeddings.
        r"^norm_moe_gen\.(.*)$": r"norm_moe_gen.\1",
        r"^embed_tokens\.(.*)$": r"language_model.embed_tokens.\1",
        # FP8 checkpoint aliases for the latent projection layers.
        r"^vae2llm\.(.*)$": r"proj_in.\1",
        r"^llm2vae\.(.*)$": r"proj_out.\1",
        # GEN pathway: per-layer (must run before the UND catch-all below).
        # Q/K/V merge into MergedColumnParallelLinear to_qkv (concat order: Q, K, V).
        r"^layers\.(\d+)\.self_attn\.add_q_proj\.(.*)$": (
            r"gen_layers.\1.cross_attention.to_qkv.\2",
            0,
            3,
        ),
        r"^layers\.(\d+)\.self_attn\.add_k_proj\.(.*)$": (
            r"gen_layers.\1.cross_attention.to_qkv.\2",
            1,
            3,
        ),
        r"^layers\.(\d+)\.self_attn\.add_v_proj\.(.*)$": (
            r"gen_layers.\1.cross_attention.to_qkv.\2",
            2,
            3,
        ),
        r"^layers\.(\d+)\.self_attn\.to_add_out\.(.*)$": r"gen_layers.\1.cross_attention.to_out.\2",
        r"^layers\.(\d+)\.self_attn\.norm_added_q\.(.*)$": r"gen_layers.\1.cross_attention.norm_q.\2",
        r"^layers\.(\d+)\.self_attn\.norm_added_k\.(.*)$": r"gen_layers.\1.cross_attention.norm_k.\2",
        r"^layers\.(\d+)\.input_layernorm_moe_gen\.(.*)$": r"gen_layers.\1.input_layernorm.\2",
        r"^layers\.(\d+)\.post_attention_layernorm_moe_gen\.(.*)$": r"gen_layers.\1.post_attention_layernorm.\2",
        # GEN MLP gate/up merge into MergedColumnParallelLinear gate_up_proj.
        # Must precede the mlp_moe_gen catch-all below.
        r"^layers\.(\d+)\.mlp_moe_gen\.gate_proj\.(.*)$": (
            r"gen_layers.\1.mlp.gate_up_proj.\2",
            0,
            2,
        ),
        r"^layers\.(\d+)\.mlp_moe_gen\.up_proj\.(.*)$": (
            r"gen_layers.\1.mlp.gate_up_proj.\2",
            1,
            2,
        ),
        r"^layers\.(\d+)\.mlp_moe_gen\.(.*)$": r"gen_layers.\1.mlp.\2",
        # UND pathway: Q/K/V merge into to_qkv; remaining attention keys
        # (to_out, norm_q, norm_k) and layernorms pass through the catch-all.
        r"^layers\.(\d+)\.self_attn\.to_q\.(.*)$": (
            r"language_model.layers.\1.self_attn.to_qkv.\2",
            0,
            3,
        ),
        r"^layers\.(\d+)\.self_attn\.to_k\.(.*)$": (
            r"language_model.layers.\1.self_attn.to_qkv.\2",
            1,
            3,
        ),
        r"^layers\.(\d+)\.self_attn\.to_v\.(.*)$": (
            r"language_model.layers.\1.self_attn.to_qkv.\2",
            2,
            3,
        ),
        # UND MLP gate/up merge into MergedColumnParallelLinear gate_up_proj.
        # Must precede the layers catch-all below.
        r"^layers\.(\d+)\.mlp\.gate_proj\.(.*)$": (
            r"language_model.layers.\1.mlp.gate_up_proj.\2",
            0,
            2,
        ),
        r"^layers\.(\d+)\.mlp\.up_proj\.(.*)$": (
            r"language_model.layers.\1.mlp.gate_up_proj.\2",
            1,
            2,
        ),
        # UND pathway: layernorms + remaining attention/mlp keys pass through
        # under language_model.layers namespace.
        r"^layers\.(\d+)\.(.*)$": r"language_model.layers.\1.\2",
    }


@dataclass
class Cosmos3VideoArchConfig(DiTArchConfig):
    """Architecture config for Cosmos3 Omni Transformer.

    Cosmos3 uses a dual-pathway design:
    - Understanding (UND): Causal self-attention for text tokens
    - Generation (GEN): Cross-attention from visual to cached UND K/V

    Field names mirror the diffusers ``transformer/config.json`` so values
    flow through ``update_model_arch`` without translation. Defaults are
    Qwen3-8B-Instruct-derived and overridden by the checkpoint at load time.
    """

    _fsdp_shard_conditions: list = field(default_factory=lambda: [is_layers])

    # Transformer architecture
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128
    intermediate_size: int = 12288

    # Latent space configuration
    latent_patch_size: int = 2
    latent_channel: int = 48
    out_channels: int = 48

    # RoPE configuration (Qwen3-VL 3D mRoPE: temporal, height, width)
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    rope_theta: float = 5000000.0
    # Populated from rope_scaling in the diffusers config when present.
    rope_scaling: dict | None = None

    # Temporal configuration
    base_fps: float = 24.0
    temporal_compression_factor: int = 4
    unified_3d_mrope_temporal_modality_margin: int = 15000

    # Audio (sound) modality
    sound_dim: int = 64
    sound_latent_fps: float = 25.0
    temporal_compression_factor_sound: int = 1

    # Action modality
    action_gen: bool = False
    action_dim: int = 64
    num_embodiment_domains: int = 32

    # Timestep embedding
    timestep_scale: float = 0.001
    frequency_embedding_size: int = 256

    # Vocab size (Qwen3-VL tokenizer)
    vocab_size: int = 151936

    # RMSNorm epsilon
    rms_norm_eps: float = 1e-6

    # Weight mapping from checkpoint to model
    param_names_mapping: dict = field(
        default_factory=_build_cosmos3_param_names_mapping
    )
    reverse_param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    # FP8 checkpoint quantization_config.ignore uses checkpoint module names;
    # translate them to model names so _is_layer_ignored matches correctly.
    quant_ignore_remap: dict = field(
        default_factory=lambda: {"vae2llm": "proj_in", "llm2vae": "proj_out"}
    )

    def __post_init__(self):
        super().__post_init__()
        # Diffusers configs nest the mrope sizes under `rope_scaling`; lift it.
        if isinstance(self.rope_scaling, dict) and "mrope_section" in self.rope_scaling:
            self.mrope_section = tuple(self.rope_scaling["mrope_section"])
        self.in_channels = self.latent_channel
        self.num_channels_latents = self.out_channels
        # Patch latent dimension: (patch_size^2) * latent_channel
        self.patch_latent_dim = (self.latent_patch_size**2) * self.latent_channel


@dataclass
class Cosmos3VideoConfig(DiTConfig):
    """DiT config wrapper for Cosmos3 Video model."""

    arch_config: DiTArchConfig = field(default_factory=Cosmos3VideoArchConfig)
    prefix: str = "Cosmos3"

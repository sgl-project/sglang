# SPDX-License-Identifier: Apache-2.0
#
# Wan-Animate-2 DiT config. The checkpoint is the Wan2.2-I2V-14B backbone with no
# extra parameter tensors, so WanVideoArchConfig is reused with in_channels=36,
# image_dim=1280, added_kv_proj_dim=5120 and a param-name remap.
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig

# Checkpoint block params use the official Wan names nested under `blocks.{i}.block.*`.
# The loader applies mapping rules in insertion order, re-scanning after each hit, so the
# extras below chain into the reused Wan lora (official -> hf) and main (hf -> native) maps.
_WAN = WanVideoArchConfig()


_WAN_ANIMATE_2_EXTRA_MAPPING: dict[str, str] = {
    # Strip the `.block` infix so the reused Wan rules match.
    r"^blocks\.(\d+)\.block\.(.*)$": r"blocks.\1.\2",
    # Self-attn / cross-attn qk-norms (not covered by the Wan lora map).
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
    # Cross-attn I2V image branch -> WanI2VCrossAttention add_k_proj/add_v_proj/norm_added_k.
    r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": r"blocks.\1.attn2.add_k_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": r"blocks.\1.attn2.add_v_proj.\2",
    r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": r"blocks.\1.attn2.norm_added_k.\2",
    # The Diffusers checkpoint keeps the official self_attn/cross_attn prefixes but names
    # the projections the diffusers way (to_q/to_k/to_v/to_out.0, add_*_proj).
    r"^blocks\.(\d+)\.self_attn\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
    r"^blocks\.(\d+)\.self_attn\.(to_q|to_k|to_v)\.(.*)$": r"blocks.\1.\2.\3",
    r"^blocks\.(\d+)\.cross_attn\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
    r"^blocks\.(\d+)\.cross_attn\.(to_q|to_k|to_v|add_k_proj|add_v_proj|norm_added_k)\.(.*)$": r"blocks.\1.attn2.\2.\3",
    # `norm3` is the block's only affine LayerNorm and feeds cross-attn; the reused rules
    # only know its hf name (`norm2`).
    r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
    # Per-block 6-way modulation -> scale_shift_table.
    r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    # Output head (`head.norm` is param-free; no-op if absent).
    r"^head\.head\.(.*)$": r"proj_out.\1",
    r"^head\.modulation$": r"scale_shift_table",
    r"^head\.norm\.(.*)$": r"norm_out.\1",
    # Embedders in official naming; hf-named checkpoints are covered by the reused rules.
    r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
    r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
    r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
    r"^img_emb\.proj\.0\.(.*)$": r"condition_embedder.image_embedder.norm1.\1",
    r"^img_emb\.proj\.1\.(.*)$": r"condition_embedder.image_embedder.ff.fc_in.\1",
    r"^img_emb\.proj\.3\.(.*)$": r"condition_embedder.image_embedder.ff.fc_out.\1",
    r"^img_emb\.proj\.4\.(.*)$": r"condition_embedder.image_embedder.norm2.\1",
    # patch_embedding, self/cross-attn q/k/v/o and ffn.* are covered by the reused Wan maps.
}


@dataclass
class WanAnimate2ArchConfig(WanVideoArchConfig):
    """Wan2.2-I2V-14B backbone dims (inherited) with the Wan-Animate-2 I/O + remap."""

    # Extras first, then Wan lora (official -> hf), then Wan main (hf -> native).
    param_names_mapping: dict[str, str] = field(
        default_factory=lambda: {
            **_WAN_ANIMATE_2_EXTRA_MAPPING,
            **_WAN.lora_param_names_mapping,
            **_WAN.param_names_mapping,
        }
    )
    # Only used for re-serialization, which the inference path does not exercise.
    reverse_param_names_mapping: dict[str, str] = field(
        default_factory=lambda: dict(_WAN.reverse_param_names_mapping)
    )
    # LoRA adapters use the official Wan names; the loader applies this before
    # param_names_mapping.
    lora_param_names_mapping: dict[str, str] = field(
        default_factory=lambda: dict(_WAN.lora_param_names_mapping)
    )

    # in_channels: 16 noise + 20 conditioning (4-ch i2v mask + 16-ch cond latent).
    # image_dim: CLIP (XLM-RoBERTa ViT-H/14) embed dim -> image_embedder.
    # added_kv_proj_dim -> WanI2VCrossAttention (add_k_proj/add_v_proj/norm_added_k).
    in_channels: int = 36
    out_channels: int = 16
    image_dim: int | None = 1280
    added_kv_proj_dim: int | None = 5120
    num_layers: int = 40

    # In-context mechanism (no extra params): reference-video RoPE offsets and the flex-attention
    # score_mod, consumed by forward_ref / forward_gen. The refer_* / log_scale names
    # match the official and diffusers transformer configs; do not rename. In those
    # names "refer" means the reference video (the motion source), not the reference image.
    refer_offset_t: int = 1
    refer_offset_h: int = 0
    refer_offset_w: int = (
        -1
    )  # <0: resolved to the gen grid width in tokens at runtime (official convention)
    refer_stride: int = 1
    # Additive attention-score bias on the keys of generation frame 1 (official name);
    # 0.0 disables it. Read from transformer/config.json when present; the base weights omit it.
    log_scale: float = 0.0
    # The image-embedding cross-attention (added_kv_proj_dim) is always built, as in the
    # official model; the checkpoint config states the same and nothing else reads this.
    use_img_emb: bool = True

    def __post_init__(self) -> None:
        # transformer/config.json uses the official Wan names for the backbone dims;
        # update_model_arch parks unknown keys in extra_attrs, so translate them here.
        extras = self.extra_attrs
        if "num_heads" in extras:
            self.num_attention_heads = extras.pop("num_heads")
        if "dim" in extras:
            self.attention_head_dim = extras.pop("dim") // self.num_attention_heads
        if "in_dim" in extras:
            self.in_channels = extras.pop("in_dim")
        if "out_dim" in extras:
            self.out_channels = extras.pop("out_dim")
        if not self.use_img_emb:
            raise ValueError(
                "Wan-Animate-2 always builds the image-embedding cross-attention; "
                "a checkpoint with use_img_emb=false is a different model."
            )
        self.patch_size = tuple(self.patch_size)
        super().__post_init__()


@dataclass
class WanAnimate2Config(DiTConfig):
    arch_config: WanAnimate2ArchConfig = field(default_factory=WanAnimate2ArchConfig)
    prefix: str = "WanAnimate2"

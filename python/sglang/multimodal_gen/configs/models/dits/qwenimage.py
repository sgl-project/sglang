# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class QwenImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)

    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".to_added_qkv", ".add_q_proj", "q"),
            (".to_added_qkv", ".add_k_proj", "k"),
            (".to_added_qkv", ".add_v_proj", "v"),
        ]
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # ---- Nunchaku-specific extra metadata --------------------------------
            # NOTE: For MLP blocks (img_mlp / txt_mlp) we keep all SVDQ metadata
            # parameters (smooth_factor_orig, wcscales) because the MLPs
            # are implemented with Nunchaku's own SVDQW4A4Linear, which defines
            # these as real parameters and expects them to be loaded.
            # For other layers (e.g. attention projections that use SGLang's
            # LinearBase wrappers) we drop these extra parameters since the
            # corresponding modules do not materialize them.
            #
            # Keep Nunchaku metadata (as real parameters) for img_mlp / txt_mlp.
            # In Nunchaku Qwen-Image checkpoints these keys look like:
            #   transformer_blocks.N.img_mlp.net.0.proj.wcscales
            #   transformer_blocks.N.txt_mlp.net.0.proj.smooth_factor_orig
            #   transformer_blocks.N.img_mlp.net.2.wcscales
            #   transformer_blocks.N.txt_mlp.net.2.smooth_factor_orig
            # and they correspond to SVDQW4A4Linear parameters used by the fused MLP kernels.
            # We therefore preserve *all* smooth_factor_orig / wcscales under img_mlp/txt_mlp.
            r"(transformer_blocks\.\d+\.(img_mlp|txt_mlp)\..*\.(smooth_factor_orig|wcscales))$": r"\1",
            # Drop wtscale everywhere: it is a float (or scalar tensor) attribute,
            # not an nn.Parameter in Nunchaku's design, and is patched separately
            # from the checkpoint rather than loaded via FSDP parameter mapping.
            r".*\.wtscale$": r"",
            # ---- QKV fusion mappings for original (unquantized) diffusers -----
            # Map separate Q/K/V projections to fused to_qkv / to_added_qkv.
            r"(.*)\.to_q\.(weight|bias)$": (r"\1.to_qkv.\2", 0, 3),
            r"(.*)\.to_k\.(weight|bias)$": (r"\1.to_qkv.\2", 1, 3),
            r"(.*)\.to_v\.(weight|bias)$": (r"\1.to_qkv.\2", 2, 3),
            r"(.*)\.add_q_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 0, 3),
            r"(.*)\.add_k_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 1, 3),
            r"(.*)\.add_v_proj\.(weight|bias)$": (r"\1.to_added_qkv.\2", 2, 3),
            # ---- Nunchaku quantized checkpoint mappings ------------------------
            # add_qkv_proj -> to_added_qkv (Nunchaku uses add_qkv_proj, sglang uses to_added_qkv)
            r"(.*)\.add_qkv_proj\.(.+)$": r"\1.to_added_qkv.\2",
            # ---- LoRA mappings -------------------------------------------------
            r"^(transformer_blocks\.\d+\.attn\..*\.lora_[AB])\.default$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class QwenImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=QwenImageArchConfig)

    prefix: str = "qwenimage"

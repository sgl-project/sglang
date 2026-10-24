# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig

# HiDream-O1-Image ships a stock Qwen3-VL config, so the pixel-head geometry and
# the special token id below are not recoverable from config.json.
HIDREAM_O1_PATCH_SIZE = 32
HIDREAM_O1_IN_CHANNELS = 3
HIDREAM_O1_TMS_TOKEN_ID = 151673


@dataclass
class HiDreamO1ImageArchConfig(DiTArchConfig):
    # Defaults mirror HiDream-ai/HiDream-O1-Image's `text_config`; the loader
    # overwrites them from the checkpoint via `update_model_arch`.
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    hidden_act: str = "silu"
    attention_bias: bool = False
    # Interleaved mrope splits head_dim/2 into per-axis channel counts.
    mrope_section: tuple[int, int, int] = (24, 20, 20)

    patch_size: int = HIDREAM_O1_PATCH_SIZE
    in_channels: int = HIDREAM_O1_IN_CHANNELS
    tms_token_id: int = HIDREAM_O1_TMS_TOKEN_ID

    # Fused checkpoint layout: q/k/v -> qkv_proj and gate/up -> gate_up_proj.
    # The tuple is (replacement, merge_index, total_merged_params).
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"(.*)\.self_attn\.q_proj\.(weight|bias)$": (
                r"\1.self_attn.qkv_proj.\2",
                0,
                3,
            ),
            r"(.*)\.self_attn\.k_proj\.(weight|bias)$": (
                r"\1.self_attn.qkv_proj.\2",
                1,
                3,
            ),
            r"(.*)\.self_attn\.v_proj\.(weight|bias)$": (
                r"\1.self_attn.qkv_proj.\2",
                2,
                3,
            ),
            r"(.*)\.mlp\.gate_proj\.(weight|bias)$": (r"\1.mlp.gate_up_proj.\2", 0, 2),
            r"(.*)\.mlp\.up_proj\.(weight|bias)$": (r"\1.mlp.gate_up_proj.\2", 1, 2),
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # `rope_scaling` only exists when the config came from the checkpoint.
        rope_scaling = self.extra_attrs.get("rope_scaling")
        if rope_scaling is not None:
            self.mrope_section = tuple(rope_scaling["mrope_section"])
        rope_channels = self.head_dim // 2
        if sum(self.mrope_section) != rope_channels:
            raise ValueError(
                f"mrope_section {self.mrope_section} must sum to head_dim / 2 = "
                f"{rope_channels}"
            )
        # The interleave writes axis `i` into channels `i::3`, so an h or w
        # section wide enough to run off the end would be silently truncated by
        # the slice instead of raising. Only the t axis may absorb the tail.
        for axis in (1, 2):
            last_channel = axis + 3 * (self.mrope_section[axis] - 1)
            if last_channel >= rope_channels:
                raise ValueError(
                    f"mrope_section {self.mrope_section} does not fit the "
                    f"interleaved rope layout: axis {axis} needs channel "
                    f"{last_channel} of {rope_channels}"
                )
        # A "latent" here is one raw pixel patch, not a VAE channel vector.
        self.num_channels_latents = self.in_channels * self.patch_size**2


@dataclass
class HiDreamO1ImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HiDreamO1ImageArchConfig)

    prefix: str = "hidream_o1_image"

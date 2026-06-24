# Krea-2 (K2) single-stream MMDiT architecture config.
#
# Parameter names follow the released K2 checkpoint, so the MMDiT safetensors load
# without remapping (identity `param_names_mapping`).
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Krea2ArchConfig(DiTArchConfig):
    features: int = 6144  # hidden dim
    tdim: int = 256  # timestep embedding dim
    txtdim: int = 2560  # text-encoder hidden dim (Qwen3-VL-4B hidden_size)
    heads: int = 48
    kvheads: int = 12  # GQA 4:1
    multiplier: int = 4  # SwiGLU expansion multiplier
    layers: int = 28
    patch: int = 2
    channels: int = 16  # VAE latent channels
    bias: bool = False
    theta: float = 1e3  # RoPE theta
    txtlayers: int = 12  # number of text-encoder hidden-state layers fused by txtfusion
    txtheads: int = 20
    txtkvheads: int = 20

    # 3-axis RoPE split over head_dim=128: [global, h, w] = (32, 48, 48).
    axes_dims: Tuple[int, int, int] = (32, 48, 48)

    # Joint (text+image) sequence is padded to a multiple of this many tokens.
    seq_multiple_of: int = 256

    # Packed patch-token width (channels * patch**2); used by the VAE unpack path.
    in_channels: int = 64

    # BaseDiT-required instance attrs (overwritten in __post_init__).
    hidden_size: int = 6144
    num_attention_heads: int = 48
    num_channels_latents: int = 16

    # Module/parameter names match the released checkpoint, so weights load with an
    # identity mapping.
    param_names_mapping: dict = field(default_factory=dict)

    # Diffusers LoRA checkpoints prefix every DiT key with the pipeline component name
    # (e.g. transformer.transformer_blocks.0.attn.to_q.lora_A). Strip that prefix so the
    # LoRA keys line up with this model's module names. Only the LoRA path uses this; the
    # main checkpoint still loads with the identity param_names_mapping above.
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {r"^transformer\.": ""}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.hidden_size = self.features
        self.num_attention_heads = self.heads
        self.num_channels_latents = self.channels
        assert self.features % self.heads == 0
        assert (
            sum(self.axes_dims) == self.features // self.heads
        ), f"sum(axes_dims)={sum(self.axes_dims)} != head_dim={self.features // self.heads}"

    @property
    def head_dim(self) -> int:
        return self.features // self.heads

    @property
    def in_features_packed(self) -> int:
        """Patch-embed input width: channels * patch**2."""
        return self.channels * self.patch**2


@dataclass
class Krea2DitConfig(DiTConfig):
    arch_config: Krea2ArchConfig = field(default_factory=Krea2ArchConfig)
    prefix: str = "k2"

# SPDX-License-Identifier: Apache-2.0
"""FLUX 3 ``JointSingleSeq`` DiT architecture configuration.

The DiT routes every modality ("content stream") through its own mode blocks
before all streams and the text context share the joint single-stream blocks.
Streams are declared by ``in_channels``; ``sequence`` maps the model inputs
(``x_<name>``) to streams and fixes their order in the joint sequence.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Flux3ArchConfig(DiTArchConfig):
    in_channels: dict[str, int] = field(
        default_factory=lambda: {"video": 96, "video_cond": 96}
    )
    sequence: dict[str, str] = field(
        default_factory=lambda: {"x_video": "video", "x_video_cond": "video_cond"}
    )
    vec_in_dim: int | None = 768
    context_in_dim: int = 20480
    hidden_size: int = 3072
    num_attention_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 28
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 10000
    mlp_ratio: float = 3.0

    # Exported policies prefix the DiT tensors with ``dit.``; every block's
    # q/k/v/mlp_in projections are fused into one ``qkv_mlp`` weight.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^dit\.(.*)$": r"\1",
            r"^(.*)\.q_proj\.(.*)$": (r"\1.qkv_mlp.\2", 0, 4),
            r"^(.*)\.k_proj\.(.*)$": (r"\1.qkv_mlp.\2", 1, 4),
            r"^(.*)\.v_proj\.(.*)$": (r"\1.qkv_mlp.\2", 2, 4),
            r"^(.*)\.mlp_in\.(.*)$": (r"\1.qkv_mlp.\2", 3, 4),
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = set(self.sequence.values()) - set(self.in_channels)
        if unknown:
            raise ValueError(f"sequence names undeclared streams: {sorted(unknown)}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if sum(self.axes_dim) != self.hidden_size // self.num_attention_heads:
            raise ValueError("axes_dim must sum to the attention head dim")
        self.num_channels_latents = self.in_channels.get("video", 0)

    def with_streams(self, extra: dict[str, int]) -> "Flux3ArchConfig":
        """Declare additional streams ``name -> channels`` (appended to the sequence)."""
        self.in_channels = {**self.in_channels, **extra}
        self.sequence = {**self.sequence, **{f"x_{name}": name for name in extra}}
        self.__post_init__()
        return self


@dataclass
class Flux3DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Flux3ArchConfig)
    prefix: str = "flux3"

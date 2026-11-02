# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class NomasArchConfig(DiTArchConfig):
    """HY-Nomas-G/2-32324-MMDiT, extrinsics-attention inference variant."""

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_layers: int = 54
    in_channels: int = 32
    out_channels: int = 32
    text_states_dim: int = 3584
    patch_size: tuple[int, int, int] = (1, 1, 1)
    rope_dim_list: tuple[int, int, int] = (16, 56, 56)
    rope_theta: float = 256.0
    camera_chunk_size: int = 1
    # Upstream checkpoint names -> the shared MLP / ModulateProjection layers.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(.*)\.fc1\.(.*)$": r"\1.fc_in.\2",
            r"^(.*)\.fc2\.(.*)$": r"\1.fc_out.\2",
            r"^(.*)\.adaLN_modulation\.1\.(.*)$": r"\1.adaLN_modulation.linear.\2",
        }
    )

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden size must be divisible by head count")
        if sum(self.rope_dim_list) != self.hidden_size // self.num_attention_heads:
            raise ValueError("RoPE dimensions must sum to head dimension")
        if len(self.rope_dim_list) != 3 or any(
            d <= 0 or d % 2 for d in self.rope_dim_list
        ):
            raise ValueError("RoPE requires three positive even dimensions")
        if tuple(self.patch_size) != (1, 1, 1):
            raise ValueError(
                "Nomas cache executor currently requires patch_size=(1,1,1)"
            )
        self.num_channels_latents = self.in_channels


@dataclass
class NomasConfig(DiTConfig):
    arch_config: NomasArchConfig = field(default_factory=NomasArchConfig)
    prefix: str = "Nomas"

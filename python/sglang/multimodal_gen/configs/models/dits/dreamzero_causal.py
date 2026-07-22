# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class DreamZeroCausalWanArchConfig(DiTArchConfig):
    """DROID DreamZero causal Wan DiT architecture defaults.

    Architecture values follow the official DreamZero implementation:
    https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
    https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
    """

    model_type: str = "i2v"
    patch_size: tuple[int, int, int] = (1, 2, 2)
    frame_seqlen: int = 880
    text_len: int = 512
    in_dim: int = 36
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    max_chunk_size: int = 4
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    num_frame_per_block: int = 2
    action_dim: int = 32
    max_state_dim: int = 64
    hidden_size: int = 1024
    action_hidden_size: int = 64
    num_action_per_block: int = 24
    num_state_per_block: int = 1
    concat_first_frame_latent: bool = True
    rope_video_max_positions: tuple[int, int, int] = (1024, 1024, 1024)
    rope_action_max_positions: int = 10240
    rope_state_max_positions: int = 1024
    use_tensor_parallel: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
            ("rope_video_max_positions", min(self.rope_video_max_positions)),
            ("rope_action_max_positions", self.rope_action_max_positions),
            ("rope_state_max_positions", self.rope_state_max_positions),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.out_dim


@dataclass
class DreamZeroCausalWanConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=DreamZeroCausalWanArchConfig)

    prefix: str = "dreamzero_causal_wan"

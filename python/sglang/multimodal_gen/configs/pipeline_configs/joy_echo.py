# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Optional

from sglang.multimodal_gen.configs.models.dits.joy_echo import JoyEchoConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig


def _default_joy_echo_vae_config() -> LTXVideoVAEConfig:
    vae_config = LTXVideoVAEConfig()
    vae_config.arch_config.ltx_variant = "ltx_2_3"
    return vae_config


JOY_ECHO_DEFAULT_SIGMAS: tuple[float, ...] = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)


@dataclass
class JoyEchoPipelineConfig(LTX2PipelineConfig):
    """Pipeline configuration for JoyEcho long-video generation."""

    task_type: ModelTaskType = ModelTaskType.T2V
    dit_config: JoyEchoConfig = field(default_factory=JoyEchoConfig)
    vae_config: LTXVideoVAEConfig = field(default_factory=_default_joy_echo_vae_config)

    monolithic_checkpoint: Optional[str] = None
    gemma_model_path: str = "google/gemma-3-12b-it"

    default_sigmas: tuple[float, ...] = field(
        default_factory=lambda: JOY_ECHO_DEFAULT_SIGMAS
    )

    enable_memory_bank: bool = True
    memory_max_size: int = 7
    memory_num_fix_frames: int = 3
    memory_position_mode: str = "reference"

    audio_window_size: int = 96
    audio_mel_bins: int = 128
    audio_mel_hop_length: int = 160
    audio_n_fft: int = 1024
    audio_downsample_factor: int = 4
    audio_window_selection_mode: str = "max_response"

    memory_video_clip_num_frames: int = 9
    video_memory_frame_selection_mode: str = "center"

    late_layer_ratio: float = 0.7

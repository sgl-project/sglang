# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/NVlabs/LongLive

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig
from sglang.multimodal_gen.configs.models.dits.longlive2 import LongLive2VideoConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config


@dataclass
class LongLive2T2VConfig(Wan2_2_TI2V_5B_Config):

    is_causal: bool = True
    task_type: ModelTaskType = ModelTaskType.T2V

    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
    expand_timesteps: bool = False
    context_noise: int = 0

    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None

    dit_config: DiTConfig = field(default_factory=LongLive2VideoConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_decoder = True


@dataclass
class LongLive2I2VConfig(LongLive2T2VConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

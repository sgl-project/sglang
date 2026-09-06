# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import field

from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25Config
from sglang.multimodal_gen.configs.models.encoders import (
    EncoderConfig,
    Gemma4UnifiedConfig,
)
from sglang.multimodal_gen.configs.models.vaes.ltx_2_5_video import LTX25VideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig

# Explicit sigma schedule the distilled LTX-2.5 DiT was trained against. Upstream
# exposes it as `diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES`.
LTX25_DISTILLED_SIGMA_VALUES: tuple[float, ...] = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
)


@dataclasses.dataclass
class LTX25PipelineConfig(LTX2PipelineConfig):
    """Pipeline configuration for LTX-2.5.

    LTX-2.5 reuses the LTX-2 pipeline class (`model_index.json` still declares
    `LTX2Pipeline`) and the LTX-2 *sigma* path -- upstream builds
    `np.linspace(1.0, 1/steps, steps)` and lets the scheduler's
    `use_dynamic_shifting: false` turn the shift into a no-op. So this must stay
    an `ltx_2` variant; do not mark it as an LTX-2.3 native variant.

    What differs from LTX-2 is the component geometry (DiT / VAE / connectors /
    text encoder), and that the shipped DiT is distilled, hence the pinned
    `default_sigmas`.
    """

    # One checkpoint drives both T2V and image-conditioned generation, so this
    # must stay TI2V -- T2V rejects `--image-path` outright.
    task_type: ModelTaskType = ModelTaskType.TI2V
    native_only_components: tuple[str, ...] = ("diffusion_decoder",)

    dit_config: LTX25Config = field(default_factory=LTX25Config)
    vae_config: LTX25VideoVAEConfig = field(default_factory=LTX25VideoVAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma4UnifiedConfig(),)
    )

    default_sigmas: tuple[float, ...] | None = field(
        default_factory=lambda: LTX25_DISTILLED_SIGMA_VALUES
    )

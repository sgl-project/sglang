from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config
from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms import current_platform


@dataclass
class HunyuanImage3PipelineConfig(SpatialImagePipelineConfig):
    """HunyuanImage-3 pipeline config."""

    vae_precision: str = "fp32"

    should_use_guidance: bool = True
    task_type: ModelTaskType = ModelTaskType.TI2I

    vae_tiling: bool = False
    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=HunyuanImage3DitConfig)

    vae_config: VAEConfig = field(default_factory=HunyuanImage3VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(T5ArchConfig(num_heads=6)),)
    )

    enable_autocast: bool = False

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()

    def supports_dynamic_batching(self):
        # The AR stage batches compatible requests in one diffusion loop
        # (run_grouped_requests), falling back to single-request execution.
        return True

    def supports_native_grouped_requests(self):
        return True

    def calculate_condition_image_size(self, image, width, height):
        """Let the native processor choose the conditional-image bucket.

        ``InputValidationStage`` normally resizes image-to-image inputs and
        snaps the output canvas to a generic ``2 * vae_scale`` grid.  That is
        not valid for HunyuanImage-3: its processor derives independent VAE
        and vision inputs from the original image and accepts a 16-pixel
        output grid.  Applying the generic resize first can therefore alter a
        requested or reference aspect ratio before the native processor sees
        it.
        """
        del image, width, height
        return None

    def prepare_calculated_size(self, image):
        """Keep HunyuanImage-3 output resolution under native AR-stage control."""
        del image
        return None

    def supports_batching_image_conditioning(self):
        # TI2I requests carry per-request conditioning (per-row masks/scatter/
        # RoPE); requests are bucketed by resolution and condition-image count.
        return True

    def supports_sequential_dit_inference(self):
        return True

    def supports_sequential_multi_output_inference(self):
        return current_platform.is_npu()

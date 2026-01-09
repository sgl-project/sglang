from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class GlmImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the GlmImage pipeline."""

    should_use_guidance: bool = False
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=GlmImageDitConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)

    enable_autocast: bool = False

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_cond,
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "prior_token_id": batch.prior_token_id,
            "prior_token_drop": batch.prior_token_drop_cond,
            "crop_coords": batch.crop_coords,
            "target_size": batch.target_size,
        }

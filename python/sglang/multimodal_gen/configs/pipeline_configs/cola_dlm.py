from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig
from sglang.multimodal_gen.configs.models.dits.cola_dlm import ColaDLMDiTConfig
from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig
from sglang.multimodal_gen.configs.models.vaes.cola_dlm import ColaDLVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class ColaDLMPipelineConfig(PipelineConfig):
    """Pipeline config for Cola-DLM text diffusion model.

    Cola-DLM uses a custom pipeline with fully custom stages
    (tokenization, block-wise denoising, text decoding).
    The standard DenoisingStage and DecodingStage are not used.
    """

    task_type: ModelTaskType = ModelTaskType.T2T
    should_use_guidance: bool = True
    dit_config: DiTConfig = field(default_factory=ColaDLMDiTConfig)
    vae_config: VAEConfig = field(default_factory=ColaDLVAEConfig)
    text_encoder_configs: tuple = ()
    text_encoder_precisions: tuple = ()
    preprocess_text_funcs: tuple = ()
    postprocess_text_funcs: tuple = ()

    # Cola-DLM specific config (loaded from model checkpoint)
    dit_path: str = "cola_dlm/cola_dit"
    vae_path: str = "cola_dlm/cola_vae"
    block_size: int = 16
    latent_dim: int = 16
    patch_size: int = 1
    vocab_size: int = 100278
    pad_token_id: int = 100277
    eos_token_id: int = 100257
    scaling_factor: float = 1.0
    shifting_factor: float = 0.0

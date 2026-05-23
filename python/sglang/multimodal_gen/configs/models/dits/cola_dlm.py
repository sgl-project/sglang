from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class ColaDLMArchConfig(DiTArchConfig):
    """Architecture config for Cola-DLM DiT (ColaDiTModel).

    Cola-DLM uses a text diffusion transformer operating in continuous latent space.
    The model uses NA (no-padding) form with txt_shape/txt_q_shape tensors.
    """

    block_size: int = 16
    latent_dim: int = 16
    patch_size: int = 1
    txt_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    # Identity mapping — weights are loaded directly from cola_dlm package
    param_names_mapping: dict = field(default_factory=lambda: {r"^(.*)$": r"\1"})

    def __post_init__(self):
        super().__post_init__()
        self.hidden_size = self.txt_dim
        self.num_attention_heads = self.num_heads
        self.num_channels_latents = self.latent_dim


@dataclass
class ColaDLMDiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=ColaDLMArchConfig)
    prefix: str = "ColaDLM"

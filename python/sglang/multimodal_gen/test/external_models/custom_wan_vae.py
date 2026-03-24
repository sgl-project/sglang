from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    AutoencoderKLWan as OriginalAutoencoderKLWan,
)


class AutoencoderKLWan(OriginalAutoencoderKLWan):
    """Custom WAN VAE model for testing external model replacement."""

    _is_custom = True

    def __init__(self, config) -> None:
        super().__init__(config)
        print(f"[CustomVAE] Initialized custom AutoencoderKLWan")


EntryClass = AutoencoderKLWan

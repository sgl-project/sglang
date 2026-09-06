# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class DiffusionDecoderLoader(PlainStateDictComponentLoader):
    """Standalone, replicated LTX-2.5 diffusion decoder."""

    component_names = ["diffusion_decoder"]
    config_classes = {"diffusion_decoder": LTX25DiffusionDecoderConfig}
    default_precision_attr = "vae_precision"

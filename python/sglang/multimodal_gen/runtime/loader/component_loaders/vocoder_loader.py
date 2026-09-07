import torch

from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import LTXVocoderConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class VocoderLoader(PlainStateDictComponentLoader):
    component_names = ["vocoder"]
    config_classes = {"vocoder": LTXVocoderConfig}
    default_precision_attr = "audio_vae_precision"
    default_dtype = torch.float32

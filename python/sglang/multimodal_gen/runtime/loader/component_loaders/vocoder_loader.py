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

    def checkpoint_key_mapping(self, model_config):
        return model_config.arch_config.param_names_mapping

    def place_model(self, model, device, dtype):
        # filter construction owns its precision independently of learned weights
        return model.to(device=device)

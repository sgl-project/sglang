# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import (
    Qwen3VLArchConfig,
    Qwen3VLConfig,
)


@dataclass
class Ideogram4TextEncoderConfig(Qwen3VLConfig):
    """Use the local Ideogram text_encoder as a language-only Qwen3-VL encoder."""

    def update_model_arch(self, source_model_dict):
        super().update_model_arch(source_model_dict)
        self.post_diffusers_config_update()

    def post_diffusers_config_update(self):
        self.arch_config.architectures = ["IdeogramQwen3VLTextEncoder"]
        quant_config = getattr(self.arch_config, "quantization_config", None)
        if isinstance(quant_config, dict):
            quant_method = quant_config.get("quant_method")
            load_in_4bit = quant_config.get("load_in_4bit", False)
        else:
            quant_method = getattr(quant_config, "quant_method", None)
            load_in_4bit = getattr(quant_config, "load_in_4bit", False)
        quant_method_name = str(quant_method).lower()
        use_bitsandbytes = "bitsandbytes" in quant_method_name and load_in_4bit
        self.arch_config.ideogram_bnb_4bit_weight_only = use_bitsandbytes
        self.arch_config.ideogram_fp8_weight_only = not use_bitsandbytes
        self.arch_config.requires_gpu_resident_text_encoder = use_bitsandbytes

    def finalize_model_arch(self):
        self.post_diffusers_config_update()

    arch_config: Qwen3VLArchConfig = field(default_factory=Qwen3VLArchConfig)

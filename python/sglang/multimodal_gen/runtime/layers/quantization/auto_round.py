# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig as SRTConfig


class AutoRoundConfig(QuantizationConfig):
    """Use SRT's serialized AutoRound kernels with diffusion linear layers."""

    checkpoint_uses_native_qkv_layout = True

    def __init__(self, srt_config: SRTConfig) -> None:
        super().__init__()
        self.srt_config = srt_config

    @classmethod
    def get_name(cls) -> str:
        return "auto-round"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return SRTConfig.get_supported_act_dtypes()

    @classmethod
    def get_min_capability(cls) -> int:
        return SRTConfig.get_min_capability()

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return SRTConfig.get_config_filenames()

    @classmethod
    def from_config(cls, config: dict) -> "AutoRoundConfig":
        srt_config = SRTConfig.from_config(config)
        if "gptq" not in srt_config.packing_format:
            raise ValueError(
                "SGLang diffusion currently supports AutoRound auto_gptq "
                f"checkpoints, but got {srt_config.packing_format!r}."
            )
        return cls(srt_config)

    def remap_checkpoint_prefixes(self, param_names_mapping: dict) -> None:
        mapping = get_param_names_mapping(param_names_mapping)
        remapped: dict[str, dict] = {}
        for prefix, layer_config in (self.srt_config.extra_config or {}).items():
            target, _, _ = mapping(f"{prefix}.weight")
            target = target.removesuffix(".weight")
            previous = remapped.setdefault(target, layer_config)
            if previous != layer_config:
                raise ValueError(
                    f"AutoRound fused module {target!r} has inconsistent shard configs."
                )

        self.srt_config.extra_config = remapped
        self.srt_config.block_name_to_quantize = None
        self.srt_config.packed_modules_mapping = self.packed_modules_mapping

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if not isinstance(layer, LinearBase):
            return None

        weight_bits, _, _ = self.srt_config.get_layer_config(layer, prefix)
        if not self.srt_config.check_quantized(weight_bits):
            return UnquantizedLinearMethod()

        return self.srt_config.apply_gptq_quant_layer(
            layer,
            prefix,
            self.srt_config.backend,
            additional_linear_types=(LinearBase,),
        )

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any, List

from sglang.multimodal_gen.configs.models.vocoder.base import (
    VocoderArchConfig,
    VocoderConfig,
)

# `LTX2VocoderWithBWE` stores both stacks with diffusers module names; SGLang
# follows ltx-core naming. The `vocoder.` / `bwe_generator.` prefixes match.
LTX_VOCODER_PARAM_NAMES_MAPPING: dict[str, str] = {
    r"^(vocoder|bwe_generator)\.conv_in\.(.*)$": r"\1.conv_pre.\2",
    r"^(vocoder|bwe_generator)\.conv_out\.(.*)$": r"\1.conv_post.\2",
    r"^(vocoder|bwe_generator)\.act_out\.(.*)$": r"\1.act_post.\2",
    r"^(vocoder|bwe_generator)\.upsamplers\.(.*)$": r"\1.ups.\2",
    r"^(vocoder|bwe_generator)\.resnets\.(.*)$": r"\1.resblocks.\2",
    # DownSample1d holds its kernel on a LowPassFilter1d submodule; UpSample1d
    # registers it directly. Must run after the renames above, so the rules are
    # evaluated in order rather than first-match-wins.
    r"^(vocoder|bwe_generator)\.(.*)downsample\.filter$": r"\1.\2downsample.lowpass.filter",
}


@dataclass
class LTXVocoderArchConfig(VocoderArchConfig):
    param_names_mapping: dict = field(
        default_factory=lambda: dict(LTX_VOCODER_PARAM_NAMES_MAPPING)
    )

    # Architecture params
    in_channels: int = 128
    hidden_channels: int = 1024
    out_channels: int = 2
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 15, 8, 4, 4])
    upsample_factors: List[int] = field(default_factory=lambda: [6, 5, 2, 2, 2])
    resnet_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resnet_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    leaky_relu_negative_slope: float = 0.1
    sample_rate: int = 24000

    # --- LTX-2.5 `LTX2VocoderWithBWE` fields -------------------------------
    # The base stack synthesises at `input_sampling_rate`, a mel STFT
    # re-analyses it, and the BWE stack resynthesises at `output_sampling_rate`.
    act_fn: str = "snake"
    final_act_fn: str | None = None
    final_bias: bool = True
    antialias: bool = False
    input_sampling_rate: int = 16000
    output_sampling_rate: int = 24000
    # Mel analysis feeding the BWE stack.
    filter_length: int = 512
    window_length: int = 512
    hop_length: int = 80
    num_mel_channels: int = 64
    # `bwe_upsample_factors` being non-empty is what marks a BWE checkpoint.
    bwe_act_fn: str = "snake"
    bwe_final_act_fn: str | None = None
    bwe_final_bias: bool = True
    bwe_hidden_channels: int = 512
    bwe_in_channels: int = 128
    bwe_out_channels: int = 2
    bwe_upsample_factors: List[int] = field(default_factory=list)
    bwe_upsample_kernel_sizes: List[int] = field(default_factory=list)
    bwe_resnet_kernel_sizes: List[int] = field(default_factory=list)
    bwe_resnet_dilations: List[List[int]] = field(default_factory=list)

    # `LTX2Vocoder` takes its BWE branch when this carries a "bwe" entry.
    vocoder: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.bwe_upsample_factors and self.vocoder is None:
            self.vocoder = self._build_nested_bwe_config()

    def _build_nested_bwe_config(self) -> dict[str, Any]:
        """Translate the flat diffusers fields into the nested ltx-core shape."""
        return {
            "vocoder": {
                "resblock": "AMP1",
                "activation": self.act_fn,
                "resblock_kernel_sizes": self.resnet_kernel_sizes,
                "resblock_dilation_sizes": self.resnet_dilations,
                "upsample_rates": self.upsample_factors,
                "upsample_kernel_sizes": self.upsample_kernel_sizes,
                "upsample_initial_channel": self.hidden_channels,
                "apply_final_activation": self.final_act_fn is not None,
                "use_tanh_at_final": self.final_act_fn == "tanh",
                "use_bias_at_final": self.final_bias,
            },
            "bwe": {
                "resblock": "AMP1",
                "activation": self.bwe_act_fn,
                "resblock_kernel_sizes": self.bwe_resnet_kernel_sizes,
                "resblock_dilation_sizes": self.bwe_resnet_dilations,
                "upsample_rates": self.bwe_upsample_factors,
                "upsample_kernel_sizes": self.bwe_upsample_kernel_sizes,
                "upsample_initial_channel": self.bwe_hidden_channels,
                "apply_final_activation": self.bwe_final_act_fn is not None,
                "use_tanh_at_final": self.bwe_final_act_fn == "tanh",
                "use_bias_at_final": self.bwe_final_bias,
                "input_sampling_rate": self.input_sampling_rate,
                "output_sampling_rate": self.output_sampling_rate,
                "n_fft": self.filter_length,
                "win_size": self.window_length,
                "hop_length": self.hop_length,
                "num_mels": self.num_mel_channels,
            },
        }


@dataclass
class LTXVocoderConfig(VocoderConfig):
    arch_config: LTXVocoderArchConfig = field(default_factory=LTXVocoderArchConfig)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/parakeet.py

from dataclasses import dataclass

from transformers import ParakeetEncoderConfig, PretrainedConfig


class ParakeetConfig(ParakeetEncoderConfig):
    def __init__(
        self,
        llm_hidden_size: int,
        projection_hidden_size: int,
        projection_bias: bool,
        sampling_rate: int,
        projection_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_hidden_size = llm_hidden_size
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate
        self.projection_eps = projection_eps

    @staticmethod
    def from_hf_config(
        config: PretrainedConfig, *, llm_hidden_size: int, max_model_len: int
    ) -> "ParakeetConfig":
        assert isinstance(config, PretrainedConfig)
        return ParakeetConfig(
            **config.to_dict(),
            scale_input=False,
            attention_bias=False,
            llm_hidden_size=llm_hidden_size,
            max_position_embeddings=max_model_len + 1,
        )


@dataclass(kw_only=True, frozen=True)
class ExtractorConfig:
    feature_size: int
    sampling_rate: int
    subsampling_factor: int
    subsampling_conv_kernel_size: int
    subsampling_conv_stride: int
    hop_length: int = 160
    clip_duration_s: int = 30
    clip_min_duration_s: float = 0.1

    @staticmethod
    def from_hf_config(config: PretrainedConfig) -> "ExtractorConfig":
        assert isinstance(config, PretrainedConfig)
        hop_length = int(getattr(config, "hop_length", ExtractorConfig.hop_length))
        return ExtractorConfig(
            feature_size=config.num_mel_bins,
            sampling_rate=config.sampling_rate,
            hop_length=hop_length,
            subsampling_factor=config.subsampling_factor,
            subsampling_conv_kernel_size=config.subsampling_conv_kernel_size,
            subsampling_conv_stride=config.subsampling_conv_stride,
        )

# coding=utf-8
# Copyright 2024 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
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
"""Mamba2 model configuration for SGLang."""

from transformers import Mamba2Config as HFMamba2Config

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape


class Mamba2Config(HFMamba2Config):
    """Config for pure Mamba-2 models such as Mamba-Codestral-7B.

    Subclasses the transformers Mamba2Config and adds the SSM hooks the Mamba2
    attention backend expects, following NemotronHConfig.
    """

    model_type = "mamba2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mamba2AttnBackend reads mamba_chunk_size; alias it to chunk_size.
        self.mamba_chunk_size = self.chunk_size

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return []

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        tp_world_size = parallel.tp_size if parallel else 1
        shape = Mamba2StateShape.create(
            tp_world_size=tp_world_size,
            intermediate_size=self.intermediate_size,
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.state_size,
            conv_kernel=self.conv_kernel,
        )
        return Mamba2CacheParams(
            shape=shape, layers=list(range(self.num_hidden_layers))
        )

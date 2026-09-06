# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Mamba (Mamba-1) model configuration for SGLang."""

from transformers import FalconMambaConfig as HFFalconMambaConfig
from transformers import MambaConfig as HFMambaConfig

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape

# Mamba-1 has no chunk size; the Mamba2 backend only reads mamba_chunk_size to
# bound the conv window, so a constant is enough.
_MAMBA1_CHUNK_SIZE = 256


def _mamba1_cache_params(config) -> Mamba2CacheParams:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    tp_world_size = parallel.tp_size if parallel else 1
    shape = Mamba2StateShape.create_full_rank(
        tp_world_size=tp_world_size,
        intermediate_size=config.intermediate_size,
        state_size=config.state_size,
        conv_kernel=config.conv_kernel,
    )
    return Mamba2CacheParams(shape=shape, layers=list(range(config.num_hidden_layers)))


class MambaConfig(HFMambaConfig):
    """Config for pure Mamba-1 models (state-spaces Mamba, -hf and raw).

    Subclasses the transformers MambaConfig and adds the same SSM hooks as
    Mamba2Config. Mamba-1 runs on the Mamba2 backend through a full-rank
    (head_dim == 1) state layout.
    """

    model_type = "mamba"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mamba_chunk_size = _MAMBA1_CHUNK_SIZE

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return []

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        return _mamba1_cache_params(self)


class FalconMambaConfig(HFFalconMambaConfig):
    """Config for Falcon-Mamba. Same as Mamba-1 aside from the B/C/dt RMS norm,
    which the model file handles; the config hooks are identical."""

    model_type = "falcon_mamba"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mamba_chunk_size = _MAMBA1_CHUNK_SIZE

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return []

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        return _mamba1_cache_params(self)

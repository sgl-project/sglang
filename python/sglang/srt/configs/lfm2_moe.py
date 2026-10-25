# Copyright 2025 SGLang Team
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
"""LFM2-MoE configuration with SGLang hybrid-cache properties."""

from typing import List, Optional

from transformers import CONFIG_MAPPING
from transformers import Lfm2MoeConfig as HFLfm2MoeConfig

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.runtime_context import get_parallel


class Lfm2MoeConfig(HFLfm2MoeConfig):
    """Extend Hugging Face's LFM2-MoE config with SGLang cache properties."""

    @property
    def full_attention_layer_ids(self) -> List[int]:
        """Return indices of attention layers for KV cache."""
        if self.layer_types is None:
            return []
        return [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]

    @property
    def linear_layer_ids(self) -> List[int]:
        """Return indices of conv layers for conv state cache."""
        if self.layer_types is None:
            return []
        return [
            i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")
        ]

    @property
    def mamba_chunk_size(self) -> int:
        """Return chunk size for Mamba2 backend. LFM2 doesn't use chunking."""
        return 1

    @property
    def mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:
        """
        Get cache params for HybridReqToTokenPool initialization.

        LFM2-MoE uses ShortConv layers with a small fixed-size cache.
        """

        conv_layer_ids = self.linear_layer_ids
        if not conv_layer_ids:
            return None

        hidden_size = self.hidden_size
        # conv_L_cache in config is kernel_size (e.g., 3)
        conv_kernel = int(self.conv_L_cache)
        # actual cache size is kernel_size - 1 (e.g., 2 for kernel=3)

        try:
            tp_size = get_parallel().attn_tp_size
        except (AssertionError, RuntimeError):
            tp_size = 1

        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=hidden_size,
            n_groups=1,
            num_heads=tp_size,  # Ensures divide works; temporal state is empty anyway
            head_dim=hidden_size,
            state_size=0,
            conv_kernel=conv_kernel,
        )

        # Uses default mamba2_state_dtype() which reads SGLANG_MAMBA_CONV_DTYPE env var
        # (defaults to bfloat16). Set SGLANG_MAMBA_CONV_DTYPE=float16 for fp16 inference.
        return Mamba2CacheParams(
            shape=shape,
            layers=conv_layer_ids,
        )


# Override the built-in config with SGLang's hybrid-cache properties.
CONFIG_MAPPING._extra_content["lfm2_moe"] = Lfm2MoeConfig

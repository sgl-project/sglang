# coding=utf-8
# Copyright 2024 Liquid AI and the HuggingFace Inc. team. All rights reserved.
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
"""LFM2 (Liquid Foundation Model 2) configuration"""

from typing import List, Optional

from transformers import CONFIG_MAPPING
from transformers import Lfm2Config as HFLfm2Config
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape

logger = logging.get_logger(__name__)


class Lfm2Config(HFLfm2Config):
    """
    SGLang configuration for LFM2 models.

    Extends HuggingFace's Lfm2Config with hybrid model properties needed by SGLang.
    LFM2 uses a hybrid architecture mixing full attention and ShortConv layers.
    """

    @property
    def full_attention_layer_ids(self) -> List[int]:
        """Return indices of attention layers for KV cache."""
        return [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]

    @property
    def linear_layer_ids(self) -> List[int]:
        """Return indices of conv layers for conv state cache."""
        return [
            i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")
        ]

    @property
    def mamba_chunk_size(self) -> int:
        """Return chunk size for Mamba2 backend. LFM2 doesn't use chunking, return 1."""
        return 1

    @property
    def mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:
        """
        Get cache params for HybridReqToTokenPool initialization.

        LFM2 uses ShortConv layers with a small fixed-size cache (kernel_size - 1).
        Unlike full Mamba2 models, LFM2 only uses the conv state, not SSM temporal state.
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        conv_layer_ids = self.linear_layer_ids
        if not conv_layer_ids:
            return None

        hidden_size = self.hidden_size
        # conv_L_cache in config is kernel_size (e.g., 3)
        conv_kernel = int(self.conv_L_cache)
        L_cache = conv_kernel - 1  # actual cache size (e.g., 2 for kernel=3)

        # get_attention_tp_size() requires initialization, default to 1 if not available
        try:
            tp_size = get_attention_tp_size()
        except (AssertionError, RuntimeError):
            tp_size = 1

        # For ShortConv layers, we use a simplified Mamba2StateShape
        # LFM2 doesn't use SSM state (state_size=0), only conv state
        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=hidden_size,
            n_groups=1,  # ShortConv doesn't use grouping
            num_heads=1,  # ShortConv is not multi-head
            head_dim=hidden_size,  # Conv operates on full hidden dim
            state_size=0,  # No SSM temporal state for ShortConv
            conv_kernel=conv_kernel,
        )

        # Uses default mamba2_state_dtype() which reads SGLANG_MAMBA_CONV_DTYPE env var
        # (defaults to bfloat16). Set SGLANG_MAMBA_CONV_DTYPE=float16 for fp16 inference.
        return Mamba2CacheParams(
            shape=shape,
            layers=conv_layer_ids,
        )


# Override HuggingFace's Lfm2Config with our extended version
# Cannot use .register() because lfm2 is already registered by transformers
# Directly modify the internal _extra_content dict instead
CONFIG_MAPPING._extra_content["lfm2"] = Lfm2Config

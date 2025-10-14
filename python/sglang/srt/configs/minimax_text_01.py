# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import MinimaxCacheParams
from sglang.srt.layers.dp_attention import get_attention_tp_size


class MiniMaxText01Config(PretrainedConfig):
    model_type = "minimax_text_01"
    model_type_m1 = "minimax_m1"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @property
    def linear_layer_ids(self):
        return [i for i, attn_type in enumerate(self.attn_type_list) if attn_type == 0]

    @property
    def full_attention_layer_ids(self):
        return [i for i, attn_type in enumerate(self.attn_type_list) if attn_type == 1]

    @property
    def state_shape(self):
        world_size = get_attention_tp_size()
        return (
            self.num_attention_heads // world_size,
            self.head_dim,
            self.head_dim,
        )

    @property
    def mamba2_cache_params(self):
        return MinimaxCacheParams(
            shape=self.state_shape, dtype=torch.float32, layers=self.linear_layer_ids
        )

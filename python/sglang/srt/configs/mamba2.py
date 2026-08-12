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
"""Mamba2 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

logger = logging.get_logger(__name__)


class Mamba2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Mamba2Model`].
    It is used to instantiate a Mamba2 model according to the specified arguments,
    defining the model architecture.

    Mamba2 is a state-space model (SSM) that doesn't use traditional attention mechanisms.
    Instead, it uses selective state spaces for efficient sequence modeling.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Mamba2 model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the weights of the input embeddings and the output embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        ssm_cfg (`dict`, *optional*):
            Configuration for the SSM (State Space Model) components.
        mamba_ssm_dtype (`str`, *optional*, defaults to "float32"):
            The dtype to use for Mamba SSM state tensors.
    """

    model_type = "mamba2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        rms_norm_eps=1e-5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=True,
        ssm_cfg=None,
        mamba_ssm_dtype="float32",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {}
        self.mamba_ssm_dtype = mamba_ssm_dtype

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_mamba2_cache_params(self):
        """Get Mamba2 cache parameters for this config."""
        return Mamba2CacheParams(self)

    def get_mamba2_state_shape(self, layer_id: int):
        """Get Mamba2 state shape for a specific layer."""
        return Mamba2StateShape(self, layer_id)

    def get_mamba2_state_dtype(self):
        """Get Mamba2 state dtypes."""
        return mamba2_state_dtype(self)

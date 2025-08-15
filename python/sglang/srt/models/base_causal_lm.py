# Copyright 2023-2024 SGLang Team
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

"""Base class for causal language models to avoid code duplication."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class BaseCausalLM(nn.Module, ABC):
    """Base class for causal language models with common functionality."""

    # Default BitandBytes target modules - can be overridden in subclasses
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]

    # Default stacked parameters mapping - can be overridden in subclasses
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    @abstractmethod
    def _init_model(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """Initialize the underlying model. Must be implemented by subclasses."""
        pass

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embedding and head weights for model operations like EAGLE3."""
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Set embedding and head weights for model operations like EAGLE3."""
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self) -> torch.Tensor:
        """Get embedding weights."""
        return self.model.embed_tokens.weight

    def set_embed(self, embed: torch.Tensor) -> None:
        """Set embedding weights."""
        # Handle special cases for models with different hidden sizes
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """Load KV cache scales for quantization."""
        if hasattr(self.model, "load_kv_cache_scales"):
            self.model.load_kv_cache_scales(quantization_param_path)
        else:
            logger.warning(
                f"Model {self.__class__.__name__} does not support KV cache scales"
            )

    def set_eagle3_layers_to_capture(
        self, layer_ids: Optional[List[int]] = None
    ) -> None:
        """Set layers to capture for EAGLE3 support."""
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Default layers for EAGLE3 support
        else:
            # Plus 1 because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        """Split prefill forward pass implementation."""
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        """Get the start layer index for pipeline parallelism."""
        return self.model.start_layer

    @property
    def end_layer(self):
        """Get the end layer index for pipeline parallelism."""
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings layer."""
        return self.model.embed_tokens

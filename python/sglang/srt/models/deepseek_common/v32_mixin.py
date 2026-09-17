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

"""DeepSeek V3.2/DSA model helpers shared by the target and NextN models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    dsa_layer_skips_topk,
    get_dsa_index_head_dim,
    get_dsa_index_kpool,
    get_dsa_index_n_heads,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.communicator import LayerCommunicator, get_attn_tp_context
from sglang.srt.layers.communicator_dsa_cp import (
    DSACPLayerCommunicator,
    maybe_prefetch_next_full_attention_kv,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel


class DeepseekV32Mixin:
    """Encapsulate V3.2/DSA setup without owning the model's forward methods."""

    def init_v32_attention(
        self,
        *,
        config: PretrainedConfig,
        hidden_size: int,
        qk_rope_head_dim: int,
        q_lora_rank: Optional[int],
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[Dict[str, Any]],
        quant_config: Optional[QuantizationConfig],
        layer_id: int,
        alt_stream: Optional[torch.cuda.Stream],
        prefix: str,
        is_nextn: bool,
        skip_rope: bool = False,
    ) -> None:
        self.use_dsa = is_deepseek_dsa(config)
        self.skip_topk = None
        self.next_skip_topk = None
        self.indexer = None
        if not self.use_dsa:
            return

        # Refer to https://arxiv.org/abs/2603.12201 for the cross-layer
        # index reuse policy used by DSA and the NextN layer.
        if is_nextn:
            self.skip_topk = True
            self.next_skip_topk = True
        else:
            self.skip_topk = dsa_layer_skips_topk(config, layer_id)
            self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

        if not self.skip_topk or is_nextn:
            indexer_cls = IndexerKPool if get_dsa_index_kpool(config) > 1 else Indexer
            indexer_kwargs = dict(
                hidden_size=hidden_size,
                index_n_heads=get_dsa_index_n_heads(config),
                index_head_dim=get_dsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_dsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
                prefix=prefix,
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
                config=config,
            )
            if indexer_cls is IndexerKPool:
                indexer_kwargs["skip_rope"] = skip_rope
            self.indexer = indexer_cls(**indexer_kwargs)

    def create_layer_communicator(
        self,
        *,
        layer_scatter_modes,
        input_layernorm,
        post_attention_layernorm,
        is_last_layer: bool,
        qkv_latent_func,
    ):
        communicator_cls = (
            DSACPLayerCommunicator
            if get_parallel().enable_prefill_cp
            else LayerCommunicator
        )
        return communicator_cls(
            layer_scatter_modes=layer_scatter_modes,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=is_last_layer,
            qkv_latent_func=qkv_latent_func,
        )

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )

    def dsa_forward_uses_topk(self) -> bool:
        if not self.use_dsa:
            return False
        backend = get_attn_backend()
        backend = getattr(backend, "primary", backend)
        return not getattr(backend, "use_mha", False)

    def dsa_layer_skips_topk(self, layer_id: int) -> bool:
        return dsa_layer_skips_topk(self.config, layer_id)

    def empty_dsa_topk_indices(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.new_empty(
            (0, get_dsa_index_topk(self.config)), dtype=torch.int32
        )

    def init_v32_attn_tp_context(self, config: PretrainedConfig) -> None:
        q_lora_rank = getattr(config, "q_lora_rank", None)
        get_attn_tp_context().init_context(q_lora_rank, self.use_dsa)

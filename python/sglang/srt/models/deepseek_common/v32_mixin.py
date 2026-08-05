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
    get_dsa_index_n_heads,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_cp_split,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.communicator import LayerCommunicator, get_attn_tp_context
from sglang.srt.layers.communicator_dsa_cp import (
    DSACPLayerCommunicator,
    maybe_prefetch_next_full_attention_kv,
)
from sglang.srt.layers.cp.utils import is_cp_v2_active
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils.cp_utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    mla_use_prefill_cp,
    prepare_context_parallel_metadata,
)
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
        dsa_enable_prefill_cp: bool,
        mla_enable_prefill_cp: bool,
    ) -> None:
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        if self.dsa_enable_prefill_cp:
            assert self.use_dsa, "CP currently only supports deepseek v3.2 model"

        # Both CP flavors reuse the attention TP group and duplicate weights.
        self.cp_size = (
            get_parallel().attn_cp_size
            if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp
            else None
        )

        self.skip_topk = None
        self.next_skip_topk = None
        if not self.use_dsa:
            return

        self.indexer = Indexer(
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

        # Refer to https://arxiv.org/abs/2603.12201 for the cross-layer
        # index reuse policy used by DSA and the NextN layer.
        if is_nextn:
            self.skip_topk = True
            self.next_skip_topk = True
        else:
            index_cli_factor = getattr(config, "cli_factor", 1)
            if index_cli_factor > 1:
                self.skip_topk = layer_id % index_cli_factor != 0
                self.next_skip_topk = (layer_id + 1) % index_cli_factor != 0
            else:
                self.skip_topk = dsa_layer_skips_topk(config, layer_id)
                self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

    def create_layer_communicator(
        self,
        *,
        layer_scatter_modes,
        input_layernorm,
        post_attention_layernorm,
        is_last_layer: bool,
        qkv_latent_func,
    ):
        # DSACPLayerCommunicator is flavor-agnostic: its internal gates handle
        # both DSA and dense MLA prefill CP.
        communicator_cls = (
            DSACPLayerCommunicator
            if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp
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

    def init_v32_model_cp(
        self,
        config: PretrainedConfig,
        *,
        mla_enable_prefill_cp: bool,
    ) -> None:
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = mla_enable_prefill_cp and not self.use_dsa
        self.cp_size = (
            get_parallel().attn_cp_size
            if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp
            else None
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

    def use_prefill_cp_v1(self, forward_batch: ForwardBatch) -> bool:
        return (
            dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
            or mla_use_prefill_cp(forward_batch, self.mla_enable_prefill_cp)
        ) and not is_cp_v2_active(forward_batch)

    def maybe_split_model_inputs_for_cp(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        is_first_pp_rank: bool,
        use_cp_v1: bool,
    ):
        if use_cp_v1:
            if is_first_pp_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        return hidden_states, positions

    def maybe_gather_model_outputs_for_cp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        use_cp_v1: bool,
    ) -> torch.Tensor:
        if not use_cp_v1:
            return hidden_states
        return cp_all_gather_rerange_output(
            hidden_states,
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )

    def init_v32_for_causal_lm(
        self,
        config: PretrainedConfig,
        *,
        mla_enable_prefill_cp: bool,
    ) -> None:
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.mla_enable_prefill_cp = mla_enable_prefill_cp and not self.use_dsa
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size
        else:
            self.cp_rank = None
            self.cp_size = None

    def init_v32_attn_tp_context(self, config: PretrainedConfig) -> None:
        q_lora_rank = getattr(config, "q_lora_rank", None)
        get_attn_tp_context().init_context(q_lora_rank, self.use_dsa)

    def maybe_prepare_cp_metadata(
        self,
        input_length: int,
        forward_batch: ForwardBatch,
    ) -> None:
        if is_cp_v2_active(forward_batch):
            return

        if self.dsa_enable_prefill_cp:
            can_split = can_dsa_cp_split(
                input_length, self.cp_size, self.use_dsa, forward_batch
            )
        elif self.mla_enable_prefill_cp:
            can_split = can_cp_split(input_length, self.cp_size, forward_batch)
        else:
            return

        if can_split:
            forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                input_length,
                self.cp_rank,
                self.cp_size,
                forward_batch.seq_lens_cpu.tolist(),
                extend_seqs_len=forward_batch.extend_seq_lens_cpu,
            )

    @staticmethod
    def gather_dsa_topk_indices_for_cp(
        topk_indices: torch.Tensor,
        local_num_tokens: int,
        cp_size: int,
        forward_batch: ForwardBatch,
        stream,
    ) -> torch.Tensor:
        if (
            is_dsa_prefill_cp_round_robin_split()
            and topk_indices.shape[0] < local_num_tokens
        ):
            pad_rows = local_num_tokens - topk_indices.shape[0]
            topk_indices = torch.cat(
                [
                    topk_indices,
                    topk_indices.new_full((pad_rows, topk_indices.shape[1]), -1),
                ],
                dim=0,
            )
        return cp_all_gather_rerange_output(
            topk_indices,
            cp_size,
            forward_batch,
            stream,
        )

# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
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
# ==============================================================================
"""The MTP (NextN) draft of ERNIE 5.5 Flash, served by EAGLE.

The checkpoint stores it as the decoder layer after the main stack: an MLA layer
with a DSA indexer and a MoE, fed ``e_proj(enorm(embed)) + h_proj(hnorm(h))``
per mHC stream, where ``h`` is the target's mHC streams before its head.
:func:`sglang.srt.configs.ernie55.to_nextn_draft_config` reshapes the target
config into this one.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.configs.ernie55 import Ernie5_5_MoeConfig
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm, exact_rms_norm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.ernie55 import (
    Ernie5_5DecoderLayer,
    Ernie5_5HcHead,
    Ernie5_5LogitsProcessor,
    Ernie5_5WeightLoader,
    check_supported_deployment,
    logits_rows,
    scatter_rows,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator, add_prefix


class Ernie5_5ModelNextN(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        # Replaced by the target's in set_embed_and_head.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("e_proj", prefix),
        )
        self.h_proj = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("h_proj", prefix),
        )
        self.decoder = Ernie5_5DecoderLayer(
            config, 0, quant_config, add_prefix("decoder", prefix)
        )
        self.hc_head = Ernie5_5HcHead(config)
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hc_mult, hidden_size = self.config.hc_mult, self.config.hidden_size
        x = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        num_tokens = x.shape[0]
        if num_tokens > 0:
            streams = forward_batch.spec_info.hidden_states.view(
                num_tokens * hc_mult, hidden_size
            )
            h, _ = self.h_proj(exact_rms_norm(streams, self.hnorm))
            e, _ = self.e_proj(exact_rms_norm(x, self.enorm))
            x = e.unsqueeze(1) + h.view(num_tokens, hc_mult, hidden_size)
        else:
            x = x.new_empty((0, hc_mult, hidden_size))
        zero_allocator = BumpAllocator(
            buffer_size=2, dtype=torch.float32, device=x.device
        )
        # Only the LM head's rows (one per request) go past the indexer; KV and
        # index-K still get every row.
        rows = None
        if (
            num_tokens > 0
            and not forward_batch.capture_hidden_mode.is_full()
            and self.decoder.ffn_reads_own_rows()
        ):
            rows = logits_rows(forward_batch)
        # The draft layer is not one of the target's layers the recorder counts.
        with (
            get_global_expert_distribution_recorder().disable_this_region(),
            get_attn_backend().read_rows(rows, num_tokens),
        ):
            x = self.decoder(positions, x, forward_batch, zero_allocator, out_rows=rows)
        pre_hc_head = x.flatten(1)
        if num_tokens == 0:
            return x.new_empty((0, hidden_size)), pre_hc_head
        x = exact_rms_norm(self.hc_head(x), self.shared_head.norm)
        if rows is not None:
            # The logits processor picks these rows out of full-batch tensors.
            x = scatter_rows(x, rows, num_tokens)
            pre_hc_head = scatter_rows(pre_hc_head, rows, num_tokens)
        return x, pre_hc_head


class Ernie5_5_MoeForCausalLMNextN(nn.Module):
    def __init__(
        self,
        config: Ernie5_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        config.validate_checkpoint_fields()
        check_supported_deployment(config)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.start_layer, self.end_layer = 0, 1
        self.model = Ernie5_5ModelNextN(
            config, quant_config, add_prefix("model", prefix)
        )
        # Replaced by the target's in set_embed_and_head.
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = Ernie5_5LogitsProcessor(config)
        get_attn_tp_context().init_context(config.hybrid_mla_q_lora_rank, is_dsa=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states, pre_hc_head = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=pre_hc_head,
        )

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        _Ernie5_5NextNWeightLoader(self).load_all(weights)
        self.post_load_weights()

    def post_load_weights(self) -> None:
        # Splits kv_b_proj of the one MLA layer, model.decoder.
        DeepseekV2WeightLoaderMixin.post_load_weights(self, is_nextn=True)
        self.model.decoder.self_attn.postmix.fold()


class _Ernie5_5NextNWeightLoader(Ernie5_5WeightLoader):
    # Checkpoint tensors that sit outside the decoder.
    _OUTER = {
        "e_proj.weight": "model.e_proj.weight",
        "e_proj.weight_scale_inv": "model.e_proj.weight_scale_inv",
        "h_proj.weight": "model.h_proj.weight",
        "h_proj.weight_scale_inv": "model.h_proj.weight_scale_inv",
        "enorm.weight": "model.enorm.weight",
        "hnorm.weight": "model.hnorm.weight",
        "hc_head_fn": "model.hc_head.fn",
        "hc_head_base": "model.hc_head.base",
        "hc_head_scale": "model.hc_head.scale",
        "shared_head.norm.weight": "model.shared_head.norm.weight",
    }
    # Shared from the target after loading, so the checkpoint never fills them.
    _BOUND_LATER = ("model.embed_tokens.weight", "lm_head.weight")

    def __init__(self, model: Ernie5_5_MoeForCausalLMNextN):
        self._init_tables(
            model,
            layers=[model.model.decoder],
            mla_layers=(0,),
            preloaded=self._BOUND_LATER,
        )
        self.mtp_prefix = f"model.layers.{model.config.mtp_checkpoint_layer_id}."

    def load(self, name: str, tensor: torch.Tensor) -> None:
        # The draft logits apply the target's multimax too.
        if name in self._MULTIMAX_RENAMES:
            self._load(self._MULTIMAX_RENAMES[name], tensor)
            return
        if not name.startswith(self.mtp_prefix):
            return
        suffix = name[len(self.mtp_prefix) :]
        if suffix in self._OUTER:
            self._load(self._OUTER[suffix], tensor)
        else:
            # The mHC parameters sit under an extra transformer_layer. prefix.
            suffix = suffix.removeprefix("transformer_layer.")
            self._load_layer_tensor("model.decoder.", 0, suffix, tensor)


EntryClass = [Ernie5_5_MoeForCausalLMNextN]

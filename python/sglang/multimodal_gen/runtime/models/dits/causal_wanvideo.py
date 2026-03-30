# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.models.dits.causal_wan_common import (
    BaseCausalWanSelfAttention,
    BaseCausalWanTransformer3DModel,
    BaseCausalWanTransformerBlock,
)


class CausalWanSelfAttention(BaseCausalWanSelfAttention):

    def _should_use_flex_attention(self, block_mask, kv_cache) -> bool:
        return kv_cache is None

    def _incremental_attention(
        self,
        roped_query: torch.Tensor,
        roped_key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        cache_start: int,
    ) -> torch.Tensor:
        frame_seqlen = roped_query.shape[1]
        current_end = current_start + roped_query.shape[1]
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]
        num_new_tokens = roped_query.shape[1]
        if (
            self.local_attn_size != -1
            and (current_end > kv_cache["global_end_index"].item())
            and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
        ):
            num_evicted_tokens = (
                num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            )
            num_rolled_tokens = (
                kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            )
            kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[
                "k"
            ][
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache[
                "v"
            ][
                :,
                sink_tokens
                + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
                - num_evicted_tokens
            )
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = value
        else:
            local_end_index = (
                kv_cache["local_end_index"].item()
                + current_end
                - kv_cache["global_end_index"].item()
            )
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"] = kv_cache["k"].detach()
            kv_cache["v"] = kv_cache["v"].detach()
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = value

        x = self.attn(
            roped_query,
            kv_cache["k"][
                :, max(0, local_end_index - self.max_attention_size) : local_end_index
            ],
            kv_cache["v"][
                :, max(0, local_end_index - self.max_attention_size) : local_end_index
            ],
        )
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return x


class CausalWanTransformerBlock(BaseCausalWanTransformerBlock):
    self_attn_cls = CausalWanSelfAttention


class CausalWanTransformer3DModel(BaseCausalWanTransformer3DModel):
    block_cls = CausalWanTransformerBlock


EntryClass = CausalWanTransformer3DModel

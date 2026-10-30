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
"""FlashInfer backend that routes all-beam decode batches to shared-context
attention; every other batch runs the parent unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.beam_search import shared_context_attention
from sglang.srt.layers.attention.flashinfer_backend import (
    DecodeMetadata,
    FlashInferAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BeamSharedContextFlashInferAttnBackend(FlashInferAttnBackend):
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.beam_attention_metadata is None:
            return super().init_forward_metadata(forward_batch)
        self.forward_metadata = DecodeMetadata([])

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if forward_batch.beam_attention_metadata is None or layer.is_cross_attention:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc.for_layer(
                        forward_batch,
                        layer,
                        swa_loc=self.forward_metadata.swa_out_cache_loc,
                    ),
                    k,
                    v,
                    *self._kv_write_scales(layer),
                )

        return shared_context_attention.forward(
            q, layer, forward_batch, self.token_to_kv_pool
        )

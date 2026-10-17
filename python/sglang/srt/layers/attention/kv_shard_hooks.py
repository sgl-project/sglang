# Copyright 2023-2026 SGLang Team
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

"""Logical-page KV-shard helpers shared by attention backends.

Page-interleaved pools build one gather plan per extend batch. Keeping pool
detection and that begin/end lifecycle here prevents each compatible attention
backend from implementing a subtly different version of the contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveKVPoolMixin

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def get_kv_shard_pool(token_to_kv_pool) -> Optional[PageInterleaveKVPoolMixin]:
    """Return the page-interleaved pool, if logical-page sharding is active."""
    return (
        token_to_kv_pool
        if isinstance(token_to_kv_pool, PageInterleaveKVPoolMixin)
        else None
    )


def prepare_kv_shard_forward(
    pool: PageInterleaveKVPoolMixin,
    req_to_token: torch.Tensor,
    forward_batch: ForwardBatch,
) -> bool:
    """Update the pool's gather plan and report whether this is an extend."""
    if not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
        pool.end_shard_extend()
        return False

    req_pool_indices = forward_batch.req_pool_indices
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    seq_lens = forward_batch.seq_lens_cpu
    if req_pool_indices is None or prefix_lens is None or seq_lens is None:
        raise RuntimeError(
            "KV-sharded attention requires request indices and CPU length metadata"
        )

    pool.begin_shard_extend(
        req_to_token,
        req_pool_indices,
        prefix_lens,
        seq_lens,
    )
    return True

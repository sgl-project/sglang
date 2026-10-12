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

"""Per-forward metadata for decode context parallel (DCP)."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DCPKVCacheGatherPlan:
    """Layer-independent send indices and receive layout for one prefix chunk."""

    kv_indices: torch.Tensor
    unpack_metadata: torch.Tensor | tuple[int, int, int, int]
    requests: list[tuple[int, int, int, int]]
    num_tokens: int
    max_prefix_len: int

    @classmethod
    def build(
        cls,
        kv_indices: torch.Tensor,
        lengths: list[int] | torch.Tensor,
        starts: Optional[list[int] | torch.Tensor] = None,
        extend_lens: Optional[list[int]] = None,
    ):
        """Build once per batch/chunk; lengths and starts must be on the CPU."""
        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        world, rank = parallel.dcp_size, parallel.dcp_rank
        lengths = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
        if isinstance(starts, torch.Tensor):
            starts = starts.tolist()
        starts = starts if starts is not None else [0] * len(lengths)
        requests, copies = [], []
        padded_start = local_start = output_start = 0
        for i, (start, length) in enumerate(zip(starts, lengths)):
            start %= world
            padded_len = (start + length + world - 1) // world
            left_pad = int(start > rank)
            local_len = padded_len - left_pad - int((start + length - 1) % world < rank)
            requests.append((padded_start, start, length, output_start))
            copies.append((padded_start + left_pad, local_start, local_len))
            padded_start += padded_len
            local_start += local_len
            output_start += length + (extend_lens[i] if extend_lens is not None else 0)

        # Padding reads the reserved pool row; unpack discards these values.
        # Aligned chunks reuse their existing indices.
        if padded_start != kv_indices.numel():
            padded_indices = kv_indices.new_zeros((padded_start,))
            for dst, src, count in copies:
                if count:
                    padded_indices[dst : dst + count].copy_(
                        kv_indices[src : src + count]
                    )
            kv_indices = padded_indices
        metadata = (
            requests[0]
            if len(requests) == 1
            else torch.tensor(requests, dtype=torch.int64, device=kv_indices.device)
        )
        return cls(
            kv_indices, metadata, requests, sum(lengths), max(lengths, default=0)
        )


# NOTE: This is intentionally a standalone dataclass, NOT a subclass of
# layers.cp.base.BaseContextParallelMetadata. It is preserved verbatim from #14194
# and is stored in its own ForwardBatch field (attn_dcp_metadata), separate from the
# prefill-CP attn_cp_metadata, so it never participates in the CP build_metadata
# contract today. Whether the decode metadata should re-parent onto
# BaseContextParallelMetadata is deferred to P2 (DecodeContextParallelStrategy); decide
# it there rather than coupling this relocation to the CP ABC.
@dataclass
class DecodeContextParallelMetadata:
    # For decode context parallel
    dcp_kv_indptr: Optional[torch.Tensor] = None
    dcp_kv_buffer: Optional[torch.Tensor] = None
    dcp_kv_indices: Optional[torch.Tensor] = None
    dcp_local_prefix_kv_indices: Optional[torch.Tensor] = None
    dcp_extend_prefix_lens_sum: Optional[int] = None
    mla_gather_plan: Optional[DCPKVCacheGatherPlan] = None
    mha_gather_plan: Optional[DCPKVCacheGatherPlan] = None

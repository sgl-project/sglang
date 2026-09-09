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
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch


# NOTE: This is intentionally a standalone dataclass, NOT a subclass of
# layers.cp.base.BaseContextParallelMetadata. It is preserved verbatim from #14194
# and is stored in its own ForwardBatch field (attn_dcp_metadata), separate from the
# prefill-CP attn_cp_metadata, so it never participates in the CP-v2 build_metadata
# contract today. Whether the decode metadata should re-parent onto
# BaseContextParallelMetadata is deferred to P2 (DecodeContextParallelStrategy); decide
# it there rather than coupling this relocation to the CP-v2 ABC.
@dataclass
class DecodeContextParallelMetadata:
    # For decode context parallel
    dcp_kv_indptr: Optional[torch.Tensor] = None
    dcp_kv_buffer: Optional[torch.Tensor] = None
    dcp_kv_indices: Optional[torch.Tensor] = None
    dcp_local_prefix_kv_indices: Optional[torch.Tensor] = None
    dcp_extend_prefix_lens_sum: Optional[int] = None


@runtime_checkable
class SupportsDecodeContextParallelMetadata(Protocol):
    def prepare_context_parallel_metadata_for_dcp(
        self,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        seq_lens_sum: int,
        kv_buffer_shape: torch.Size,
        kv_cache_dtype: Any,
        kv_cache_device: Any,
        create_chunked_prefix_cache_kv_indices_fn: Callable[..., Any],
    ) -> Optional[DecodeContextParallelMetadata]: ...

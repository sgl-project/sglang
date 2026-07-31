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

"""Direct-to-final Query publication for decode context parallel MLA.

Each DCP rank computes only its local Query heads and publishes the strided
``q_nope`` and ``q_rope`` inputs through an NVLS multicast mapping directly
into their offsets in the final complete Query buffer on every consumer. The
attention kernels keep their existing local-input contract.

The underlying multimem kernel has an entry barrier before it overwrites the
shared buffer and an acquire-release exit barrier after publication. A single
workspace can therefore be reused by sequential MLA layers: a rank cannot enter
the next publication until its local attention consumer has finished, and the
entry barrier prevents any producer from overwriting the previous Query until
all ranks have reached that point.
"""

from __future__ import annotations

import torch
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherState,
    all_gather_split_inner,
    create_state,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator


class DCPDirectFinalQueryGatherer:
    """Persistent DCP Query workspace backed by symmetric NVLS storage."""

    def __init__(
        self,
        *,
        group: GroupCoordinator,
        max_tokens: int,
        local_heads: int,
        nope_dim: int,
        rope_dim: int,
        device: torch.device,
    ) -> None:
        if group.world_size <= 1:
            raise ValueError("Direct-final DCP Query requires at least two ranks.")
        if max_tokens <= 0:
            raise ValueError(
                f"Direct-final DCP Query max_tokens must be positive, got {max_tokens}."
            )
        if local_heads <= 0 or nope_dim <= 0 or rope_dim <= 0:
            raise ValueError(
                "Direct-final DCP Query dimensions must be positive, got "
                f"local_heads={local_heads}, nope_dim={nope_dim}, rope_dim={rope_dim}."
            )

        self.group = group
        self.max_tokens = int(max_tokens)
        self.local_heads = int(local_heads)
        self.global_heads = self.local_heads * group.world_size
        self.nope_dim = int(nope_dim)
        self.rope_dim = int(rope_dim)
        self.head_dim = self.nope_dim + self.rope_dim
        self.local_hidden = self.local_heads * self.head_dim
        self.global_hidden = self.global_heads * self.head_dim

        if self.local_hidden % 8 != 0:
            raise ValueError(
                "Direct-final DCP Query requires each BF16 shard row to contain "
                f"a multiple of eight elements, got {self.local_hidden}."
            )

        self.state: MultimemAllGatherState = create_state(
            group=group.device_group,
            rank_in_group=group.rank_in_group,
            max_tokens=self.max_tokens,
            hidden_size=self.global_hidden,
            device=device,
        )
        if self.state.symm_mem_hdl.multicast_ptr == 0:
            raise RuntimeError(
                "Direct-final DCP Query requires an NVLS multicast mapping for "
                f"world_size={group.world_size}."
            )

    def __call__(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(q_nope, q_rope)

        final_query = all_gather_split_inner(self.state, q_nope, q_rope)
        final_nope, final_rope = final_query.split(
            (self.nope_dim, self.rope_dim), dim=-1
        )
        return final_nope, final_rope

    def _validate_inputs(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
    ) -> None:
        expected_nope = (self.local_heads, self.nope_dim)
        expected_rope = (self.local_heads, self.rope_dim)
        if q_nope.ndim != 3 or tuple(q_nope.shape[1:]) != expected_nope:
            raise ValueError(
                "Direct-final DCP Query expected q_nope shape "
                f"[T,{self.local_heads},{self.nope_dim}], got {tuple(q_nope.shape)}."
            )
        if q_rope.ndim != 3 or tuple(q_rope.shape[1:]) != expected_rope:
            raise ValueError(
                "Direct-final DCP Query expected q_rope shape "
                f"[T,{self.local_heads},{self.rope_dim}], got {tuple(q_rope.shape)}."
            )
        if q_nope.shape[0] != q_rope.shape[0]:
            raise ValueError(
                "Direct-final DCP Query requires matching token counts, got "
                f"{q_nope.shape[0]} and {q_rope.shape[0]}."
            )
        if not 0 < q_nope.shape[0] <= self.max_tokens:
            raise ValueError(
                "Direct-final DCP Query token count exceeds the persistent "
                f"workspace: {q_nope.shape[0]} > {self.max_tokens}."
            )
        if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
            raise TypeError(
                "Direct-final DCP Query currently requires BF16 inputs, got "
                f"{q_nope.dtype} and {q_rope.dtype}."
            )
        if not q_nope.is_cuda or not q_rope.is_cuda:
            raise ValueError("Direct-final DCP Query inputs must be CUDA tensors.")
        if q_nope.device != q_rope.device or q_nope.device != self.state.device:
            raise ValueError(
                "Direct-final DCP Query inputs and workspace must share a device, "
                f"got {q_nope.device}, {q_rope.device}, and {self.state.device}."
            )

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

"""The decode-context-parallel (DCP) attention communication group.

``DcpAttnComm`` owns the three facts that every DCP call site used to re-derive
for itself:

1. **Head arithmetic.** Attention weights are sharded over ``attn_tp_size``
   ranks, so ``o_proj`` consumes ``num_heads // attn_tp_size`` heads. Under DCP
   each rank holds only a slice of the KV sequence, so it must evaluate every
   head its DCP partners evaluate: the kernel runs on ``dcp_size x`` that many
   heads. ``num_kernel_heads`` is the widened count.

2. **Head-shard mapping.** The post-attention merge hands each rank one chunk of
   the widened head set, and ``o_proj`` requires that chunk to be the flat-TP
   head shard ``attn_tp_rank`` owns. ``head_shard_index`` names the chunk and
   ``check_layout`` asserts the rank-layout identity that makes it the right
   one.

3. **Comm-backend dispatch.** ``ag_rs`` / ``a2a`` / ``fi_a2a``, plus the LSE log
   base each attention backend reports in, resolved here instead of separately
   at each merge site.

Topology is read through ``get_parallel()`` on each access rather than captured
at construction, so ``ParallelContext.override()`` still works in tests and the
accessor can be created before the DCP process group exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dcp.comm import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mha,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
    init_fi_a2a_workspace,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# FlashMLA reports natural-log softmax LSE; FlashInfer-MLA and the other MLA DCP
# decode backends report base-2. The merge must use the matching exp/log pair --
# a mismatch computes wrong cross-rank softmax weights silently rather than
# raising, so this mapping is the only place the convention is recorded.
_NATURAL_LOG_LSE_BACKENDS = frozenset({"flashmla"})

# The MHA/GQA decode path derives its LSE with torch.logsumexp (natural log),
# independent of which kernel produced the attention output.
_MHA_LSE_IS_BASE_ON_E = True


def is_lse_base_on_e(attention_backend: Optional[str]) -> bool:
    """Whether ``attention_backend`` reports natural-log (rather than base-2) LSE."""
    return attention_backend in _NATURAL_LOG_LSE_BACKENDS


class DcpAttnComm:
    """Read-through accessor for the DCP attention communication group."""

    __slots__ = ()

    # ---------------------------------------------------------------- topology

    @property
    def enabled(self) -> bool:
        return get_parallel().dcp_enabled

    @property
    def size(self) -> int:
        """DCP degree, or 1 when DCP is disabled (never raises)."""
        return get_parallel().attn_dcp_size

    @property
    def rank(self) -> int:
        """This rank's position in the DCP group, or 0 when DCP is disabled."""
        return get_parallel().attn_dcp_rank

    @property
    def group(self) -> GroupCoordinator:
        """The DCP process group. Only valid while ``enabled``."""
        return get_parallel().dcp_group

    @property
    def comm_backend(self) -> str:
        """Post-attention reduction pattern: ``ag_rs`` | ``a2a`` | ``fi_a2a``."""
        if not self.enabled:
            return "ag_rs"
        return get_parallel().dcp_comm_backend

    # ------------------------------------------------------- head bookkeeping

    def num_kernel_heads(self, num_local_heads: int) -> int:
        """Widen an ``o_proj``-layout head count to the count the kernel evaluates."""
        return num_local_heads * self.size

    @property
    def head_shard_index(self) -> int:
        """Which chunk of the widened head set this rank keeps after the merge.

        Both reduction patterns are structural -- reduce-scatter and all-to-all
        each hand rank *r* chunk *r* -- so this is the DCP rank. It is named
        separately because it is the quantity ``o_proj`` alignment depends on
        (see ``check_layout``), not an interchangeable synonym for the rank.
        """
        return self.rank

    def local_head_offset(self, num_local_heads: int) -> int:
        """Offset of this rank's head shard inside the widened head set."""
        return self.head_shard_index * num_local_heads

    def narrow_local_heads(
        self, widened: torch.Tensor, num_local_heads: int, dim: int = 1
    ) -> torch.Tensor:
        """Select this rank's head shard from a widened tensor without communicating."""
        return widened.narrow(
            dim, self.local_head_offset(num_local_heads), num_local_heads
        )

    def check_layout(self) -> None:
        """Validate the rank layout the post-attention merge depends on.

        The Q all-gather concatenates the DCP partners' head shards in DCP-rank
        order, and the merge returns chunk ``head_shard_index`` of that
        concatenation. For ``o_proj`` -- which is sharded over the flat
        ``attn_tp`` layout -- to receive the head shard it owns, the chunk this
        rank keeps must be its own contribution, i.e.

            dcp_rank == attn_tp_rank % dcp_size

        which holds exactly while the DCP groups are contiguous, lowest-order
        slices of the attention TP group. It fails when ``attn_tp_size`` is not a
        multiple of ``dcp_size``, because DCP groups are carved out of the *full*
        TP group while attention heads are sharded over ``attn_tp_size`` only --
        so DP-attention or prefill-CP can leave a rank's DCP partners spanning
        several attention head shards.
        """
        if not self.enabled:
            return

        parallel = get_parallel()
        dcp_size = self.size
        attn_tp_size = parallel.attn_tp_size

        if attn_tp_size % dcp_size != 0:
            raise ValueError(
                f"attn_tp_size ({attn_tp_size}) must be a multiple of dcp_size "
                f"({dcp_size}). Decode context parallelism replicates each "
                "attention head shard across the DCP dimension, so the DCP "
                "group must fit inside one attention TP group. Got "
                f"attn_tp_size = tp_size ({parallel.tp_size}) // attn_dp_size "
                f"({parallel.attn_dp_size}) // attn_cp_size "
                f"({parallel.attn_cp_size}). Reduce --dcp-size, or reduce "
                "--dp-size / --attention-context-parallel-size."
            )

        expected = parallel.attn_tp_rank % dcp_size
        if self.head_shard_index != expected:
            raise ValueError(
                f"DCP head-shard mapping is inconsistent: head_shard_index="
                f"{self.head_shard_index} but attn_tp_rank "
                f"({parallel.attn_tp_rank}) % dcp_size ({dcp_size}) = {expected}. "
                "The post-attention merge would hand o_proj a different head "
                "shard than its weights were loaded for. This means the DCP "
                "group is no longer a contiguous, lowest-order slice of the "
                "attention TP group; head_shard_index must be remapped to match "
                "the new layout."
            )

    # --------------------------------------------------------- query gathering

    def gather_q_mla(
        self, q_nope_out: torch.Tensor, q_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Widen MLA queries from the ``o_proj`` head shard to the kernel head set."""
        return all_gather_q_for_mla_decode(q_nope_out=q_nope_out, q_pe=q_pe)

    def gather_q_heads(self, q: torch.Tensor) -> torch.Tensor:
        """Widen ``[B, H, D]`` queries along the head dim to the kernel head set.

        ``q`` may be a non-contiguous view; the copy that makes it gatherable is
        allocated from the group's symmetric-memory pool.
        """
        group = self.group
        with use_symmetric_memory(group):
            q = q.contiguous()
        return group.all_gather(q, dim=1).contiguous()

    # ---------------------------------------------------------------- reduction

    def combine_mla(
        self,
        attn_output: torch.Tensor,
        lse: torch.Tensor,
        *,
        attention_backend: Optional[str],
        cuda_graph_buffers: Optional[dict] = None,
    ) -> torch.Tensor:
        """Merge MLA partial outputs across DCP ranks.

        Takes ``[B, num_kernel_heads, D]`` partials and returns
        ``[B, num_local_heads, D]`` in the ``o_proj`` head layout, for every comm
        backend -- ``ag_rs`` produces a head-major result, so it is transposed
        back here rather than at each call site.
        """
        comm_backend = self.comm_backend
        base_on_e = is_lse_base_on_e(attention_backend)

        if comm_backend in ("a2a", "fi_a2a"):
            return dcp_a2a_lse_reduce(
                attn_output.contiguous(),
                lse.contiguous(),
                self.group,
                is_lse_base_on_e=base_on_e,
                cuda_graph_buffers=cuda_graph_buffers,
                comm_backend=comm_backend,
            )

        merged = cp_lse_ag_out_rs_mla(
            attn_output,
            lse,
            self.group,
            is_lse_base_on_e=base_on_e,
        )
        return merged.transpose(0, 1)

    def combine_mha(
        self,
        attn_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> torch.Tensor:
        """Merge MHA/GQA partial outputs across DCP ranks.

        Takes ``[B, num_kernel_heads, D]`` partials and returns
        ``[B, num_local_heads, D]``.
        """
        comm_backend = self.comm_backend

        if comm_backend in ("a2a", "fi_a2a"):
            return dcp_a2a_lse_reduce(
                attn_output.contiguous(),
                lse.contiguous(),
                self.group,
                is_lse_base_on_e=_MHA_LSE_IS_BASE_ON_E,
                comm_backend=comm_backend,
            )

        return cp_lse_ag_out_rs_mha(attn_output, lse, self.group)

    def combine_mha_with_lse(
        self, attn_output: torch.Tensor, lse: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge MHA/GQA partials and also return the merged LSE.

        Always uses ``ag_rs``, regardless of ``--dcp-comm-backend``: the caller
        (the chunked-prefix extend path) folds this result into the current
        chunk's output with a second LSE merge, and ``dcp_a2a_lse_reduce`` does
        not surface the merged LSE. Split from ``combine_mha`` so the constraint
        is visible in the signature rather than as a silent fallback.
        """
        return cp_lse_ag_out_rs_mha(attn_output, lse, self.group, return_lse=True)

    # ------------------------------------------------------------- lifecycle

    def init_workspace(self) -> None:
        """Allocate the ``fi_a2a`` MNNVL workspace. Must run before CUDA-graph capture."""
        if not self.enabled or self.comm_backend != "fi_a2a":
            return
        init_fi_a2a_workspace(self.group)


_DCP_ATTN_COMM = DcpAttnComm()


def get_dcp_attn_comm() -> DcpAttnComm:
    """Return the process-wide DCP attention communication accessor."""
    return _DCP_ATTN_COMM

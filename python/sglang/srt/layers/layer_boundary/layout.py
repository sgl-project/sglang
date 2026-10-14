# Copyright 2023-2024 SGLang Team
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
"""Token axes, layouts and process groups of what a layer hands across its
boundaries, and the configuration and batch facts they depend on."""

from enum import Enum, auto
from typing import Dict, FrozenSet, List, Mapping, Optional

import msgspec

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
)
from sglang.srt.layers.cp.utils import is_mla_cp_active, is_mla_cp_enabled
from sglang.srt.layers.dp_attention import (
    get_moe_cp_size,
    is_dp_attention_enabled,
    is_enable_moe_cp_allgather,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    is_moe_input_scattered_across_dp_ranks,
    post_experts_reduction_group,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel


class TokenAxis(Enum):
    """A parallel dimension across which ranks hold different tokens."""

    ATTN_DP = auto()
    ATTN_CP = auto()
    # Each attention-TP rank holds a slice of its group's tokens.
    ATTN_TP = auto()


class Layout(msgspec.Struct, frozen=True):
    """Describe token-row sharding under the active parallel topology.

    Fields:
        sharded: Axes across which ranks own different token rows. Factories
            omit size-one axes through sharded_over().

    This describes supported token partitions, not a general device mesh or
    hidden-dimension placement. Row order and padding come from the batch's
    shared token mapping, not from per-tensor metadata in this object.
    """

    sharded: FrozenSet[TokenAxis]

    @classmethod
    def sharded_over(
        cls, *axes: TokenAxis, axis_sizes: Mapping[TokenAxis, int]
    ) -> "Layout":
        return cls(frozenset(axis for axis in axes if axis_sizes[axis] > 1))


class SumGroup(Enum):
    """The group a stage's output is summed over, by name: the handle comes
    from get_parallel() when the sum runs."""

    ATTN_TP = auto()
    ATTN_CP = auto()
    TP = auto()
    # The group one all-reduce of a MoE output runs over.
    MOE_OUTPUT = auto()


def is_dense_ffn_fully_dp():
    return get_parallel().moe_dense_tp_size == 1


def batches_are_unpadded() -> bool:
    """Whether batches may reach the stages without padding to a multiple of
    attention TP: --disable-attn-tp-gather skips that padding unless attention
    DP is on."""
    parallel = get_parallel()
    return (
        parallel.attn_tp_size > 1
        and not parallel.attn_dp_enabled
        and parallel.disable_attn_tp_gather
    )


def _prefill_cp_shards_tokens() -> bool:
    """Whether the strategy prefill CP path shards prefill tokens across CP ranks."""
    parallel = get_parallel()
    return parallel.attn_cp_size > 1 and parallel.enable_prefill_cp


def input_scattered_configured() -> bool:
    """Whether the parallel configuration allows input-scattered attention:
    enabled, TP > 1, no attention DP, no prefill CP sharding the tokens, no
    a2a MoE backend and no dense FFN fully data-parallel. Stages bind the
    input-scattered variant exactly when this holds; the model's own
    conditions (``AttnTpContext.init_context``) may only narrow it."""
    parallel = get_parallel()
    return (
        parallel.enable_attn_tp_input_scattered
        and parallel.tp_size > 1
        and not parallel.attn_dp_enabled
        and not _prefill_cp_shards_tokens()
        and get_moe_a2a_backend().is_none()
        and not is_dense_ffn_fully_dp()
    )


def _batch_size(forward_batch: ForwardBatch) -> int:
    return (
        forward_batch.input_ids.shape[0] if hasattr(forward_batch, "input_ids") else 0
    )


def _ffn_has_tokens(forward_batch: ForwardBatch) -> bool:
    if is_dp_attention_enabled():
        # The FFN runs on every DP rank's tokens, so every rank decides alike.
        return (getattr(forward_batch, "global_dp_buffer_len", None) or 0) > 0
    return _batch_size(forward_batch) > 0


def moe_cp_gathered_rows(forward_batch: ForwardBatch) -> Optional[List[int]]:
    """The real rows each MoE-CP rank contributes when this batch's FFN input is
    gathered across the MoE-CP group, or None when it is not: only a context
    parallel extend with CP metadata gathers, and only when the group has more
    than one rank. The batch is read first, so a batch that is not a CP extend
    never reads the group."""
    if (
        forward_batch.forward_mode.is_context_parallel_extend()
        and forward_batch.attn_cp_metadata is not None
        and get_moe_cp_size() > 1
    ):
        return forward_batch.attn_cp_metadata.per_rank_actual_token
    return None


def moe_gathers_over_moe_cp() -> bool:
    """Whether a sparse MoE's input is gathered over the MoE-CP group on a CP
    extend: a MoE on the TP group under GQA CP whose MoE-CP group is wider than
    the MoE's data-parallel groups. DSA and MLA CP gather over attention CP
    instead."""
    return (
        not is_moe_input_scattered_across_dp_ranks()
        and is_enable_moe_cp_allgather()
        and not _cp_gathers_over_attn_cp()
    )


def batch_gathers_over_moe_cp(forward_batch: ForwardBatch) -> bool:
    """Whether a sparse MoE's input is gathered over the MoE-CP group on this
    batch: a CP extend, for a MoE that gathers there."""
    return moe_gathers_over_moe_cp() and moe_cp_gathered_rows(forward_batch) is not None


def _cp_shard_token_rows(forward_batch: ForwardBatch) -> List[int]:
    """Rows of each CP rank's shard that hold tokens; the shards are padded to
    one length after them."""
    metadata = forward_batch.attn_cp_metadata
    return metadata.per_rank_logical_token or metadata.per_rank_actual_token


def _sum_group(group: SumGroup) -> GroupCoordinator:
    parallel = get_parallel()
    if group is SumGroup.ATTN_TP:
        return parallel.attn_tp_group
    if group is SumGroup.ATTN_CP:
        return parallel.attn_cp_group
    if group is SumGroup.TP:
        return parallel.tp_group
    if group is SumGroup.MOE_OUTPUT:
        return post_experts_reduction_group()
    raise ValueError(f"unsupported output sum group: {group!r}")


def token_axis_sizes(*, cp_active: bool = False) -> Dict[TokenAxis, int]:
    """The token axes' sizes for a batch: attention CP shards tokens only on
    a CP extend (``cp_active``); otherwise every CP rank holds them all."""
    parallel = get_parallel()
    return {
        TokenAxis.ATTN_DP: parallel.attn_dp_size,
        TokenAxis.ATTN_CP: parallel.attn_cp_size if cp_active else 1,
        TokenAxis.ATTN_TP: parallel.attn_tp_size,
    }


def _cp_gathers_over_attn_cp() -> bool:
    """Whether a CP extend gathers the FFN input over the attention-CP group in
    equal shards and takes the output back with a reduce-scatter there: DSA and
    MLA CP. GQA prefill CP gathers over the MoE-CP group instead."""
    return is_dsa_enable_prefill_cp() or is_mla_cp_enabled()


def _batch_shards_over_cp(forward_batch: ForwardBatch) -> bool:
    """Whether this batch's tokens are split across the CP ranks. Only a context
    parallel extend is, so other batches, decode graph capture among them,
    never read the CP predicates."""
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    if _cp_gathers_over_attn_cp():
        return dsa_use_prefill_cp(forward_batch) or is_mla_cp_active(forward_batch)
    return moe_cp_gathered_rows(forward_batch) is not None


def _same_ranks(a: GroupCoordinator, b: GroupCoordinator) -> bool:
    return sorted(a.ranks) == sorted(b.ranks)

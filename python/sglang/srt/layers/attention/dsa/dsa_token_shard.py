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

"""Runtime gate and tensor plumbing for DSA token sharding.

Only the query path is sharded: after ``q_b_proj`` and the ``w_kc`` absorb a
rank swaps "my heads for every token" for "every head for my tokens", and the
swap is undone after attention, before ``w_vc``. The KV write and the indexer
stay full width.

Composes with the indexer's query sharding
(``SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING``): the indexer derives its
rows from this plan, so both pick the same rows.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    DsaTokenShardPlan,
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def dsa_token_shard_enabled() -> bool:
    """Asked at model build AND at forward, so both agree.

    ``is_npu()``: ``deepseek_v2.py`` is shared by every backend, and a true
    answer builds an extra full-head ``RadixAttention`` only NPU ever uses.
    """
    return (
        envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.get()
        and is_npu()
        and get_parallel().attn_tp_size > 1
    )


# Distinguishes "no plan cached yet" from "cached, and it is None".
_MISSING = object()


def get_dsa_token_shard_plan(
    forward_batch: "ForwardBatch",
) -> Optional[DsaTokenShardPlan]:
    """This forward's token slice for this rank, or None. Cached on the batch.

    Extend only: decode is already fast under graph capture. Every refusal is
    logged once, because a silent refusal looks the same as the feature running.
    """
    if not envs.SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD.get():
        return None
    cached = getattr(forward_batch, "npu_dsa_token_shard_plan", _MISSING)
    if cached is not _MISSING:
        return cached
    plan = _build_dsa_token_shard_plan(forward_batch)
    forward_batch.npu_dsa_token_shard_plan = plan
    return plan


def _build_dsa_token_shard_plan(forward_batch) -> Optional[DsaTokenShardPlan]:
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        print_info_once("DSA token-shard is off: attention TP size is 1")
        return None

    if parallel.dcp_enabled:
        # DCP's DSA path all-gathers the query across the DCP group, which
        # expects every head-sharded query row, not this rank's token slice.
        print_info_once("DSA token-shard is off: DCP all-gathers the attention query")
        return None

    # The same modes the indexer query shard admits.
    if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
        return None

    if get_attn_tp_context().input_scattered:
        # A scattered input is already sliced; slicing it again cuts twice.
        print_info_once(
            "DSA token-shard is off: the attention input is scattered across the "
            "TP group"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once(
            "DSA token-shard is off: this extend carries no CPU length metadata"
        )
        return None

    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    if plan.num_tokens < parallel.attn_tp_size:
        print_info_once(
            f"DSA token-shard is off for batches under {parallel.attn_tp_size} "
            f"tokens (this one has {plan.num_tokens}): the slice would be mostly "
            "padding"
        )
        return None

    # Logged on engage too: an unset flag returns before every refusal, so the
    # absence of a refusal proves nothing.
    print_info_once(
        f"DSA token-shard is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_token_shard_cumulative_query_lens(
    forward_batch: "ForwardBatch", plan: DsaTokenShardPlan, device: torch.device
) -> torch.Tensor:
    """Cumulative query lengths as an int32 device tensor, built once per
    forward rather than per layer: each build is a blocking H2D copy."""
    cached = getattr(forward_batch, "npu_dsa_token_shard_cu_query_lens", None)
    if cached is None:
        cached = torch.tensor(
            cumulative(plan.query_lens), dtype=torch.int32, device=device
        )
        forward_batch.npu_dsa_token_shard_cu_query_lens = cached
    return cached


def dsa_token_shard_slice(x: torch.Tensor, plan: DsaTokenShardPlan) -> torch.Tensor:
    """This rank's ``plan.rows`` rows, zero-padded at the tail to the row count
    the query has after the all-to-all."""
    sliced = x[plan.local_start : plan.local_end]
    missing = plan.rows - sliced.shape[0]
    if missing <= 0:
        return sliced
    return torch.cat([sliced, sliced.new_zeros((missing, *x.shape[1:]))], dim=0)


def dsa_token_shard_redistribute_heads(
    x: torch.Tensor, plan: DsaTokenShardPlan
) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by one all-to-all.

    Heads come out in global order: chunk ``s`` comes from rank ``s``, which a
    ColumnParallelLinear gives heads ``[s*h, (s+1)*h)``.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA token-shard was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad}. The batch is padded by a wider rule than "
        "attn_tp_size; the plan must be built from the padded width."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    # reshape, not view: q_nope_out arrives non-contiguous from
    # npu_transpose_batchmatmul.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_token_shard_restore_tokens(
    x: torch.Tensor, plan: DsaTokenShardPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``, the inverse.

    ``num_rows`` must be the width handed to the redistribute: a width that
    arrived padded must leave padded, or the next layer's KV write indexes a
    narrower tensor than its slot map. Rows past ``num_tokens`` are zeroed -- the operator never wrote
    them, and untouched device memory can be NaN.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA token-shard restore expects a full head set, got {x.shape[1]} for "
        f"tp_size {tp}"
    )
    assert plan.num_tokens <= num_rows <= plan.num_tokens_pad, (
        f"DSA token-shard restore asked for {num_rows} rows, outside "
        f"[{plan.num_tokens}, {plan.num_tokens_pad}]"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    out = recv.reshape(plan.num_tokens_pad, h, d)[:num_rows]
    if num_rows > plan.num_tokens:
        out[plan.num_tokens :].zero_()
    return out

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
"""The communication a boundary runs: the attention input, the FFN input and the
FFN output's way back, over attention TP, DP and CP."""

from typing import Callable, List, Optional, Tuple, Union

import torch

from sglang.srt.distributed import (
    GroupCoordinator,
    attention_tensor_model_parallel_all_reduce,
    attention_tensor_model_parallel_quant_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_reduce_scatter_tensor,
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    dp_gather_partial,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    get_moe_cp_rank,
    is_allocation_symmetric,
    moe_cp_all_gather_into_tensor,
)
from sglang.srt.layers.layer_boundary.adapters.attention import (
    attn_tp_gather,
    attn_tp_slice,
)
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    TokenAxis,
    _cp_shard_token_rows,
    moe_cp_gathered_rows,
)
from sglang.srt.layers.layer_boundary.residual import ResidualUpdate
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_parallel


def tp_reduce_scatter(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Complete a TP sum onto each rank's token slice."""
    parallel = get_parallel()
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    assert hidden_states.shape[0] % parallel.tp_size == 0, (
        f"Expected total tokens {hidden_states.shape[0]} % tp_size {parallel.tp_size} to be 0"
    )
    local_tokens = hidden_states.shape[0] // parallel.tp_size
    output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])
    parallel.tp_group.reduce_scatter_tensor(output, hidden_states)
    if residual is not None:
        residual = residual.tensor_split(parallel.tp_size)[parallel.tp_rank]
    return output, residual


def tp_slice(hidden_states, residual):
    """The rows reduce-scatter would return, when the sum is already complete."""
    parallel = get_parallel()
    hidden_states = hidden_states.tensor_split(parallel.tp_size)[
        parallel.tp_rank
    ].clone()
    if residual is not None:
        residual = residual.tensor_split(parallel.tp_size)[parallel.tp_rank]
    return hidden_states, residual


def attn_tp_gather_input(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    forward_batch: ForwardBatch,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    parallel = get_parallel()
    if isinstance(hidden_states, tuple):
        gathered_hidden_states = []
        for local_hidden_states in hidden_states:
            with use_symmetric_memory(
                parallel.tp_group,
                disabled=not is_allocation_symmetric(),
            ):
                output = torch.empty(
                    (
                        local_hidden_states.shape[0] * parallel.attn_tp_size,
                        *local_hidden_states.shape[1:],
                    ),
                    dtype=local_hidden_states.dtype,
                    device=local_hidden_states.device,
                )
            attn_tp_all_gather_into_tensor(
                output,
                local_hidden_states,
            )
            gathered_hidden_states.append(output)
        return tuple(gathered_hidden_states)

    return attn_tp_gather(hidden_states)


def attn_tp_reduce_scatter(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    parallel = get_parallel()
    local_hidden_states = hidden_states.tensor_split(parallel.attn_tp_size)[
        parallel.attn_tp_rank
    ]
    attn_tp_reduce_scatter_tensor(local_hidden_states, hidden_states)
    return local_hidden_states


def attn_cp_interleave_reduce_scatter(hidden_states: torch.Tensor):
    """Sum rank-major output onto each rank's equal, padded interleave shard."""
    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


def moe_cp_gather(
    hidden_states: torch.Tensor, rows: List[int], moe_cp_size: int
) -> torch.Tensor:
    # Zigzag split can produce unequal token counts across CP ranks
    # (when seq_len % (cp_size * 2) != 0). NCCL allgather requires
    # equal input sizes, so pad to the max per-rank token count.
    max_tokens = max(rows)
    pad_size = max_tokens - hidden_states.shape[0]
    if pad_size > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, [0, 0, 0, pad_size])

    output = torch.empty(
        (max_tokens * moe_cp_size, hidden_states.shape[1]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    moe_cp_all_gather_into_tensor(output, hidden_states)
    return output


def attn_tp_all_reduce(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch, may_quantize: bool = True
) -> torch.Tensor:
    if (
        may_quantize
        and not forward_batch.forward_mode.is_decode_or_idle()
        and get_exec().comm.enable_quant_communications
    ):
        return attention_tensor_model_parallel_quant_all_reduce(hidden_states)
    return attention_tensor_model_parallel_all_reduce(hidden_states)


def dp_gather(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_replicate(
        global_hidden_states, hidden_states, forward_batch, cp_shard_counts
    )
    return global_hidden_states


def dp_gather_sum(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_partial(
        global_hidden_states, hidden_states, forward_batch, cp_shard_counts
    )
    return global_hidden_states


def move_rows(
    hidden_states: torch.Tensor,
    rows: Layout,
    to: Layout,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """A complete value from the rows it is on to ``to``: gathered over the
    token axes ``to`` does not shard (attention TP, then attention DP), or cut to
    this rank's share of those it does (attention DP, then attention TP)."""
    gathered, cut = rows.sharded - to.sharded, to.sharded - rows.sharded
    if (gathered and cut) or TokenAxis.ATTN_CP in gathered | cut:
        raise NotImplementedError(f"{rows=} {to=}")
    if TokenAxis.ATTN_TP in gathered:
        hidden_states = attn_tp_gather(hidden_states)
    if TokenAxis.ATTN_DP in gathered:
        hidden_states = dp_gather(hidden_states, forward_batch)
    if TokenAxis.ATTN_DP in cut:
        hidden_states = to_dp_local(_dp_scatter_step, forward_batch, hidden_states)
    if TokenAxis.ATTN_TP in cut:
        parallel = get_parallel()
        hidden_states = hidden_states.tensor_split(parallel.attn_tp_size)[
            parallel.attn_tp_rank
        ]
    return hidden_states


def _dp_scatter_group() -> GroupCoordinator:
    parallel = get_parallel()
    if parallel.tp_size == parallel.attn_dp_size:
        return parallel.tp_group
    return parallel.attn_tp_group


def dp_reduce_scatterv(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    get_parallel().tp_group.reduce_scatterv(
        hidden_states,
        output=local_hidden_states,
        sizes=get_dp_global_num_tokens(),
    )


def dp_reduce_scatter(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_reduce_scatter_tensor(local_hidden_states, hidden_states)


def _dp_scatter_step(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_scatter(local_hidden_states, hidden_states, forward_batch)


def moe_cp_take_back(hidden_states: torch.Tensor, rows: List[int]) -> torch.Tensor:
    moe_cp_rank = get_moe_cp_rank()
    # The allgather was padded to max_tokens_per_rank (equal chunks).
    # Extract this rank's actual (non-padded) tokens from its chunk.
    max_tokens_per_rank = max(rows)
    actual_local_tokens = rows[moe_cp_rank]
    return hidden_states.narrow(
        0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens
    ).contiguous()


def to_dp_local(
    step: Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None],
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    local_hidden_states = get_local_dp_buffer(_dp_scatter_group())
    step(local_hidden_states, hidden_states, forward_batch)
    return local_hidden_states


def all_reduce_to_dp_local(
    group: GroupCoordinator, forward_batch: ForwardBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    return to_dp_local(_dp_scatter_step, forward_batch, group.all_reduce(hidden_states))


def keep_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    **kwargs,
):
    return hidden_states, residual


def update_attn_tp_gather_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    update: ResidualUpdate = PLAIN_ADD,
    **kwargs,
):
    hidden_states = update.update(hidden_states, residual)
    return attn_tp_gather(hidden_states), None


def residual_slice_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    sums: bool,
    gathers_back: bool,
    update: ResidualUpdate,
    **kwargs,
):
    """Bring the FFN output onto the slice of the rows the residual is on:
    a reduce-scatter that also completes its sum when ``sums``, else this
    rank's slice of the complete output. With ``gathers_back`` it is
    written into the residual there and the full rows are gathered back."""
    parallel = get_parallel()
    if sums:
        hidden_states, _ = tp_reduce_scatter(hidden_states, None)
    else:
        hidden_states = hidden_states.tensor_split(parallel.tp_size)[parallel.tp_rank]
    if not gathers_back:
        return hidden_states, residual
    local_states = update.update(hidden_states, residual)
    hidden_states = local_states.new_empty(
        local_states.shape[0] * parallel.tp_size, *local_states.shape[1:]
    )
    parallel.tp_group.all_gather_into_tensor(hidden_states, local_states)
    return hidden_states, None


def attn_tp_slice_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
):
    assert residual is None, "not yet handled residual!=None"
    return attn_tp_slice(hidden_states), None


def attn_cp_take_back_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    **kwargs,
):
    """DSA and MLA CP: this rank's shard of a complete output gathered in
    equal shards over the attention-CP group."""
    parallel = get_parallel()
    shard = hidden_states.tensor_split(parallel.attn_cp_size)[parallel.attn_cp_rank]
    return shard, residual


def attn_cp_reduce_scatter_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    **kwargs,
):
    """DSA and MLA CP: sum the FFN output over the attention-CP group and
    keep this rank's shard."""
    return attn_cp_interleave_reduce_scatter(hidden_states), residual


def dp_cp_take_back_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    **kwargs,
):
    """This rank's CP shard of the rows the DP gather put in its DP group's
    slot, at the shard's padded length with the padding zeroed."""
    held = forward_batch.attn_cp_metadata.per_rank_actual_token
    local_hidden_states = get_local_dp_buffer(_dp_scatter_group())[
        : held[get_parallel().attn_cp_rank]
    ]
    dp_scatter(
        local_hidden_states,
        hidden_states,
        forward_batch,
        _cp_shard_token_rows(forward_batch),
    )
    return local_hidden_states, residual


def moe_cp_take_back_output(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    **kwargs,
):
    """Return a MoE output computed on the MoE-CP-gathered rows to this rank's attention rows.

    After moe_tensor_model_parallel_all_reduce (which runs unconditionally since
    mlp_reduce_scatter=False for this path), all ranks in the moe_cp group hold the
    full MoE result for all cp_per_moe token chunks. We simply slice out this rank's
    CP-local portion.

    If DP>1, further scatter back to the local DP slice.
    """
    # Only scatter back during prefill; decode was never allgathered so no-op.
    # CP extend has non-zero tokens per rank, and decode skips this path.
    rows = moe_cp_gathered_rows(forward_batch)
    if rows is not None:
        hidden_states = moe_cp_take_back(hidden_states, rows)

    if get_parallel().attn_dp_size > 1:
        hidden_states = to_dp_local(_dp_scatter_step, forward_batch, hidden_states)

    return hidden_states, residual

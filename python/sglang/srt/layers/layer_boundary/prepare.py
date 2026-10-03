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
"""Complete pending output, update the residual, and read the consumer input."""

from typing import Callable, Optional, Tuple, Union

import torch

from sglang.srt.distributed import (
    attention_tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.interleave import (
    attn_cp_interleave_gather,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    dp_scatter,
    get_moe_cp_size,
    is_allocation_symmetric,
)
from sglang.srt.layers.layer_boundary.adapters.attention import (
    AttentionInputs,
    attn_tp_gather,
    attn_tp_slice,
    get_attn_tp_context,
    tp_gather,
)
from sglang.srt.layers.layer_boundary.layout import (
    SumGroup,
    _cp_shard_token_rows,
    _sum_group,
)
from sglang.srt.layers.layer_boundary.output import (
    DeferredFinalize,
    UnreducedOutput,
    complete_owed,
)
from sglang.srt.layers.layer_boundary.residual import ResidualReadout, ResidualUpdate
from sglang.srt.layers.layer_boundary.residual.add_norm import NORM_READOUT, PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import DeclaredSum
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    is_npu,
)

_is_npu = is_npu()


if _is_npu:
    from sglang.srt.hardware_backend.npu.cmo import prepare_weight_cache

from sglang.srt.layers.layer_boundary.ops import (
    attn_tp_all_reduce,
    attn_tp_reduce_scatter,
    dp_gather,
    dp_gather_sum,
    moe_cp_gather,
)


def _reduce_update_read(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    gathers_residual: bool,
    fusions: Tuple[Callable, ...],
    group: SumGroup = SumGroup.ATTN_TP,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    """Complete the sum the input owes over ``group`` on the rows it is on,
    unless one of ``fusions`` does it with the residual add and the norm, then
    write it into the residual and read the input."""
    if gathers_residual:
        residual = update.gather_residual_attn_tp(residual)
    for fused in fusions:
        result = fused(hidden_states, residual, forward_batch)
        if result is not None:
            return result
    if group is SumGroup.ATTN_TP:
        # MHC sums its streams in full precision.
        hidden_states = attn_tp_all_reduce(
            hidden_states, forward_batch, may_quantize=update.is_plain_add
        )
    elif group is SumGroup.TP:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    elif group is not None:
        hidden_states = _sum_group(group).all_reduce(hidden_states)
    if _is_npu and cache is not None:
        _ = prepare_weight_cache(hidden_states, cache)
    return read.update_and_read(update, hidden_states, residual, norm)


def _reduce_update_read_dp_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    gathers_residual: bool,
    reduces_attention_tp: bool,
    places_cp_shards: bool = False,
    group: SumGroup = SumGroup.ATTN_TP,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    """Attention DP: complete the attention-TP sum if it is owed, write the
    output into the residual and read the FFN input locally, then gather. With
    ``places_cp_shards`` each CP rank puts its shard of the DP group's tokens
    beside the others' in the group's slot."""
    if gathers_residual:
        residual = update.gather_residual_attn_tp(residual)
    if hidden_states.shape[0] != 0:
        if reduces_attention_tp:
            hidden_states = (
                attention_tensor_model_parallel_all_reduce(hidden_states)
                if group is SumGroup.ATTN_TP
                else _sum_group(group).all_reduce(hidden_states)
            )
        with use_symmetric_memory(
            get_parallel().tp_group,
            disabled=not is_allocation_symmetric(),
        ):
            hidden_states, residual = read.update_and_read(
                update, hidden_states, residual, norm
            )
    else:
        hidden_states, residual = read.update_and_read(
            update, hidden_states, residual, norm
        )
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    return (
        dp_gather(hidden_states, forward_batch, cp_shard_counts),
        residual,
    )


def _dp_gather_sum_read(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    gathers_residual: bool,
    places_cp_shards: bool = False,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    """Attention DP: one rank adds the residual, the gather sums it, then read
    the input from the sum. With ``places_cp_shards`` each CP rank puts its
    shard of the DP group's tokens beside the others' in the group's slot."""
    if gathers_residual:
        residual = attn_tp_gather(residual)
    if get_parallel().attn_tp_rank == 0:
        hidden_states += residual
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    hidden_states = dp_gather_sum(hidden_states, forward_batch, cp_shard_counts)
    dp_scatter(residual, hidden_states, forward_batch, cp_shard_counts)
    hidden_states, _ = read.read(hidden_states, norm)
    return hidden_states, residual


def _attn_input_default(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    qkv_latent_func: Optional[Callable],
) -> torch.Tensor:
    """Give the attention's QKV hook the attention input."""
    if qkv_latent_func is not None:
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, qkv_latent_func)
        )
    return hidden_states


def _attn_input_scattered(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    qkv_latent_func: Optional[Callable],
) -> torch.Tensor:
    """Input-scattered attention takes each rank's slice, and its QKV hook
    gathers the rows after the projection. DSA and attention without a hook
    consume full hidden states, so those are gathered here, and the hook is
    told they are."""
    ctx = get_attn_tp_context()
    if ctx.is_dsa or qkv_latent_func is None:
        hidden_states = tp_gather(hidden_states, forward_batch)
    if qkv_latent_func is not None:
        ctx.set_attn_inputs(
            AttentionInputs(
                hidden_states,
                forward_batch,
                qkv_latent_func,
                is_pre_gathered=ctx.is_dsa,
            )
        )
    return hidden_states


def _dispatch_by_update(
    hidden_states, residual, forward_batch, norm, *, paths, update=PLAIN_ADD, **call
):
    if residual is None:
        # A missing residual cannot join a partial sum. The non-plain path
        # performs init_residual/read without that optimization.
        prepare = paths[False]
    else:
        try:
            prepare = paths[update.is_plain_add]
        except KeyError:
            raise RuntimeError("producer update has no bound input path") from None
    return prepare(hidden_states, residual, forward_batch, norm, update=update, **call)


def _run_entry(
    hidden_states: Union[torch.Tensor, UnreducedOutput, DeferredFinalize],
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    step: Callable,
    is_plain_add: bool,
    carried_fusions: Tuple[Callable, ...],
    expected_sum: Optional[SumGroup] = None,
    completed_step: Optional[Callable] = None,
    pending=None,
    written_step: Optional[Callable] = None,
    written: bool = False,
    update: ResidualUpdate = PLAIN_ADD,
    **call,
):
    """A boundary's half into a stage: complete what the value carries, a sum
    or a handoff its producer left for this batch, then run ``step`` on the
    complete value. Under attention DP the reduction back to this rank's tokens
    comes first; otherwise one of ``carried_fusions`` may complete it with the
    residual add and the read, and an empty batch has nothing to sum. An input
    whose declared sum is already completed uses ``completed_step`` instead;
    a pending declared sum must match ``expected_sum``. ``call`` is what the stage's read takes (the attention's
    ``quant_format`` and ``post_residual_addition``)."""
    if residual is not None and update.is_plain_add != is_plain_add:
        raise RuntimeError("producer update does not match the boundary's capability")
    if pending is not None:
        if isinstance(pending.owed, DeclaredSum):
            if pending.owed.group is not expected_sum:
                raise RuntimeError("producer sum does not match the input declaration")
        elif expected_sum is not None:
            if pending.owed is not None:
                raise RuntimeError("a declared input sum carried another completion")
            step = completed_step
    if written and written_step is not None:
        step = written_step
    if not isinstance(hidden_states, torch.Tensor):
        owed = hidden_states
        if residual is None:
            raise RuntimeError(f"{type(owed).__name__} requires residual input")
        if expected_sum is not None:
            raise RuntimeError(
                f"an input that owes its sum by construction arrived as "
                f"{type(owed).__name__}"
            )
        if isinstance(owed, UnreducedOutput) and owed.reduce_to_dp_local is not None:
            hidden_states = complete_owed(owed)
        else:
            for fused in carried_fusions:
                result = fused(
                    owed, residual, forward_batch, call.get("post_residual_addition")
                )
                if result is not None:
                    return result
            if isinstance(owed, DeferredFinalize) or owed.partial.shape[0] != 0:
                hidden_states = complete_owed(owed)
            else:
                hidden_states = owed.partial
    return step(hidden_states, residual, forward_batch, norm, update=update, **call)


def _update_read(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    pre_move: Optional[Callable],
    enters_stack: bool,
    read: ResidualReadout,
    update: ResidualUpdate = PLAIN_ADD,
    quant_format: str = "",
    post_residual_addition: Optional[torch.Tensor] = None,
):
    """Complete what the input owes by construction (``pre_move``), then
    write the previous stage's output into the residual and read the stage's
    input with ``norm``. The layer stack's first stage (``enters_stack``)
    starts its residual from its input."""
    enters = residual is None and enters_stack
    if pre_move is not None:
        hidden_states, residual = pre_move(hidden_states, residual)
    if enters:
        hidden_states, residual = read.init_residual(hidden_states), None
    if residual is None:
        # The previous layer already wrote its output into the residual.
        return read.read(hidden_states, norm, quant_format)
    return read.update_and_read(
        update,
        hidden_states,
        residual,
        norm,
        quant_format=quant_format,
        post_residual_addition=post_residual_addition,
    )


def _attn_tp_reduce_scatter_update_read(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    scatters_residual: bool,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    hidden_states = attn_tp_reduce_scatter(hidden_states)
    if scatters_residual:
        residual = update.slice_residual_attn_tp(residual)
    return read.update_and_read(update, hidden_states, residual, norm)


def _attn_tp_slice_update_read(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    scatters_residual: bool,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    hidden_states = attn_tp_slice(hidden_states).clone()
    if scatters_residual and residual is not None:
        residual = update.slice_residual_attn_tp(residual)
    return read.update_and_read(update, hidden_states, residual, norm)


def _tp_reduce_scatter_update_read_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    reduces: bool = True,
    read: ResidualReadout,
    update: ResidualUpdate,
):
    """The residual stays on each rank's slice while the FFN takes the full
    rows: reduce-scatter the attention output onto the slice, which completes
    its sum, write it into the residual and read the FFN input there, then
    gather the input back into the attention output's rows."""
    parallel = get_parallel()
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    shard = hidden_states.tensor_split(parallel.tp_size)[parallel.tp_rank]
    if reduces:
        parallel.tp_group.reduce_scatter_tensor(shard, hidden_states)
    else:
        shard = shard.clone()
    shard, residual = read.update_and_read(update, shard, residual, norm)
    attn_tp_all_gather_into_tensor(hidden_states, shard)
    return hidden_states, residual


def _tp_sum_with_residual_read(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    read: ResidualReadout = NORM_READOUT,
    update: ResidualUpdate = PLAIN_ADD,
):
    return _tp_all_reduce_with_scattered_residual(hidden_states, residual, norm, read)


def _then_attn_cp_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    gather: Callable,
    update: ResidualUpdate = PLAIN_ADD,
    **call,
):
    """DSA and MLA CP: complete this rank's shard, then gather the shards, of
    equal length, over the attention-CP group. The residual stays on the
    shard."""
    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, norm, update=update, cache=cache, **call
    )
    return attn_cp_interleave_gather(hidden_states), residual


def _move_before_read(
    hidden_states, residual, forward_batch, norm, *, move, read, **call
):
    """Complete and place a producer contribution before its residual update."""
    hidden_states, residual = move(hidden_states, residual, forward_batch)
    return read(hidden_states, residual, forward_batch, norm, **call)


def _then_moe_cp_gather(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    *,
    cache=None,
    gather: Callable,
    update: ResidualUpdate = PLAIN_ADD,
):
    """Read each shard, then gather the MoE group's tokens on its CP path.

    StagePlan selects this path only for a CP extend with MoE-CP rows; zigzag
    eligibility guarantees nonempty shards. Decode and non-CP batches use the
    ordinary path. The residual stays on this rank's attention rows.
    """
    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, norm, update=update, cache=cache
    )
    hidden_states = moe_cp_gather(
        hidden_states,
        forward_batch.attn_cp_metadata.per_rank_actual_token,
        get_moe_cp_size(),
    )
    return hidden_states, residual


def _tp_all_reduce_with_scattered_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
    read: ResidualReadout = NORM_READOUT,
):
    parallel = get_parallel()
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states

    scattered_states = hidden_states.tensor_split(parallel.tp_size)[parallel.tp_rank]
    scattered_states += residual
    residual = tensor_model_parallel_all_reduce(hidden_states)
    return read.read(residual, norm)

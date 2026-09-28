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
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.communicator.adapters.attention import (
    AttentionInputs,
    _redistribute_from_attn_tp_shards,
    _redistribute_to_attn_tp_shards,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator.layout import (
    CommunicateContext,
    Layout,
    SumGroup,
    TokenAxis,
    _cp_shard_token_rows,
    moe_cp_gathered_rows,
)
from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.communicator.residual import StageRead, StageUpdate
from sglang.srt.layers.communicator.residual.add_norm import ADD, NORM_READ
from sglang.srt.layers.communicator_dsa_cp import (
    dsa_cp_gather_hidden_states,
    dsa_cp_reduce_scatter_hidden_states,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    can_use_dp_reduce_scatter,
    dp_gather_partial,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    get_moe_cp_rank,
    get_moe_cp_size,
    is_allocation_symmetric,
    moe_cp_all_gather_into_tensor,
)
from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import (
    is_npu,
)

_is_npu = is_npu()


if _is_npu:
    from sglang.srt.hardware_backend.npu.cmo import prepare_weight_cache


def tp_reduce_scatter(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    context: "CommunicateContext",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Module-level so MHC communicators can reuse it without a
    ``LayerCommunicator`` instance."""
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    assert hidden_states.shape[0] % context.tp_size == 0, (
        f"Expected total tokens {hidden_states.shape[0]} % tp_size {context.tp_size} to be 0"
    )
    local_tokens = hidden_states.shape[0] // context.tp_size
    output = hidden_states.new_empty(local_tokens, *hidden_states.shape[1:])
    get_parallel().tp_group.reduce_scatter_tensor(output, hidden_states)
    if residual is not None:
        residual = residual.tensor_split(context.tp_size)[context.tp_rank]
    return output, residual


def layer_input_buffer(
    hidden_states: Union[torch.Tensor, UnreducedOutput],
) -> torch.Tensor:
    """The tensor holding a layer's input, for reusing its memory without reading it."""
    if isinstance(hidden_states, UnreducedOutput):
        return hidden_states.partial
    return hidden_states


class CommunicateSimpleFn:
    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:
        return hidden_states

    @staticmethod
    def _scattered_to_tp_attn_full(
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(hidden_states, tuple):
            gathered_hidden_states = []
            for local_hidden_states in hidden_states:
                with use_symmetric_memory(
                    get_parallel().tp_group,
                    disabled=not is_allocation_symmetric(),
                ):
                    output = torch.empty(
                        (
                            local_hidden_states.shape[0] * context.attn_tp_size,
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

        return _redistribute_from_attn_tp_shards(hidden_states)


def _reduce_and_redistribute_output_to_attn_tp_shards(
    hidden_states: torch.Tensor, context: CommunicateContext
) -> torch.Tensor:
    local_hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
        context.attn_tp_rank
    ]
    attn_tp_reduce_scatter_tensor(local_hidden_states, hidden_states)
    return local_hidden_states


def _redistribute_input_to_moe_cp(
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


def _mlp_input_reduce_output(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch, may_quantize: bool = True
) -> torch.Tensor:
    if (
        may_quantize
        and not forward_batch.forward_mode.is_decode_or_idle()
        and get_exec().comm.enable_quant_communications
    ):
        return attention_tensor_model_parallel_quant_all_reduce(hidden_states)
    return attention_tensor_model_parallel_all_reduce(hidden_states)


def _redistribute_input_to_dp(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    cp_shard_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    global_hidden_states = get_global_dp_buffer(get_parallel().tp_group)
    dp_gather_replicate(
        global_hidden_states, hidden_states, forward_batch, cp_shard_counts
    )
    return global_hidden_states


def _reduce_and_redistribute_output_to_dp(
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
    if TokenAxis.ATTN_TP_SCATTER in gathered:
        hidden_states = _redistribute_from_attn_tp_shards(hidden_states)
    if TokenAxis.ATTN_DP in gathered:
        hidden_states = _redistribute_input_to_dp(hidden_states, forward_batch)
    if TokenAxis.ATTN_DP in cut:
        hidden_states = _to_local_tokens(
            _redistribute_output, forward_batch, hidden_states
        )
    if TokenAxis.ATTN_TP_SCATTER in cut:
        parallel = get_parallel()
        hidden_states = hidden_states.tensor_split(parallel.attn_tp_size)[
            parallel.attn_tp_rank
        ]
    return hidden_states


def _mlp_input_without_dp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    fusions: Tuple[Callable, ...],
    group: SumGroup = SumGroup.ATTN_TP,
    read: StageRead = NORM_READ,
    update: StageUpdate = ADD,
):
    """Complete the sum the input owes over ``group`` on the rows it is on,
    unless one of ``fusions`` does it with the residual add and the norm, then
    write it into the residual and read the input."""
    if gathers_residual:
        residual = update.residual_from_attn_tp_shards(residual)
    for fused in fusions:
        result = fused(hidden_states, residual, forward_batch)
        if result is not None:
            return result
    if group is SumGroup.ATTN_TP:
        # MHC sums its streams in full precision.
        hidden_states = _mlp_input_reduce_output(
            hidden_states, forward_batch, may_quantize=update.adds_plainly
        )
    else:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
    if _is_npu and context.cache is not None:
        _ = prepare_weight_cache(hidden_states, context.cache)
    return read.update_and_read(update, hidden_states, residual, layernorm)


def _mlp_input_dp_replicate(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    reduces_attention_tp: bool,
    places_cp_shards: bool = False,
    read: StageRead = NORM_READ,
    update: StageUpdate = ADD,
):
    """Attention DP: complete the attention-TP sum if it is owed, write the
    output into the residual and read the FFN input locally, then gather. With
    ``places_cp_shards`` each CP rank puts its shard of the DP group's tokens
    beside the others' in the group's slot."""
    if gathers_residual:
        residual = update.residual_from_attn_tp_shards(residual)
    if hidden_states.shape[0] != 0:
        if reduces_attention_tp:
            hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        with use_symmetric_memory(
            get_parallel().tp_group,
            disabled=not is_allocation_symmetric(),
        ):
            hidden_states, residual = read.update_and_read(
                update, hidden_states, residual, layernorm
            )
    else:
        hidden_states, residual = read.update_and_read(
            update, hidden_states, residual, layernorm
        )
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    return (
        _redistribute_input_to_dp(hidden_states, forward_batch, cp_shard_counts),
        residual,
    )


def _mlp_input_dp_partial(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gathers_residual: bool,
    places_cp_shards: bool = False,
    read: StageRead = NORM_READ,
):
    """Attention DP: one rank adds the residual, the gather sums it, then read
    the input from the sum. With ``places_cp_shards`` each CP rank puts its
    shard of the DP group's tokens beside the others' in the group's slot."""
    if gathers_residual:
        residual = _redistribute_from_attn_tp_shards(residual)
    if context.attn_tp_rank == 0:
        hidden_states += residual
    cp_shard_counts = _cp_shard_token_rows(forward_batch) if places_cp_shards else None
    hidden_states = _reduce_and_redistribute_output_to_dp(
        hidden_states, forward_batch, cp_shard_counts
    )
    dp_scatter(residual, hidden_states, forward_batch, cp_shard_counts)
    hidden_states, _ = read.read(hidden_states, layernorm)
    return hidden_states, residual


def _hand_qkv_hook_its_input(
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


def _hand_scattered_input_to_attention(
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
        hidden_states = _tp_all_gather_scattered_rows(hidden_states, forward_batch)
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


def _tp_all_gather_scattered_rows(
    hidden_states: torch.Tensor, forward_batch: ForwardBatch
) -> torch.Tensor:
    # Input-scattered attention keeps the same number of tokens on every TP rank.
    total_tokens = forward_batch.input_ids.shape[0]
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_parallel().tp_group.all_gather_into_tensor(output, hidden_states)
    return output


def _consumer_step(
    hidden_states: Union[torch.Tensor, UnreducedOutput, HandoffOutput],
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    context: CommunicateContext,
    *,
    step: Callable,
    carried_fusions: Tuple[Callable, ...],
    owes_by_construction: bool = False,
    **call,
):
    """A boundary's half into a stage: complete what the value carries, a sum
    or a handoff its producer left for this batch, then run ``step`` on the
    complete value. Under attention DP the reduction back to this rank's tokens
    comes first; otherwise one of ``carried_fusions`` may complete it with the
    residual add and the read, and an empty batch has nothing to sum. An input
    that owes its sum by construction (``owes_by_construction``) is that sum's
    only carrier. ``call`` is what the stage's read takes (the attention's
    ``quant_format`` and ``post_residual_addition``)."""
    if not isinstance(hidden_states, torch.Tensor):
        owed = hidden_states
        if residual is None:
            raise RuntimeError(f"{type(owed).__name__} requires residual input")
        if owes_by_construction:
            raise RuntimeError(
                f"an input that owes its sum by construction arrived as "
                f"{type(owed).__name__}"
            )
        if (
            isinstance(owed, UnreducedOutput)
            and owed.reduce_and_redistribute is not None
        ):
            hidden_states = reduce_output(owed)
        else:
            for fused in carried_fusions:
                result = fused(
                    owed, residual, forward_batch, call.get("post_residual_addition")
                )
                if result is not None:
                    return result
            if isinstance(owed, HandoffOutput) or owed.partial.shape[0] != 0:
                hidden_states = reduce_output(owed)
            else:
                hidden_states = owed.partial
    return step(hidden_states, residual, forward_batch, norm, context, **call)


def _read_input(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    context: CommunicateContext,
    *,
    layer_input: Optional[Callable],
    enters_stack: bool,
    read: StageRead,
    update: StageUpdate,
    quant_format: str = "",
    post_residual_addition: Optional[torch.Tensor] = None,
):
    """Complete what the input owes by construction (``layer_input``), then
    write the previous stage's output into the residual and read the stage's
    input with ``norm``. The layer stack's first stage (``enters_stack``)
    starts its residual from its input."""
    enters = residual is None and enters_stack
    if layer_input is not None:
        hidden_states, residual = layer_input(hidden_states, residual, context)
    if enters:
        hidden_states, residual = read.enter(hidden_states), None
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


def _mlp_input_scatter(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    scatters_residual: bool,
    read: StageRead = NORM_READ,
    update: StageUpdate = ADD,
):
    hidden_states = _reduce_and_redistribute_output_to_attn_tp_shards(
        hidden_states, context
    )
    if scatters_residual:
        residual = update.residual_to_attn_tp_shard(residual, context)
    return read.update_and_read(update, hidden_states, residual, layernorm)


def _mlp_input_slice(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    scatters_residual: bool,
    read: StageRead = NORM_READ,
    update: StageUpdate = ADD,
):
    hidden_states = _redistribute_to_attn_tp_shards(hidden_states, context).clone()
    if scatters_residual and residual is not None:
        residual = update.residual_to_attn_tp_shard(residual, context)
    return read.update_and_read(update, hidden_states, residual, layernorm)


def _mlp_input_on_residual_shard(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    read: StageRead,
    update: StageUpdate,
):
    """The residual stays on each rank's slice while the FFN takes the full
    rows: reduce-scatter the attention output onto the slice, which completes
    its sum, write it into the residual and read the FFN input there, then
    gather the input back into the attention output's rows."""
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    shard = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
    get_parallel().tp_group.reduce_scatter_tensor(shard, hidden_states)
    shard, residual = read.update_and_read(update, shard, residual, layernorm)
    attn_tp_all_gather_into_tensor(hidden_states, shard)
    return hidden_states, residual


def _mlp_input_residual_into_sum(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    read: StageRead = NORM_READ,
):
    return _tp_all_reduce_with_scattered_residual(
        hidden_states, residual, layernorm, context, read
    )


def _tp_all_reduce_with_scattered_residual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    read: StageRead = NORM_READ,
):
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states

    scattered_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
    scattered_states += residual
    residual = tensor_model_parallel_all_reduce(hidden_states)
    return read.read(residual, layernorm)


def _mlp_input_gather_attention_cp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gather: Callable,
):
    """DSA and MLA CP: complete this rank's shard, then gather the shards, of
    equal length, over the attention-CP group. The residual stays on the
    shard."""
    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, layernorm, context
    )
    return dsa_cp_gather_hidden_states(hidden_states), residual


def _mlp_input_gather_moe_cp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    layernorm: torch.nn.Module,
    context: CommunicateContext,
    *,
    gather: Callable,
):
    """Gather for the FFN, then over the MoE-CP group so each rank holds all
    tokens of its MoE group (moe_dp_size < attn_cp_size). The residual stays at
    TP_ATTN_FULL."""
    # Early return on empty tensor is safe for MOE_CP because:
    # - During CP extend: zigzag split guarantees all CP ranks have non-zero tokens,
    #   so no rank hits this path while others proceed to the allgather.
    # - During decode: moe_cp allgather is skipped (guarded by is_context_parallel_extend).
    # - CUDA graph warmup: not applicable when --cuda-graph-backend-prefill=disabled is used.
    if hidden_states.shape[0] == 0:
        return hidden_states, residual

    hidden_states, residual = gather(
        hidden_states, residual, forward_batch, layernorm, context
    )

    rows = moe_cp_gathered_rows(forward_batch)
    if rows is not None and hidden_states.shape[0] > 0:
        hidden_states = _redistribute_input_to_moe_cp(
            hidden_states, rows, get_moe_cp_size()
        )

    return hidden_states, residual


def _dp_scatter_group() -> GroupCoordinator:
    parallel = get_parallel()
    if parallel.tp_size == parallel.attn_dp_size:
        return parallel.tp_group
    return parallel.attn_tp_group


def _reduce_and_redistribute_output_varlen(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    get_parallel().tp_group.reduce_scatterv(
        hidden_states,
        output=local_hidden_states,
        sizes=get_dp_global_num_tokens(),
    )


def _reduce_and_redistribute_output_max_len(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_reduce_scatter_tensor(local_hidden_states, hidden_states)


def _redistribute_output(
    local_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    dp_scatter(local_hidden_states, hidden_states, forward_batch)


def _redistribute_output_from_moe_cp(
    hidden_states: torch.Tensor, rows: List[int]
) -> torch.Tensor:
    moe_cp_rank = get_moe_cp_rank()
    # The allgather was padded to max_tokens_per_rank (equal chunks).
    # Extract this rank's actual (non-padded) tokens from its chunk.
    max_tokens_per_rank = max(rows)
    actual_local_tokens = rows[moe_cp_rank]
    return hidden_states.narrow(
        0, moe_cp_rank * max_tokens_per_rank, actual_local_tokens
    ).contiguous()


def _reduce_and_redistribute_output_step(
    forward_batch: ForwardBatch,
    *,
    leaves_for_reduce_scatter: bool,
    leaves_for_reduce_scatterv: bool,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
    """The reduce-scatter that brings an FFN output gathered over attention
    DP back to this rank's tokens when the FFN leaves its sum to it (see
    StageOutput); None when the FFN reduces the output and only a scatter
    remains."""
    if should_use_dp_reduce_scatterv() and leaves_for_reduce_scatterv:
        return _reduce_and_redistribute_output_varlen
    if (
        leaves_for_reduce_scatter
        and forward_batch.dp_padding_mode.is_max_len()
        and can_use_dp_reduce_scatter()
    ):
        return _reduce_and_redistribute_output_max_len
    return None


def _to_local_tokens(
    step: Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None],
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    local_hidden_states = get_local_dp_buffer(_dp_scatter_group())
    step(local_hidden_states, hidden_states, forward_batch)
    return local_hidden_states


def _all_reduce_then_to_local_tokens(
    group: GroupCoordinator, forward_batch: ForwardBatch, hidden_states: torch.Tensor
) -> torch.Tensor:
    return _to_local_tokens(
        _redistribute_output, forward_batch, group.all_reduce(hidden_states)
    )


class CommunicateSummableTensorPairFn:
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
        is_layer_sparse: bool = False,
    ):
        step = (
            _reduce_and_redistribute_output_step(
                forward_batch,
                leaves_for_reduce_scatter=allow_reduce_scatter,
                # A MoE block leaves its sum to reduce_scatterv whenever it
                # applies (should_skip_post_experts_all_reduce).
                leaves_for_reduce_scatterv=allow_reduce_scatter or is_layer_sparse,
            )
            or _redistribute_output
        )
        return _to_local_tokens(step, forward_batch, hidden_states), residual

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        update: StageUpdate = ADD,
        **kwargs,
    ):
        hidden_states = update.update(hidden_states, residual)
        return _redistribute_from_attn_tp_shards(hidden_states), None

    @staticmethod
    def _onto_residual_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        sums: bool,
        gathers_back: bool,
        update: StageUpdate,
        **kwargs,
    ):
        """Bring the FFN output onto the slice of the rows the residual is on:
        a reduce-scatter that also completes its sum when ``sums``, else this
        rank's slice of the complete output. With ``gathers_back`` it is
        written into the residual there and the full rows are gathered back."""
        if sums:
            hidden_states, _ = tp_reduce_scatter(hidden_states, None, context)
        else:
            hidden_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        if not gathers_back:
            return hidden_states, residual
        local_states = update.update(hidden_states, residual)
        hidden_states = local_states.new_empty(
            local_states.shape[0] * context.tp_size, *local_states.shape[1:]
        )
        get_parallel().tp_group.all_gather_into_tensor(hidden_states, local_states)
        return hidden_states, None

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ):
        assert residual is None, "not yet handled residual!=None"
        return _redistribute_to_attn_tp_shards(hidden_states, context), None

    @staticmethod
    def _take_back_attention_cp_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """DSA and MLA CP: this rank's shard of a complete output gathered in
        equal shards over the attention-CP group."""
        parallel = get_parallel()
        shard = hidden_states.tensor_split(parallel.attn_cp_size)[parallel.attn_cp_rank]
        return shard, residual

    @staticmethod
    def _reduce_scatter_over_cp(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """DSA and MLA CP: sum the FFN output over the attention-CP group and
        keep this rank's shard."""
        return dsa_cp_reduce_scatter_hidden_states(hidden_states), residual

    @staticmethod
    def _take_back_cp_shard(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
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

    @staticmethod
    def _scatter_hidden_states_moe(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        """Scatter MoE output back to TP_ATTN_FULL after MOE_FULL computation.

        After moe_tensor_model_parallel_all_reduce (which runs unconditionally since
        mlp_reduce_scatter=False for this path), all ranks in the moe_cp group hold the
        full MoE result for all cp_per_moe token chunks. We simply slice out this rank's
        CP-local portion.

        If DP>1, further scatter back to the local DP slice.
        """
        # Only scatter back during prefill; decode was never allgathered so no-op.
        # Safe w.r.t. empty tensors: same reasoning as _mlp_input_gather_moe_cp
        # — CP extend always has non-zero tokens per rank, and decode skips this path.
        rows = moe_cp_gathered_rows(forward_batch)
        if rows is not None:
            hidden_states = _redistribute_output_from_moe_cp(hidden_states, rows)

        if context.attn_dp_size > 1:
            hidden_states = _to_local_tokens(
                _redistribute_output, forward_batch, hidden_states
            )

        return hidden_states, residual

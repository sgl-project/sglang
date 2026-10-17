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
"""Megatron-style LayerNorm sequence parallelism (SP, arXiv:2205.05198).

Under pure tensor parallelism the row-parallel ``all_reduce`` is algebraically a
``reduce_scatter`` (g-bar) followed by an ``all_gather`` (g). Splitting it that
way lets the LayerNorm / residual regions run on sequence-sharded activations --
each rank holds 1/tp of the tokens -- which cuts the transient activation memory
of long-context prefill with no extra communication volume (all_reduce and
reduce_scatter+all_gather move the same bytes).

Everything SP lives here so the feature stays decoupled from model code and from
``dp_attention.py``:

  - which models opt in (the Qwen3-dense allowlist) and config validation,
  - the per-forward ``sp_active`` flag (a ForwardFlags bool) read at depth by the
    participant linears and the layer-boundary batch selector,
  - the entry-scatter / exit-gather collectives, and
  - the fused matmul + collective fast-paths for the participant linears.

SP runs for prefill (EXTEND) only and is off by default; with the flag off,
nothing in this module executes.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    get_parallel,
)
from sglang.srt.utils.common import ceil_align

# Architectures whose decoder layers route attention/MLP through
# layer boundaries with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
SP_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    architecture = architectures[0] if architectures else None
    enabled = bool(
        get_parallel().enable_layernorm_sp
        and architecture in SP_SUPPORTED_ARCHITECTURES
    )
    get_flags().sp.enabled = enabled


def layernorm_sp_enabled() -> bool:
    return get_flags().sp.enabled


def runs_sp(forward_mode) -> bool:
    """Whether this forward runs SP: an enabled model, prefill only.

    Code outside the CUDA-graph-captured region must use this and not
    ``get_forward().sp_active``: Python writes made inside that region do not
    re-execute on graph replay, so the flag is stale there.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND


class _SPForwardState:
    """Real (unpadded) token count of the current SP forward.

    An instance attribute, not a ForwardFlags int slot: this is read inside
    torch.compile-traced linear code, where dynamo gives an attribute-source int
    automatic-dynamic, while a dict-slot int recompiles per sequence length.
    """

    num_tokens: int = 0


_sp_state = _SPForwardState()


def set_sp_num_tokens(num_tokens: int) -> None:
    _sp_state.num_tokens = num_tokens


def sp_num_tokens() -> int:
    return _sp_state.num_tokens


def _group_size_rank(group) -> tuple[int, int]:
    if hasattr(group, "world_size"):
        return int(group.world_size), int(group.rank_in_group)
    return dist.get_world_size(group), dist.get_rank(group)


def shard_token_rows(tensor: torch.Tensor, *, group=None) -> torch.Tensor:
    """Pad and locally select this TP rank's contiguous token-row shard."""
    group = group or get_parallel().tp_group
    world_size, rank = _group_size_rank(group)
    padded_rows = ceil_align(tensor.shape[0], world_size)
    if padded_rows != tensor.shape[0]:
        tensor = torch.nn.functional.pad(
            tensor, (0, 0, 0, padded_rows - tensor.shape[0])
        )
    return tensor.tensor_split(world_size, dim=0)[rank].contiguous()


def all_gather_token_rows(
    local_rows: torch.Tensor, *, total_rows: Optional[int] = None, group=None
) -> torch.Tensor:
    """Gather equal token-row shards and optionally trim collective padding."""
    group = group or get_parallel().tp_group
    world_size, _ = _group_size_rank(group)
    output = local_rows.new_empty(
        (local_rows.shape[0] * world_size, *local_rows.shape[1:])
    )
    local_rows = local_rows.contiguous()
    if hasattr(group, "all_gather_into_tensor"):
        group.all_gather_into_tensor(output, local_rows)
    else:
        dist.all_gather_into_tensor(output, local_rows, group=group)
    return output if total_rows is None else output[:total_rows]


def reduce_scatter_token_rows(
    full_partial: torch.Tensor, *, group=None
) -> torch.Tensor:
    """Pad, sum and scatter token rows across a TP group."""
    group = group or get_parallel().tp_group
    world_size, _ = _group_size_rank(group)
    padded_rows = ceil_align(full_partial.shape[0], world_size)
    if padded_rows != full_partial.shape[0]:
        full_partial = torch.nn.functional.pad(
            full_partial, (0, 0, 0, padded_rows - full_partial.shape[0])
        )
    output = full_partial.new_empty(
        (padded_rows // world_size, *full_partial.shape[1:])
    )
    full_partial = full_partial.contiguous()
    if hasattr(group, "reduce_scatter_tensor"):
        group.reduce_scatter_tensor(output, full_partial)
    else:
        dist.reduce_scatter_tensor(output, full_partial, group=group)
    return output


def run_replicated_token_rows(
    hidden_states: torch.Tensor,
    compute: Callable[[torch.Tensor], torch.Tensor],
    *,
    output_is_tp_partial: bool,
    total_rows: Optional[int] = None,
    group=None,
) -> torch.Tensor:
    """Run a full-row-only operation while the surrounding region stays SP.

    Sum partial TP output while scattering it; locally shard complete output.
    """
    group = group or get_parallel().tp_group
    full_rows = all_gather_token_rows(hidden_states, total_rows=total_rows, group=group)
    with get_forward().scoped(sp_active=False):
        full_output = compute(full_rows)
    if output_is_tp_partial:
        return reduce_scatter_token_rows(full_output, group=group)
    return shard_token_rows(full_output, group=group)


# --- entry scatter / exit gather (once per forward, at the boundary) ----------
def sp_entry_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Shard the replicated ``[M, h]`` hidden states along the token dim.

    Pads M up to a multiple of tp_size; the padding rows are dropped by the exit
    gather. The input is replicated across the TP group, so this is a local slice.
    """
    num_tokens = hidden_states.shape[0]
    set_sp_num_tokens(num_tokens)
    group = get_parallel().tp_group
    if group.world_size == 1:
        return hidden_states
    return shard_token_rows(hidden_states, group=group)


def sp_exit_gather(hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """g: all-gather the per-rank shards back to the full sequence along dim 0,
    then narrow to ``num_tokens`` (dropping the entry-scatter padding)."""
    group = get_parallel().tp_group
    if group.world_size == 1:
        return hidden_states[:num_tokens]
    return all_gather_token_rows(
        hidden_states,
        total_rows=num_tokens,
        group=group,
    )


def maybe_exit_gather(
    *,
    hidden_states: torch.Tensor,
    hidden_states_before_norm: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    forward_mode,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Undo the sequence sharding before the LM head, and leave the region.

    No-op unless this forward runs SP. The token count comes from ``input_ids``
    and the predicate from ``runs_sp``, so this stays correct on CUDA graph
    replay, where the writes made inside the captured region do not re-execute.
    """
    if not runs_sp(forward_mode) or input_ids is None:
        return hidden_states, hidden_states_before_norm
    num_tokens = input_ids.shape[0]
    hidden_states = sp_exit_gather(hidden_states, num_tokens=num_tokens)
    if hidden_states_before_norm is not None:
        hidden_states_before_norm = sp_exit_gather(
            hidden_states_before_norm, num_tokens=num_tokens
        )
    get_forward().set("sp_active", False)
    return hidden_states, hidden_states_before_norm


# --- fused matmul + collective fast-paths for the participant linears ---------
# Fused matmul+reduce-scatter (g-bar) and all-gather+matmul (g) overlap the
# collective with the GEMM. Availability is probed once at import; TP groups are
# registered for symmetric memory lazily by the fused ops on first use (the old
# enable_symm_mem_for_group is a deprecated no-op), so we only import the module
# to register the torch.ops.symm_mem namespace the probe checks. NVLink/NVSwitch.
try:
    import torch.distributed._symmetric_memory  # noqa: F401

    _HAS_TORCH_SYMM_MEM_FUSED = hasattr(
        torch.ops.symm_mem, "fused_matmul_reduce_scatter"
    ) and hasattr(torch.ops.symm_mem, "fused_all_gather_matmul")
except Exception:
    _HAS_TORCH_SYMM_MEM_FUSED = False


def sp_fused_matmul_eligible(linear) -> bool:
    """Whether the torch symm_mem fused matmul+collective fast-path applies: the
    ops are available and ``linear`` is unquantized, bias-free, bf16/fp16 (the
    case the fused ops support). Depends only on static layer properties, so the
    decision is identical across TP ranks.
    """
    if not _HAS_TORCH_SYMM_MEM_FUSED or linear.bias is not None:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    return linear.weight.dtype in (torch.bfloat16, torch.float16)


def column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor, bias
) -> torch.Tensor:
    """Megatron SP g for a ColumnParallelLinear participant (qkv / gate_up).

    The input is this rank's sequence shard ``[M_pad/tp, K]``; all-gather it back
    to the full sequence, matmul, and narrow to the real token count (recorded at
    the entry scatter). Uses the fused symm-mem kernel when eligible (all-gather +
    GEMM in one shot), else a plain all-gather + matmul.
    """
    num_tokens = sp_num_tokens()
    if sp_fused_matmul_eligible(linear):
        group_name = get_parallel().tp_group.device_group.group_name
        _, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            input_parallel.contiguous(),
            [linear.weight.t()],
            gather_dim=0,
            group_name=group_name,
        )
        return mm_outputs[0][:num_tokens]
    gathered = sp_exit_gather(input_parallel, num_tokens=num_tokens)
    return linear.quant_method.apply(linear, gathered, bias)


def row_parallel_gbar_matmul(linear, input_: torch.Tensor, bias) -> torch.Tensor:
    """Megatron SP g-bar for a RowParallelLinear participant (o_proj / down).

    Computes ``input_ @ weight.T`` and reduce-scatters (sum) the result across the
    TP group along dim 0, leaving this rank's ``[M_pad/tp, h]`` shard. The token
    dim is padded to a multiple of tp_size (padding rows are zeros, dropped by the
    exit gather). Uses the fused symm-mem kernel when eligible, else matmul + a
    plain reduce-scatter.
    """
    tp_size = linear.tp_group.world_size if linear.tp_group is not None else 1
    x = input_.contiguous()
    num_tokens = x.shape[0]
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - num_tokens))
    if sp_fused_matmul_eligible(linear):
        group_name = get_parallel().tp_group.device_group.group_name
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x,
            linear.weight.t(),
            "sum",
            scatter_dim=0,
            group_name=group_name,
        )
    full = linear.quant_method.apply(linear, x, bias)
    output = full.new_empty((padded // tp_size, *full.shape[1:]))
    get_parallel().tp_group.reduce_scatter_tensor(output, full)
    return output

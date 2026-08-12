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
    participant linears and the ``LayerCommunicator``,
  - the entry-scatter / exit-gather collectives, and
  - the fused matmul + collective fast-paths for the participant linears.

SP runs for prefill (EXTEND) only and is off by default; with the flag off,
nothing in this module executes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_forward, get_server_args
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# Architectures whose decoder layers route attention/MLP through
# ``LayerCommunicator`` with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
_SP_SUPPORTED_ARCHITECTURES = frozenset({"Qwen3ForCausalLM"})


def _model_architecture() -> Optional[str]:
    architectures = get_server_args().get_model_config().hf_config.architectures
    return architectures[0] if architectures else None


def layernorm_sp_enabled() -> bool:
    """True when --enable-layernorm-sp is set and the model is on the allowlist.

    Pure config read; safe at construction and per forward.
    """
    return (
        get_server_args().enable_layernorm_sp
        and _model_architecture() in _SP_SUPPORTED_ARCHITECTURES
    )


def validate_layernorm_sp(
    *,
    architecture: Optional[str],
    tp_size: int,
    enable_dp_attention: bool,
    speculative_algorithm: Optional[str],
) -> None:
    """Fail loud for unsupported / incompatible configs.

    A pure function of the passed config, called from
    ``ServerArgs.check_server_args`` (the caller gates on the flag). It lives at
    that startup point, not in ``LayerCommunicator``, so it also fires for models
    that never build one -- otherwise the flag would be silently ignored there.
    """
    if architecture not in _SP_SUPPORTED_ARCHITECTURES:
        raise ValueError(
            "--enable-layernorm-sp is only supported for "
            f"{sorted(_SP_SUPPORTED_ARCHITECTURES)}; got {architecture}."
        )
    if tp_size <= 1:
        raise ValueError(
            "--enable-layernorm-sp requires tp_size > 1: there is no sequence to "
            "shard across a single TP rank."
        )
    if enable_dp_attention:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with --enable-dp-attention: "
            "SP shards the sequence across the full TP group, which under DP "
            "attention spans data-parallel groups holding different sequences."
        )
    if speculative_algorithm is not None:
        raise ValueError(
            "--enable-layernorm-sp is not compatible with speculative decoding "
            "(EAGLE/EAGLE3): the captured aux hidden states would be "
            "sequence-sharded."
        )


def runs_sp(forward_mode) -> bool:
    """Whether a forward in ``forward_mode`` runs SP: enabled model, prefill only.

    A pure function of config plus the forward mode, so every caller recomputes
    the same answer independently. Callers that sit outside the CUDA-graph-captured
    region (the exit gather) must use this rather than the ``sp_active`` flag:
    Python writes inside the captured region do not re-execute on graph replay, so
    the flag is stale there while this stays correct.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND


def should_activate_sp(forward_batch: ForwardBatch) -> bool:
    """Whether this forward runs SP (see ``runs_sp``)."""
    return runs_sp(forward_batch.forward_mode)


# --- per-forward active flag (read at depth by linears / communicator) --------
def is_sp_active() -> bool:
    return get_forward().sp_active


def set_sp_active(active: bool) -> None:
    """Set/clear the SP region flag.

    Set fresh at the first decoder layer of an SP forward and cleared at the exit
    gather, so it is self-correcting each forward (a crash mid-loop cannot leak
    into the next). A bool (<=2 values) is torch.compile-safe as a plain slot;
    see ``ForwardFlags._GRAPH_VISIBLE``.
    """
    get_forward().set("sp_active", active)


class _SPForwardState:
    """Real (unpadded) token count of the current SP forward.

    Recorded at the entry scatter, read by the participant column linears (to
    narrow after the g all-gather, before rotary/attention) and by the exit
    gather. Held as an instance attribute -- deliberately not a ForwardFlags dict
    slot or contextvar -- because it is read inside torch.compile-traced linear
    code: dynamo gives attribute-source ints automatic-dynamic (one signature),
    whereas a dict-slot int recompiles per distinct sequence length and a
    contextvar is untraceable. Process-global is safe: SP is prefill-only and a
    forward is single-threaded.
    """

    num_tokens: int = 0


_sp_state = _SPForwardState()


def set_sp_num_tokens(num_tokens: int) -> None:
    _sp_state.num_tokens = num_tokens


def sp_num_tokens() -> int:
    return _sp_state.num_tokens


# --- entry scatter / exit gather (once per forward, at the boundary) ----------
def sp_entry_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Shard the replicated ``[M, h]`` hidden states along the token dim.

    Pads M up to a multiple of tp_size (padding rows are dropped by the exit
    gather) and slices this rank's equal shard. The post-embedding hidden states
    are replicated across the TP group, so this is a local slice, no comm.
    Records the real token count M for the per-layer narrows and the exit gather.
    """
    num_tokens = hidden_states.shape[0]
    set_sp_num_tokens(num_tokens)
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        hidden_states = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, padded - num_tokens)
        )
    return hidden_states.tensor_split(tp_size)[tp_group.rank_in_group].contiguous()


def sp_exit_gather(hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """g: all-gather the per-rank shards back to the full sequence along dim 0,
    then narrow to ``num_tokens`` (dropping the entry-scatter padding).

    ``num_tokens`` is passed in rather than read from module state so callers
    outside the CUDA-graph-captured region stay correct on replay.
    """
    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states[:num_tokens]
    hidden_states = hidden_states.contiguous()
    output = hidden_states.new_empty(
        (hidden_states.shape[0] * tp_size, *hidden_states.shape[1:])
    )
    tp_group.all_gather_into_tensor(output, hidden_states)
    return output[:num_tokens]


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
        group_name = get_tp_group().device_group.group_name
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
    tp_size = linear.tp_size
    x = input_.contiguous()
    num_tokens = x.shape[0]
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - num_tokens))
    if sp_fused_matmul_eligible(linear):
        group_name = get_tp_group().device_group.group_name
        return torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x,
            linear.weight.t(),
            "sum",
            scatter_dim=0,
            group_name=group_name,
        )
    full = linear.quant_method.apply(linear, x, bias)
    output = full.new_empty((padded // tp_size, *full.shape[1:]))
    get_tp_group().reduce_scatter_tensor(output, full)
    return output

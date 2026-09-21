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

import logging
from typing import Optional

import torch

from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    get_parallel,
)
from sglang.srt.utils.common import ceil_align, is_npu

logger = logging.getLogger(__name__)

# Architectures whose decoder layers route attention/MLP through
# ``LayerCommunicator`` with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
SP_SUPPORTED_ARCHITECTURES = frozenset({"Qwen3ForCausalLM"})


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    enabled = bool(
        get_parallel().enable_layernorm_sp
        and architectures
        and architectures[0] in SP_SUPPORTED_ARCHITECTURES
    )
    get_flags().sp.enabled = enabled
    if enabled:
        logger.info(
            "LayerNorm sequence parallelism (SP) ENABLED for architecture=%s; "
            "prefill uses reduce_scatter+all_gather via the SP region.",
            architectures[0],
        )


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


# --- entry scatter / exit gather (once per forward, at the boundary) ----------
def sp_entry_scatter(hidden_states: torch.Tensor) -> torch.Tensor:
    """Shard the replicated ``[M, h]`` hidden states along the token dim.

    Pads M up to a multiple of tp_size; the padding rows are dropped by the exit
    gather. The input is replicated across the TP group, so this is a local slice.
    """
    num_tokens = hidden_states.shape[0]
    set_sp_num_tokens(num_tokens)
    tp_group = get_parallel().tp_group
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
    then narrow to ``num_tokens`` (dropping the entry-scatter padding)."""
    tp_group = get_parallel().tp_group
    tp_size = tp_group.world_size
    if tp_size == 1:
        return hidden_states[:num_tokens]
    hidden_states = hidden_states.contiguous()
    output = hidden_states.new_empty(
        (hidden_states.shape[0] * tp_size, *hidden_states.shape[1:])
    )
    tp_group.all_gather_into_tensor(output, hidden_states)
    return output[:num_tokens]


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


# --- NPU (Ascend) MC2 fused matmul+collective fast-path -----------------------
# torch_npu mirrors the CUDA symm_mem fused ops with the CANN MC2 (Matmul +
# Collective-Communication) kernels (CANN >= 8.0.RC2):
#   npu_mm_reduce_scatter_base == ReduceScatter(x1 @ x2)   [row-parallel g-bar]
#   npu_all_gather_base_mm     == allgather(x1) @ x2       [column-parallel g]
# They require fp16/bf16, 2-D inputs and the Hccl backend handle (``hcom``).
# Availability is probed once at import; on non-NPU builds _HAS_NPU_MC2_FUSED is
# left False so the fallback collective path is used.
_HAS_NPU_MC2_FUSED = False
_npu_mm_reduce_scatter_base = None
_npu_all_gather_base_mm = None
if is_npu():
    try:
        import torch_npu  # noqa: F401

        _npu_mm_reduce_scatter_base = getattr(
            torch_npu, "npu_mm_reduce_scatter_base", None
        )
        _npu_all_gather_base_mm = getattr(torch_npu, "npu_all_gather_base_mm", None)
        _HAS_NPU_MC2_FUSED = (
            _npu_mm_reduce_scatter_base is not None
            and _npu_all_gather_base_mm is not None
        )
    except Exception:
        _HAS_NPU_MC2_FUSED = False


def _npu_hccl_comm_name() -> str:
    """HCCL communicator handle (``hcom``) of the TP group, consumed by the MC2
    fused ops. ``device_group`` is the raw torch ProcessGroup; the Hccl backend
    exposes ``get_hccl_comm_name`` for the global rank."""
    tp_group = get_parallel().tp_group
    backend = tp_group.device_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(tp_group.rank)


def sp_fused_matmul_eligible(linear) -> bool:
    """Whether the fused matmul+collective fast-path applies: the relevant ops
    are available for the current device and ``linear`` is unquantized,
    bias-free, bf16/fp16 (the case the fused ops support). Depends only on
    static layer properties, so the decision is identical across TP ranks.
    """
    if linear.bias is not None:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    if linear.weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if is_npu():
        # CANN MC2 fused ops are 2-D-only (matmul over 2-D [m, k] x [k, n]).
        return _HAS_NPU_MC2_FUSED and linear.weight.ndim == 2
    return _HAS_TORCH_SYMM_MEM_FUSED


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
        if is_npu():
            # CANN MC2: allgather(input shard) @ weight.t(), then narrow.
            try:
                mm_out, _ = _npu_all_gather_base_mm(
                    input_parallel.contiguous(),
                    linear.weight.t().contiguous(),
                    _npu_hccl_comm_name(),
                    linear.tp_size,
                )
                return mm_out[:num_tokens]
            except Exception:
                pass  # fall through to the plain all-gather + matmul below.
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
    tp_size = linear.tp_size
    x = input_.contiguous()
    num_tokens = x.shape[0]
    padded = ceil_align(num_tokens, tp_size)
    if padded != num_tokens:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - num_tokens))
    if sp_fused_matmul_eligible(linear):
        if is_npu():
            # CANN MC2: ReduceScatter(x @ weight.t()). x's padded token dim is
            # already a multiple of tp_size (required by the fused op).
            try:
                return _npu_mm_reduce_scatter_base(
                    x,
                    linear.weight.t().contiguous(),
                    _npu_hccl_comm_name(),
                    tp_size,
                    reduce_op="sum",
                )
            except Exception:
                pass  # fall through to matmul + plain reduce-scatter below.
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

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

from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import (
    get_exec,
    get_flags,
    get_forward,
    get_parallel,
    get_resources,
)
from sglang.srt.utils import is_cuda, is_npu
from sglang.srt.utils.common import ceil_align
from sglang.srt.utils.device import get_device

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

# Architectures whose decoder layers route attention/MLP through
# layer boundaries with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
MUSE_SP_ARCHITECTURES = frozenset({"MuseSparkForConditionalGeneration"})
KIMI_K3_SP_ARCHITECTURES = frozenset(
    {"KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration"}
)
SP_SUPPORTED_ARCHITECTURES = (
    MUSE_SP_ARCHITECTURES
    | KIMI_K3_SP_ARCHITECTURES
    | frozenset(
        {
            "Qwen3ForCausalLM",
            "Qwen4ExpForCausalLM",
            "Qwen4ExpForConditionalGeneration",
        }
    )
)
# Enabled where validated, not by --enable-layernorm-sp: extends of at least
# MINIMAX_M3_SP_MIN_TOKENS only, with the TP-sharded MoE run on the gathered sequence.
MINIMAX_M3_SP_ARCHITECTURES = frozenset(
    {"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
)
MINIMAX_M3_SP_MIN_TOKENS = 4096


def is_muse_spark_architecture(architecture: Optional[str]) -> bool:
    return architecture in MUSE_SP_ARCHITECTURES


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    architecture = architectures[0] if architectures else None
    m3_min_tokens = _minimax_m3_sp_min_tokens(architecture)
    enabled = bool(
        (
            get_parallel().enable_layernorm_sp
            and architecture in SP_SUPPORTED_ARCHITECTURES
        )
        or m3_min_tokens > 0
    )
    get_flags().sp.enabled = enabled
    get_flags().sp.min_tokens = m3_min_tokens
    if m3_min_tokens > 0:
        _init_push_comm(
            hidden=model_config.hidden_size,
            min_tokens=m3_min_tokens,
        )


def _minimax_m3_sp_min_tokens(architecture: Optional[str]) -> int:
    """The SP threshold for an M3 model, or 0 for off: on only for the validated
    CUDA TP2 setups (TP MoE, or MegaMoE over the TP ranks)."""
    if architecture not in MINIMAX_M3_SP_ARCHITECTURES:
        return 0
    if _minimax_m3_sp_unsupported(MINIMAX_M3_SP_MIN_TOKENS) is not None:
        return 0
    return MINIMAX_M3_SP_MIN_TOKENS


def _minimax_m3_sp_unsupported(min_tokens: int) -> Optional[str]:
    from sglang.srt.layers.layer_boundary import FUSE_ALLREDUCE_MAX_BATCH_SIZE
    from sglang.srt.model_executor.cuda_graph_config import Backend

    # The push comm is CUDA symmetric memory, and the fallback gathers bf16 tensors.
    if not is_cuda():
        return "requires CUDA"
    parallel = get_parallel()
    # The published config, not get_moe_a2a_backend(): the MoE flags are
    # seeded after this runs and would read as the "none" default here.
    a2a_backend = get_exec().moe.moe_a2a_backend
    a2a_supported = a2a_backend == "none" or (
        a2a_backend == "megamoe"
        and sorted(parallel.moe_ep_group.ranks) == sorted(parallel.tp_group.ranks)
    )
    if not (
        parallel.tp_size > 1
        and parallel.attn_tp_size == parallel.tp_size
        and parallel.attn_cp_size == 1
        and parallel.pp_size == 1
        and parallel.moe_dense_tp_size != 1
        and a2a_supported
    ):
        return (
            "requires TP > 1 without attention DP/CP or PP, and no a2a MoE "
            "backend other than MegaMoE over the TP ranks"
        )
    # Above two ranks the reduce-scatter sums in a different order than the
    # all-reduce; TP2 is the setup measured bitwise-identical.
    if parallel.tp_size != 2:
        return "requires TP2"
    # A breakable graph captures the whole forward, logits included, so each
    # bucket decides SP once from its padded size; other captured backends do not.
    backend = get_exec().graph.cuda_graph_config.prefill.backend
    if (
        backend not in (Backend.DISABLED, Backend.BREAKABLE)
        and _max_prefill_bucket() >= min_tokens
    ):
        return (
            f"must exceed the largest {backend} prefill CUDA graph bucket "
            f"({_max_prefill_bucket()})"
        )
    comm = get_exec().comm
    # These replace the replicated path's bf16 all-reduce + norm below this size,
    # so SP there would not reproduce its numerics.
    if min_tokens <= FUSE_ALLREDUCE_MAX_BATCH_SIZE and (
        comm.flashinfer_allreduce_fusion_backend is not None
        or comm.enable_quant_communications
    ):
        return (
            f"must exceed {FUSE_ALLREDUCE_MAX_BATCH_SIZE} with all-reduce fusion "
            "or quantized communications enabled"
        )
    return None


def _max_prefill_bucket() -> int:
    return max(get_exec().graph.cuda_graph_config.prefill.bs or [0])


def layernorm_sp_enabled() -> bool:
    return get_flags().sp.enabled


def runs_sp(forward_mode, num_tokens: Optional[int] = None) -> bool:
    """Whether this forward runs SP: an enabled model, prefill only, and more
    than --layernorm-sp-min-tokens tokens (0, the default, admits every pass),
    or at least ``flags.sp.min_tokens`` when the model sets its own threshold.

    ``num_tokens`` is the pass's token count, before the entry scatter. Every
    in-tree caller passes it, so the entry scatter, the layer-boundary entry and
    the exit gathers take the same decision for one pass. In a captured prefill
    graph the entry decides on the bucket size while the exit gather outside
    the graph sees the real count; they agree because the threshold is a
    captured bucket (checked at resolution, ``handle_layernorm_sp``). A caller
    that asks only about decode / verify passes may omit the count: those
    never run SP. An extend pass without it raises.

    Code outside the CUDA-graph-captured region must use this and not
    ``get_forward().sp_active``: Python writes made inside that region do not
    re-execute on graph replay, so the flag is stale there.
    """
    from sglang.srt.model_executor.forward_mode import ForwardMode

    if not (layernorm_sp_enabled() and forward_mode == ForwardMode.EXTEND):
        return False
    if num_tokens is None:
        raise ValueError(
            "runs_sp needs num_tokens (the pass's token count) for an extend pass: "
            "without it the caller skips --layernorm-sp-min-tokens and, on a pass "
            "at or under the threshold, disagrees with the entry and exit gathers."
        )
    model_min_tokens = get_flags().sp.min_tokens
    if model_min_tokens > 0:
        return num_tokens >= model_min_tokens
    return num_tokens > get_parallel().layernorm_sp_min_tokens


def stack_rows(forward_mode, num_tokens: int) -> int:
    """Rows a rank holds inside the layer stack: its padded shard of an SP pass,
    else every token."""
    if not runs_sp(forward_mode, num_tokens):
        return num_tokens
    world_size, _ = _group_size_rank(get_parallel().tp_group)
    return ceil_align(num_tokens, world_size) // world_size


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
    push_comm = get_resources().layernorm_sp_push_comm
    if push_comm is not None:
        push_comm.reset()
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
    num_tokens = _sp_exit_tokens(input_ids, forward_mode)
    if num_tokens is None:
        return hidden_states, hidden_states_before_norm
    hidden_states = sp_exit_gather(hidden_states, num_tokens=num_tokens)
    if hidden_states_before_norm is not None:
        hidden_states_before_norm = sp_exit_gather(
            hidden_states_before_norm, num_tokens=num_tokens
        )
    get_forward().set("sp_active", False)
    return hidden_states, hidden_states_before_norm


def maybe_gather_aux_hidden_states(aux_hidden_states, *, input_ids, forward_mode):
    """Gather the sequence-sharded aux captures back to the replicated token
    order the draft consumes, packed into one tensor; no-op outside SP."""
    from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states

    num_tokens = _sp_exit_tokens(input_ids, forward_mode)
    if aux_hidden_states is None or num_tokens is None:
        return aux_hidden_states
    return sp_exit_gather(
        pack_aux_hidden_states(aux_hidden_states), num_tokens=num_tokens
    )


def _sp_exit_tokens(input_ids, forward_mode) -> Optional[int]:
    """The token count to gather back to when this forward ran SP, else None."""
    if input_ids is None or not runs_sp(forward_mode, input_ids.shape[0]):
        return None
    return input_ids.shape[0]


class _PushComm:
    """TP2 reduce-scatter / all-gather of ``[tokens, hidden]`` rows through
    symmetric memory: each rank stores the peer's rows straight into the peer's
    buffer, then one GPU barrier. Two generations per op kind; every op's
    barrier orders a generation's reuse two ops later after its last read.
    """

    def __init__(
        self,
        *,
        group,
        max_tokens: int,
        hidden: int,
        dtype,
        device,
        gather_mxfp8: bool = False,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = group.rank_in_group
        self.peer = 1 - self.rank
        name = group.device_group.group_name

        def pair(shape, dtype=dtype):
            local = symm_mem.empty(shape, dtype=dtype, device=device)
            handle = symm_mem.rendezvous(local, name)
            return local, handle.get_buffer(self.peer, shape, dtype), handle

        self.max_tokens = max_tokens
        self.recv = [pair((max_tokens // 2, hidden)) for _ in range(2)]
        self.gather = [pair((max_tokens, hidden)) for _ in range(2)]
        self.partials = [pair((max_tokens, hidden)) for _ in range(2)]
        # MXFP8 gathers: [tokens, hidden] e4m3 bytes, and the 128x4-swizzled
        # UE8M0 scales, hidden / 32 bytes per row.
        self.gather_mxfp8 = None
        if gather_mxfp8:
            self.gather_mxfp8 = [
                (
                    pair((max_tokens, hidden), torch.uint8),
                    pair((max_tokens * (hidden // 32),), torch.uint8),
                )
                for _ in range(2)
            ]
        self.reset()

    def reset(self) -> None:
        self.recv_gen = 0
        self.gather_gen = 0
        self.partial_gen = 0

    def partial_sink(self, rows: int) -> torch.Tensor:
        """Where the next stage compute may write its partial, letting the
        reduce-scatter read the peer's rows in place (``pull``)."""
        return self.partials[self.partial_gen][0][:rows]

    def pull_reduce_scatter(self, rows: int) -> torch.Tensor:
        local, peer, handle = self.partials[self.partial_gen]
        self.partial_gen ^= 1
        own = slice(self.rank * rows // 2, (self.rank + 1) * rows // 2)
        handle.barrier(channel=0)
        return torch.add(local[own], peer[own])

    def reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        rows = partial.shape[0] // 2
        local, peer, handle = self.recv[self.recv_gen]
        self.recv_gen ^= 1
        peer[:rows].copy_(partial[self.peer * rows : (self.peer + 1) * rows])
        handle.barrier(channel=0)
        return torch.add(
            partial[self.rank * rows : (self.rank + 1) * rows], local[:rows]
        )

    def all_gather(self, shard: torch.Tensor) -> torch.Tensor:
        rows = shard.shape[0]
        local, peer, handle = self.gather[self.gather_gen]
        self.gather_gen ^= 1
        own = slice(self.rank * rows, (self.rank + 1) * rows)
        local[own].copy_(shard)
        peer[own].copy_(shard)
        handle.barrier(channel=0)
        return local[: 2 * rows]

    def all_gather_mxfp8(
        self, data: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather a shard's swizzled MXFP8 rows; the shard must hold a
        multiple of 128 rows, the unit the scale layout concatenates on."""
        rows, sf_bytes = data.shape[0], scales.shape[0]
        (q_local, q_peer, handle), (s_local, s_peer, _) = self.gather_mxfp8[
            self.gather_gen
        ]
        self.gather_gen ^= 1
        own = slice(self.rank * rows, (self.rank + 1) * rows)
        own_sf = slice(self.rank * sf_bytes, (self.rank + 1) * sf_bytes)
        data = data.view(torch.uint8)
        q_local[own].copy_(data)
        q_peer[own].copy_(data)
        s_local[own_sf].copy_(scales)
        s_peer[own_sf].copy_(scales)
        handle.barrier(channel=0)
        return (
            q_local[: 2 * rows].view(torch.float8_e4m3fn),
            s_local[: 2 * sf_bytes],
        )


def take_partial_sink(like: torch.Tensor) -> Optional[torch.Tensor]:
    """The symmetric buffer an SP stage compute should write its final partial
    into (``torch.add(..., out=sink)``), or None to allocate as usual."""
    forward = get_forward()
    sink = forward.sp_partial_sink
    forward.set("sp_partial_sink", None)
    if sink is None or sink.shape != like.shape or sink.dtype != like.dtype:
        return None
    return sink


def _init_push_comm(*, hidden: int, min_tokens: int) -> None:
    """Allocate the TP2 symmetric-memory comm at startup, before the KV pool is
    sized, at the largest prefill the server admits; skipped when that is below
    the SP threshold."""
    from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
        recommended_max_tokens,
    )

    tp_group = get_parallel().tp_group
    max_tokens = ceil_align(
        recommended_max_tokens(include_prefill=True, floor=_max_prefill_bucket()), 2
    )
    if tp_group.world_size != 2 or max_tokens < min_tokens:
        return
    get_resources().layernorm_sp_push_comm = _PushComm(
        group=tp_group,
        max_tokens=max_tokens,
        hidden=hidden,
        dtype=torch.bfloat16,
        device=torch.device("cuda", torch.cuda.current_device()),
        gather_mxfp8=hidden % 128 == 0,
    )


def sp_gathers_mxfp8(shard_rows: int) -> bool:
    """Whether an SP stage may read its ``shard_rows``-row input as swizzled
    MXFP8 and have ``sp_gathered_compute`` gather it in that form."""
    comm = get_resources().layernorm_sp_push_comm
    num_tokens = sp_num_tokens()
    return (
        comm is not None
        and comm.gather_mxfp8 is not None
        and num_tokens <= comm.max_tokens
        and shard_rows * 2 == num_tokens
        and shard_rows % 128 == 0
    )


def sp_gathered_compute(
    compute, hidden_states: Union[torch.Tensor, Mxfp8SwizzledInput]
) -> torch.Tensor:
    """Run a TP-partial stage compute on the whole sequence: all-gather this
    rank's shard, run ``compute`` outside the SP region (participant linears
    take their replicated path), then reduce-scatter its partial output.

    At TP2 the two-way sum rounds alike in either order, so the result is
    bitwise the replicated path's all-reduce, sliced to this rank's rows.
    """
    from sglang.srt.layers.layer_boundary.ops import tp_reduce_scatter
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput

    num_tokens = sp_num_tokens()
    padded = ceil_align(num_tokens, get_parallel().tp_size)
    comm = get_resources().layernorm_sp_push_comm
    if isinstance(hidden_states, Mxfp8SwizzledInput):
        # Read that way only when sp_gathers_mxfp8 held for this shard.
        full = Mxfp8SwizzledInput(
            *comm.all_gather_mxfp8(hidden_states.data, hidden_states.scales)
        )
    elif (
        comm is None
        or padded > comm.max_tokens
        or hidden_states.dtype != torch.bfloat16
    ):
        comm = None
        full = sp_exit_gather(hidden_states, num_tokens=num_tokens)
    else:
        full = comm.all_gather(hidden_states)[:num_tokens]
    sink = (
        comm.partial_sink(num_tokens)
        if comm is not None and padded == num_tokens
        else None
    )
    with get_forward().scoped(sp_active=False, sp_partial_sink=sink):
        partial = compute(full)
    if sink is not None and partial.data_ptr() == sink.data_ptr():
        return comm.pull_reduce_scatter(num_tokens)
    if padded != num_tokens:
        partial = torch.nn.functional.pad(partial, (0, 0, 0, padded - num_tokens))
    if comm is not None:
        return comm.reduce_scatter(partial)
    return tp_reduce_scatter(partial, None)[0]


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
# CANN >= 8.0.RC2 fused ops: npu_mm_reduce_scatter_base == ReduceScatter(x1 @ x2),
# npu_all_gather_base_mm == allgather(x1) @ x2; probed once at import.
_HAS_NPU_MC2_FUSED = False
_npu_mm_reduce_scatter_base = None
_npu_all_gather_base_mm = None
if is_npu():
    try:
        import torch_npu  # noqa: F401

        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        # arch35 (Ascend 950) MC2 fused ops accumulate at lower GEMM precision
        # and regress model accuracy; keep the plain collective fallback there.
        _npu_mm_reduce_scatter_base = getattr(
            torch_npu, "npu_mm_reduce_scatter_base", None
        )
        _npu_all_gather_base_mm = getattr(torch_npu, "npu_all_gather_base_mm", None)
        _HAS_NPU_MC2_FUSED = (
            _npu_mm_reduce_scatter_base is not None
            and _npu_all_gather_base_mm is not None
            and not is_npu_arch35()
        )
    except Exception:
        _HAS_NPU_MC2_FUSED = False


def _npu_hccl_comm_name() -> str:
    """HCCL handle (``hcom``) of the TP group, consumed by the MC2 fused ops."""
    tp_group = get_parallel().tp_group
    backend = tp_group.device_group._get_backend(torch.device(get_device()))
    return backend.get_hccl_comm_name(tp_group.rank)


def sp_fused_matmul_eligible(linear) -> bool:
    """Whether the selected fused matmul+collective path applies.

    Muse Spark selects CUTLASS; other models keep the Torch symmetric-memory
    path. Both require an unquantized, bias-free bf16/fp16 linear. The decision
    depends only on static layer properties and is identical across TP ranks.
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
    if getattr(linear, "_sp_use_cutlass", False):
        return True
    return _HAS_TORCH_SYMM_MEM_FUSED


def column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor, bias
) -> torch.Tensor:
    """Megatron SP g for a ColumnParallelLinear participant (qkv / gate_up).

    The input is this rank's sequence shard ``[M_pad/tp, K]``; all-gather it back
    to the full sequence, matmul, and narrow to the real token count (recorded at
    the entry scatter). Eligible Muse Spark layers use one CUTLASS GEMM with
    tile-gated all-gather; other models retain their existing implementation.
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
                    linear.tp_group.world_size,
                )
                return mm_out[:num_tokens]
            except Exception:
                # CANN MC2 unavailable for this shape/dtype: fall through to the
                # plain all-gather + matmul below (symm_mem is CUDA-only).
                pass
        elif getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.communication.sp_collective_gemm import (
                all_gather_matmul,
            )
            from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
                sp_collective_gemm_workspace,
            )

            return all_gather_matmul(
                input_parallel.contiguous(),
                linear.weight,
                get_parallel().tp_group.device_group,
                workspace=sp_collective_gemm_workspace,
            )[:num_tokens]
        else:
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
    exit gather). Eligible Muse Spark layers use one CUTLASS GEMM with a
    multicast reduce-scatter epilogue; other models retain their existing path.
    """
    tp_size = linear.tp_group.world_size if linear.tp_group is not None else 1
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
                # CANN MC2 unavailable for this shape/dtype: fall through to
                # matmul + plain reduce-scatter below (symm_mem is CUDA-only).
                pass
        elif getattr(linear, "_sp_use_cutlass", False):
            from sglang.kernels.ops.communication.sp_collective_gemm import (
                matmul_reduce_scatter,
            )
            from sglang.srt.layers.communication.sp_collective_gemm_workspace import (
                sp_collective_gemm_workspace,
            )

            return matmul_reduce_scatter(
                x,
                linear.weight,
                get_parallel().tp_group.device_group,
                workspace=sp_collective_gemm_workspace,
            )
        else:
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

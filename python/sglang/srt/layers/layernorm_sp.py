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

import functools
import logging
import re
from collections import Counter
from typing import Callable, Optional

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    get_parallel,
)
from sglang.srt.utils.common import ceil_align

logger = logging.getLogger(__name__)

# Architectures whose decoder layers route attention/MLP through
# layer boundaries with the standard participant linears, and for which SP
# has been validated. Other models reject --enable-layernorm-sp at construction.
# The mechanism is generic; extend the allowlist as families are validated.
# LlamaForCausalLM uses the same layer-boundary / residual_batch wiring as
# Qwen3 and is the target of the Cake ``sp_all_gather_matmul`` route
# (Llama-3.1-70B: K = 8192, packed-QKV N = 1280 at TP8); its stock SP run is
# still to be validated on GPU alongside that route.
SP_SUPPORTED_ARCHITECTURES = frozenset({"Qwen3ForCausalLM"})
# Llama-3.1 dense models become SP participants only when the Cake
# all-gather+matmul route is selected (opt-in); the stock SP allowlist is
# unchanged otherwise.
_CAKE_SP_EXTRA_ARCHITECTURES = frozenset({"LlamaForCausalLM"})


def sp_supported_architectures() -> frozenset:
    """Architectures accepted by ``--enable-layernorm-sp`` for this process."""
    if cake_route_enabled("sp_all_gather_matmul"):
        return SP_SUPPORTED_ARCHITECTURES | _CAKE_SP_EXTRA_ARCHITECTURES
    return SP_SUPPORTED_ARCHITECTURES


def initialize_layernorm_sp(*, model_config) -> None:
    """Materialize ``flags.sp.enabled``; runs once per worker after distributed
    setup, alongside ``initialize_dp_attention``."""
    architectures = model_config.hf_config.architectures
    get_flags().sp.enabled = bool(
        get_parallel().enable_layernorm_sp
        and architectures
        and architectures[0] in sp_supported_architectures()
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


def sp_fused_matmul_eligible(linear) -> bool:
    """Whether the torch symm_mem fused matmul+collective fast-path applies: the
    ops are available and ``linear`` is unquantized, bias-free, bf16/fp16 (the
    case the fused ops support). Depends only on static layer properties, so the
    decision is identical across TP ranks. Once the Cake route has selected the
    NVSHMEM symmetric-memory backend (``_cake_sp_symmetric_backend_ready``) the
    torch fused ops are off: their allocator rejects a process group under that
    backend, so the non-admitted participants take the plain collective + GEMM.
    """
    if not _HAS_TORCH_SYMM_MEM_FUSED or linear.bias is not None:
        return False
    if _cake_sp_symm_backend:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    return linear.weight.dtype in (torch.bfloat16, torch.float16)


# --- Cake all-gather + matmul route (SGLANG_CAKE_ROUTES=sp_all_gather_matmul) --
# ``column_parallel_g_matmul`` is the engine's all-gather + GEMM site. With the
# route selected, an unquantized bias-free BF16/FP16 participant whose shard
# the adapter admits (K = 8192, any local row count, N % 256 == 0: Llama-3.1-70B
# qkv N = 1280 and gate_up N = 7168 at TP8, N = 2560 / 14336 at TP4) runs
# FlashInfer's Cake kernel through one capacity-bound prepared launcher per
# participant (``communication.prepare_all_gather_matmul`` with ``max_rows``;
# the functional ``communication.all_gather_matmul`` is the fallback when the
# launcher cannot be prepared). Everything else keeps the stock path unchanged.
#
# The kernel consumes the engine's ``[N, K]`` parameter through its ``weight.t()``
# view (a supported weight layout; no transposed copy) and pads the row count
# inside the kernel, so the launcher is prepared once per participant for the
# engine's prefill chunk cap spread over the TP ranks and re-prepared only when
# a larger shard arrives (every rank sees the same shard rows) or the parameter
# storage changes (weight reload). SP is prefill-only, so nothing here runs
# inside CUDA-graph capture; the guard below keeps that invariant explicit.

CAKE_ROUTE_SP_ALL_GATHER_MATMUL = "sp_all_gather_matmul"
_CAKE_LOG_PREFIX = "[cake-route]"

_cake_sp_logged: set[tuple[str, str]] = set()
_cake_sp_rejected: set[tuple] = set()
# id(linear) -> (weight storage key, capacity rows, prepared launcher)
_cake_sp_launchers: dict[
    int, tuple[tuple, int, Callable[[torch.Tensor], torch.Tensor]]
] = {}
# torch symmetric-memory backend selection for the route: None = not attempted,
# True = NVSHMEM selected (FlashInfer's Cake backend requires it), False = unavailable.
_cake_sp_symm_backend: Optional[bool] = None
# (event, N) -> calls; events: taken-prepared, taken, fallback, prepare. Read by
# ``cake_sp_call_counts`` (tests, offline tools); with ``SGLANG_CAKE_DEBUG`` every
# call is also logged, so admission can be counted from the engine log.
_cake_sp_calls: Counter = Counter()


def _cake_sp_reason_kind(detail: str) -> str:
    """Digit-normalised prefix of a fallback detail (the text before the tensor
    dump), so one line is emitted per distinct reason, not per shape."""
    return re.sub(r"\d+", "N", detail.split(":", 1)[0])[:64]


def _log_cake_sp_once(event: str, detail: str) -> None:
    key = (event, _cake_sp_reason_kind(detail) if event == "fallback" else "")
    if key in _cake_sp_logged:
        return
    _cake_sp_logged.add(key)
    if event.startswith("taken"):
        logger.info(
            "%s %s: Cake kernel selected (%s)",
            _CAKE_LOG_PREFIX,
            CAKE_ROUTE_SP_ALL_GATHER_MATMUL,
            detail,
        )
    else:
        logger.info(
            "%s %s: fallback to stock all-gather + matmul (%s)",
            _CAKE_LOG_PREFIX,
            CAKE_ROUTE_SP_ALL_GATHER_MATMUL,
            detail,
        )


def _count_cake_sp_call(
    event: str, rows: int, n: int, world_size: int, **extra
) -> None:
    _cake_sp_calls[(event, int(n))] += 1
    if envs.SGLANG_CAKE_DEBUG.get():
        logger.info(
            "%s %s call: event=%s rows=%d N=%d world_size=%d%s",
            _CAKE_LOG_PREFIX,
            CAKE_ROUTE_SP_ALL_GATHER_MATMUL,
            event,
            rows,
            n,
            world_size,
            "".join(f" {k}={v}" for k, v in extra.items()),
        )


def _cake_sp_symmetric_backend_ready(device: torch.device) -> bool:
    """Select torch's NVSHMEM symmetric-memory backend once per process.

    FlashInfer's Cake all-gather matmul allocates its scratch and flags through
    ``torch.distributed._symmetric_memory`` and refuses any other backend. The
    selection happens on the first eligible call (eager prefill, before any
    symmetric allocation of this process); when it fails the route falls back
    for the rest of the process with one logged reason.
    """
    global _cake_sp_symm_backend
    if _cake_sp_symm_backend is None:
        try:
            import torch.distributed._symmetric_memory as symm_mem

            if str(symm_mem.get_backend(device) or "").upper() != "NVSHMEM":
                symm_mem.set_backend("NVSHMEM")
            _cake_sp_symm_backend = (
                str(symm_mem.get_backend(device) or "").upper() == "NVSHMEM"
            )
            if not _cake_sp_symm_backend:
                _log_cake_sp_once(
                    "fallback",
                    "NVSHMEM symmetric-memory backend not selected: "
                    f"get_backend returned {symm_mem.get_backend(device)!r}",
                )
        except Exception as exc:  # backend missing in this torch / no NVSHMEM
            _cake_sp_symm_backend = False
            _log_cake_sp_once(
                "fallback", f"NVSHMEM symmetric-memory backend unavailable ({exc!r})"
            )
    return _cake_sp_symm_backend


def cake_sp_call_counts() -> dict[tuple[str, int], int]:
    """Per-(event, N) call counts of the SP all-gather matmul route so far."""
    return dict(_cake_sp_calls)


def reset_cake_sp_state_for_tests() -> None:
    global _cake_sp_symm_backend
    _cake_sp_logged.clear()
    _cake_sp_rejected.clear()
    _cake_sp_launchers.clear()
    _cake_sp_calls.clear()
    _cake_sp_symm_backend = None


@functools.lru_cache(maxsize=None)
def _cake_sp_kernels() -> tuple[Callable, Callable, Callable, Callable]:
    """Lazy (supports_ag, supports_prepare, ag, prepare) for the Cake AG-matmul."""
    from sglang.kernels.cake_kernels.communication import (
        supports_all_gather_matmul,
        supports_prepare_all_gather_matmul,
    )
    from sglang.kernels.ops.communication.cake import (
        cake_all_gather_matmul,
        cake_prepare_all_gather_matmul,
    )

    return (
        supports_all_gather_matmul,
        supports_prepare_all_gather_matmul,
        cake_all_gather_matmul,
        cake_prepare_all_gather_matmul,
    )


def cake_sp_eligible(linear, bias) -> bool:
    """Static per-participant gate for the Cake route: unquantized, bias-free,
    BF16/FP16 2-D weight (the Cake kernel has no bias / quantized form)."""
    if bias is not None or linear.bias is not None:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    weight = linear.weight
    return weight.ndim == 2 and weight.dtype in (torch.bfloat16, torch.float16)


def _cake_sp_weight(linear) -> torch.Tensor:
    """The engine's ``[N, K]`` parameter as the ``[K, N]`` view the Cake kernel
    consumes in place (the same operand the stock symm-mem path passes)."""
    return linear.weight.detach().t()


def _cake_sp_weight_key(linear) -> tuple:
    weight = linear.weight
    return (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)


def _cake_sp_capacity_rows(rows: int, world_size: int) -> int:
    """Rows per rank the prepared launcher is sized for.

    The engine's prefill chunk cap (``chunked_prefill_size``, else
    ``max_prefill_tokens``) spread over the TP ranks the way ``sp_entry_scatter``
    shards it, and at least the shard at hand; without a resolved schedule the
    shard at hand is the capacity and the launcher grows on demand.
    """
    cap = 0
    try:
        from sglang.srt.runtime_context import get_schedule

        schedule = get_schedule()
        chunked = int(getattr(schedule, "chunked_prefill_size", 0) or 0)
        cap = (
            chunked
            if chunked > 0
            else int(getattr(schedule, "max_prefill_tokens", 0) or 0)
        )
    except Exception:  # no schedule bag published (unit tests, offline tools)
        cap = 0
    per_rank = ceil_align(cap, world_size) // world_size if cap > 0 else 0
    return max(int(rows), per_rank)


def _cake_sp_launcher(
    linear, input_parallel: torch.Tensor, group, *, world_size: int, detail: str
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """The participant's prepared launcher sized for this shard, or ``None``."""
    _supports_ag, supports_prepare, _ag_matmul, prepare = _cake_sp_kernels()
    rows = int(input_parallel.shape[0])
    weight_key = _cake_sp_weight_key(linear)
    entry = _cake_sp_launchers.get(id(linear))
    if entry is not None and entry[0] != weight_key:
        # Weight reload: the bound storage changed, re-prepare on the new one.
        del _cake_sp_launchers[id(linear)]
        entry = None
    if entry is not None and rows <= entry[1]:
        return entry[2]
    capacity = _cake_sp_capacity_rows(rows, world_size)
    if entry is not None:
        capacity = max(capacity, entry[1])
    w = _cake_sp_weight(linear)
    if not supports_prepare(
        input_parallel, w, world_size=world_size, max_rows=capacity
    ):
        return None
    try:
        launcher = prepare(input_parallel, w, group, max_rows=capacity)
        _count_cake_sp_call("prepare", rows, w.shape[1], world_size, capacity=capacity)
    except (NotImplementedError, ValueError) as error:  # FlashInfer host refusal
        _cake_sp_rejected.add((id(linear), rows, input_parallel.dtype, world_size))
        _log_cake_sp_once(
            "fallback", f"FlashInfer refused to prepare ({error}): {detail}"
        )
        return None
    _cake_sp_launchers[id(linear)] = (weight_key, capacity, launcher)
    return launcher


def cake_column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor
) -> Optional[torch.Tensor]:
    """Cake all-gather + matmul of this rank's shard; ``None`` means "stock".

    Returns the full ``[M_pad, N]`` output in TP-rank order, the same row order
    as ``sp_exit_gather`` followed by the matmul; the caller narrows it.
    """
    tp_group = get_parallel().tp_group
    world_size = tp_group.world_size
    detail = (
        f"inp={tuple(input_parallel.shape)}/"
        f"{str(input_parallel.dtype).removeprefix('torch.')} "
        f"weight={tuple(linear.weight.shape)} world_size={world_size}"
    )
    if torch.cuda.is_current_stream_capturing():
        _log_cake_sp_once(
            "fallback", f"inside CUDA-graph capture (no eager preparation): {detail}"
        )
        _count_cake_sp_call(
            "fallback",
            int(input_parallel.shape[0]),
            int(linear.weight.shape[0]),
            world_size,
            reason="capture",
        )
        return None
    if not _cake_sp_symmetric_backend_ready(input_parallel.device):
        _count_cake_sp_call(
            "fallback",
            int(input_parallel.shape[0]),
            int(linear.weight.shape[0]),
            world_size,
            reason="no-nvshmem",
        )
        return None
    key = (id(linear), int(input_parallel.shape[0]), input_parallel.dtype, world_size)
    if key in _cake_sp_rejected:
        _count_cake_sp_call(
            "fallback",
            int(input_parallel.shape[0]),
            int(linear.weight.shape[0]),
            world_size,
            reason="rejected",
        )
        return None
    supports_ag, _supports_prepare, ag_matmul, _prepare = _cake_sp_kernels()
    group = tp_group.device_group
    launcher = _cake_sp_launcher(
        linear, input_parallel, group, world_size=world_size, detail=detail
    )
    rows, n_out = int(input_parallel.shape[0]), int(linear.weight.shape[0])
    if launcher is not None:
        output = launcher(input_parallel)
        _log_cake_sp_once("taken-prepared", f"prepared launcher: {detail}")
        _count_cake_sp_call("taken-prepared", rows, n_out, world_size)
        return output
    if key in _cake_sp_rejected:
        _count_cake_sp_call("fallback", rows, n_out, world_size, reason="rejected")
        return None
    w = _cake_sp_weight(linear)
    if supports_ag(input_parallel, w, world_size=world_size):
        try:
            output = ag_matmul(input_parallel, w, group)
        except (NotImplementedError, ValueError) as error:  # FlashInfer host refusal
            _cake_sp_rejected.add(key)
            _log_cake_sp_once("fallback", f"FlashInfer refused ({error}): {detail}")
            _count_cake_sp_call("fallback", rows, n_out, world_size, reason="refused")
            return None
        _log_cake_sp_once("taken", f"functional all_gather_matmul: {detail}")
        _count_cake_sp_call("taken", rows, n_out, world_size)
        return output
    _log_cake_sp_once("fallback", f"adapter admission rejected: {detail}")
    _count_cake_sp_call("fallback", rows, n_out, world_size, reason="admission")
    return None


def column_parallel_g_matmul(
    linear, input_parallel: torch.Tensor, bias
) -> torch.Tensor:
    """Megatron SP g for a ColumnParallelLinear participant (qkv / gate_up).

    The input is this rank's sequence shard ``[M_pad/tp, K]``; all-gather it back
    to the full sequence, matmul, and narrow to the real token count (recorded at
    the entry scatter). Uses the fused symm-mem kernel when eligible (all-gather +
    GEMM in one shot), else a plain all-gather + matmul. With
    ``SGLANG_CAKE_ROUTES=sp_all_gather_matmul`` an admitted participant takes
    the Cake all-gather + matmul first (see ``cake_column_parallel_g_matmul``).
    """
    num_tokens = sp_num_tokens()
    if cake_route_enabled(CAKE_ROUTE_SP_ALL_GATHER_MATMUL) and cake_sp_eligible(
        linear, bias
    ):
        output = cake_column_parallel_g_matmul(linear, input_parallel)
        if output is not None:
            return output[:num_tokens]
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
    tp_size = linear.tp_size
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

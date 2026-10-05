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
import os
import re
from typing import Callable, Optional

import torch

from sglang.kernels.cake_kernels._routes import cake_route_enabled
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


def _symm_mem_module():
    """``torch.distributed._symmetric_memory`` (indirection for tests)."""
    import importlib

    return importlib.import_module("torch.distributed._symmetric_memory")


def select_cake_sp_symm_mem_backend() -> bool:
    """Choose torch's NVSHMEM symmetric-memory backend for the Cake route.

    FlashInfer's push-wait all-gather matmul allocates its flags and scratch
    through ``torch.distributed._symmetric_memory`` and refuses any backend but
    NVSHMEM. The backend is process-global and cannot change once the allocator
    has been used (an allocation, or even a backend query resolves it), so it is
    selected here right after distributed setup, before any query. sglang's own
    all-reduce buffers do not go through torch symmetric memory unless the
    torch-symm-mem all-reduce is enabled; when the switch is refused the route
    falls back to the stock path on every call (the adapter admission checks
    the backend). Called from ``distributed.bootstrap.init_parallel_runtime``
    right after the device is selected. Returns whether NVSHMEM is now the
    backend.
    """
    global _cake_sp_nvshmem_backend
    try:
        symm_mem = _symm_mem_module()
    except ImportError as error:
        _log_cake_sp_once("fallback", f"torch symmetric memory unavailable ({error})")
        return False
    available = getattr(symm_mem, "is_nvshmem_available", lambda: False)()
    if not available:
        _log_cake_sp_once(
            "fallback", "NVSHMEM symmetric-memory backend unavailable in this torch"
        )
        return False
    try:
        symm_mem.set_backend("NVSHMEM")
    except Exception as error:  # noqa: BLE001 - allocator already in use
        device = torch.device("cuda", torch.cuda.current_device())
        try:
            current = str(symm_mem.get_backend(device))
        except Exception:  # noqa: BLE001
            current = "unknown"
        if current.upper() == "NVSHMEM":
            _cake_sp_nvshmem_backend = True
            return True
        _log_cake_sp_once(
            "fallback",
            f"cannot select the NVSHMEM symmetric-memory backend ({error}); "
            f"torch backend stays {current}",
        )
        return False
    _cake_sp_nvshmem_backend = True
    logger.info(
        "%s %s: torch symmetric-memory backend set to NVSHMEM "
        "(torch fused symm-mem SP ops disabled for this process)",
        _CAKE_LOG_PREFIX,
        CAKE_ROUTE_SP_ALL_GATHER_MATMUL,
    )
    return True


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
    decision is identical across TP ranks.
    """
    if _cake_sp_nvshmem_backend:
        return False
    if not _HAS_TORCH_SYMM_MEM_FUSED or linear.bias is not None:
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return False
    return linear.weight.dtype in (torch.bfloat16, torch.float16)


# --- Cake all-gather + matmul route (SGLANG_CAKE_ROUTES=sp_all_gather_matmul) --
# ``column_parallel_g_matmul`` is the engine's all-gather + GEMM site. With the
# route selected, an unquantized bias-free BF16/FP16 participant whose shard
# the adapter admits (local rows % 128 == 0, K = 8192; prepared packed-QKV
# launcher for (world_size, N) in {(8, 1280), (4, 2560)} or the functional
# kernel for N = 2048) runs FlashInfer's Cake kernel
# (``communication.prepare_all_gather_matmul`` / ``communication.
# all_gather_matmul``). Everything else keeps the stock path above unchanged.
#
# FlashInfer contract constraints: the weight must be K-major ``[K, N]``
# contiguous while the engine stores ``[N, K]``, so one transposed copy per
# participant is prepared on first eager use and cached on the weight storage
# version; the prepared launcher binds the row count, so launchers are cached
# per (participant, rows) and bounded. SP is prefill-only, so nothing here runs
# inside CUDA-graph capture; the guard below keeps that invariant explicit.

CAKE_ROUTE_SP_ALL_GATHER_MATMUL = "sp_all_gather_matmul"
_CAKE_LOG_PREFIX = "[cake-route]"
_CAKE_SP_MAX_LAUNCHERS_PER_WEIGHT = 8

_cake_sp_logged: set[tuple[str, str]] = set()
# True once select_cake_sp_symm_mem_backend() made NVSHMEM the torch
# symmetric-memory backend. torch's own fused symm-mem ops
# (fused_all_gather_matmul / fused_matmul_reduce_scatter) allocate their
# workspace with a group name, which the NVSHMEM allocator rejects, so the
# stock fused fast path is disabled for the whole process in that case.
_cake_sp_nvshmem_backend = False
_cake_sp_rejected: set[tuple] = set()
# id(linear) -> (weight storage key, K-major weight copy)
_cake_sp_weights: dict[int, tuple[tuple, torch.Tensor]] = {}
# (id(linear), rows, dtype, world_size) -> prepared launcher
_cake_sp_launchers: dict[tuple, Callable[[torch.Tensor], torch.Tensor]] = {}


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


def reset_cake_sp_state_for_tests() -> None:
    global _cake_sp_nvshmem_backend
    _cake_sp_nvshmem_backend = False
    _cake_sp_logged.clear()
    _cake_sp_rejected.clear()
    _cake_sp_weights.clear()
    _cake_sp_launchers.clear()


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
    """K-major ``[K, N]`` contiguous copy of ``linear.weight`` (FI contract).

    Forced by the FlashInfer contract (the stock symm-mem path passes the
    ``weight.t()`` view; Cake needs contiguous K-major storage). Re-prepared
    when the parameter storage or its in-place version changes (weight
    reload), which also drops the launchers prepared on the old copy.
    """
    weight = linear.weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)
    entry = _cake_sp_weights.get(id(linear))
    if entry is None or entry[0] != key:
        w_kn = weight.detach().t().contiguous()
        _cake_sp_weights[id(linear)] = (key, w_kn)
        for launcher_key in [k for k in _cake_sp_launchers if k[0] == id(linear)]:
            del _cake_sp_launchers[launcher_key]
        return w_kn
    return entry[1]


# Diagnostic (SGLANG_CAKE_SP_CHECK=1): after every Cake all-gather + matmul,
# recompute the stock all-gather + matmul on the same inputs and compare within
# the BF16 tolerance (atol 1e-2 + rtol 1e-2). Costs one extra collective + GEMM
# per participant call; for A/B attribution runs only, never for serving.
_CAKE_SP_CHECK = os.environ.get("SGLANG_CAKE_SP_CHECK", "0") == "1"
_CAKE_SP_DUMP_DIR = os.environ.get("SGLANG_CAKE_SP_DUMP_DIR", "")
_CAKE_SP_CHECK_ATOL = 1e-2
_CAKE_SP_CHECK_RTOL = 1e-2
_cake_sp_check_stats = {
    "calls": 0,
    "bad_calls": 0,
    "bad_elems": 0,
    "elems": 0,
    "max_abs": 0.0,
    "dumps": 0,
}


def _cake_sp_check(linear, input_parallel, bias, output, num_tokens) -> None:
    tp_group = get_parallel().tp_group
    rank = int(getattr(tp_group, "rank_in_group", 0))
    gathered = sp_exit_gather(input_parallel, num_tokens=num_tokens)
    ref = linear.quant_method.apply(linear, gathered, bias)
    st = _cake_sp_check_stats
    st["calls"] += 1
    if tuple(ref.shape) != tuple(output.shape):
        logger.warning(
            "%s sp check: shape mismatch cake=%s stock=%s (rank %d)",
            _CAKE_LOG_PREFIX,
            tuple(output.shape),
            tuple(ref.shape),
            rank,
        )
        st["bad_calls"] += 1
        return
    out32 = output.float()
    ref32 = ref.float()
    diff = (out32 - ref32).abs()
    bad = int((diff > _CAKE_SP_CHECK_ATOL + _CAKE_SP_CHECK_RTOL * ref32.abs()).sum())
    max_abs = float(diff.max())
    st["elems"] += diff.numel()
    st["bad_elems"] += bad
    st["max_abs"] = max(st["max_abs"], max_abs)
    if bad:
        st["bad_calls"] += 1
        if st["bad_calls"] <= 5:
            logger.warning(
                "%s sp check: %d/%d elements outside tol, max|d|=%.4g, |ref| max=%.4g "
                "(inp=%s weight=%s rank %d call %d)",
                _CAKE_LOG_PREFIX,
                bad,
                diff.numel(),
                max_abs,
                float(ref32.abs().max()),
                tuple(input_parallel.shape),
                tuple(linear.weight.shape),
                rank,
                st["calls"],
            )
        if _CAKE_SP_DUMP_DIR and st["dumps"] < 4:
            dump_dir = os.path.join(_CAKE_SP_DUMP_DIR, f"rank{rank}")
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(
                {
                    "input_parallel": input_parallel.detach().cpu(),
                    "weight": linear.weight.detach().cpu(),
                    "cake_output": output.detach().cpu(),
                    "stock_output": ref.detach().cpu(),
                    "num_tokens": num_tokens,
                    "world_size": tp_group.world_size,
                    "rank": rank,
                },
                os.path.join(dump_dir, f"sp_check_{st['dumps']}.pt"),
            )
            st["dumps"] += 1
    if st["calls"] % 50 == 1:
        logger.info(
            "%s sp check summary: calls=%d bad_calls=%d bad_elems=%d/%d max|d|=%.4g (rank %d)",
            _CAKE_LOG_PREFIX,
            st["calls"],
            st["bad_calls"],
            st["bad_elems"],
            st["elems"],
            st["max_abs"],
            rank,
        )


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
        return None
    key = (id(linear), int(input_parallel.shape[0]), input_parallel.dtype, world_size)
    if key in _cake_sp_rejected:
        return None
    supports_ag, supports_prepare, ag_matmul, prepare = _cake_sp_kernels()
    w_kn = _cake_sp_weight(linear)
    group = tp_group.device_group
    if supports_prepare(input_parallel, w_kn, world_size=world_size):
        launcher = _cake_sp_launchers.get(key)
        if launcher is None:
            prepared_here = sum(1 for k in _cake_sp_launchers if k[0] == id(linear))
            if prepared_here < _CAKE_SP_MAX_LAUNCHERS_PER_WEIGHT:
                try:
                    launcher = prepare(input_parallel, w_kn, group)
                except (NotImplementedError, ValueError) as error:  # FI host refusal
                    _cake_sp_rejected.add(key)
                    _log_cake_sp_once(
                        "fallback", f"FlashInfer refused to prepare ({error}): {detail}"
                    )
                    return None
                _cake_sp_launchers[key] = launcher
        if launcher is not None:
            output = launcher(input_parallel)
            _log_cake_sp_once(
                "taken-prepared", f"prepared packed-QKV launcher: {detail}"
            )
            return output
    if supports_ag(input_parallel, w_kn, world_size=world_size):
        try:
            output = ag_matmul(input_parallel, w_kn, group)
        except (NotImplementedError, ValueError) as error:  # FI host refusal
            _cake_sp_rejected.add(key)
            _log_cake_sp_once("fallback", f"FlashInfer refused ({error}): {detail}")
            return None
        _log_cake_sp_once("taken", f"functional all_gather_matmul: {detail}")
        return output
    _log_cake_sp_once("fallback", f"adapter admission rejected: {detail}")
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
            if _CAKE_SP_CHECK:
                _cake_sp_check(
                    linear, input_parallel, bias, output[:num_tokens], num_tokens
                )
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

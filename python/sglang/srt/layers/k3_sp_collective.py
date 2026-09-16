"""K3 SP-MoE MNNVL reduce-scatter/all-gather dispatch.

The reduce-scatter folds the pending attention residual into its reduction
epilogue.  The matching all-gather reassembles the token shards after MoE.
Both reuse CustomAllReduceV2's push workspace and fall back as a pair outside
the checked-in GB200/GB300 tuning envelope.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import k3_ar_fusion
from sglang.srt.runtime_context import get_exec, get_parallel

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 7168
# Each supported world size must also have an exact-device tuning table; see
# the dispatch probe in _init_state().
_SUPPORTED_WORLD_SIZES = (4, 8, 16)

# Named persistent symmetric buffers, one per NVLS-aliased tensor. Every rank
# must resolve the same (buffer, offset) for these, which a per-forward
# allocation does not give -- see k3_ar_fusion.symm_buffer.
_O_PROJ = "sp_o_proj"  # TP-partial o_proj output, read by the pull RS
_ALL_GATHER = "sp_all_gather"  # full-batch output of the direct AG
_ATTN_RES_AG = "sp_attn_res_ag"  # ... and of its attention-residual fusion


class _State:
    def __init__(self, group: GroupCoordinator, comm: CustomAllReduceV2):
        self.group = group
        self.comm = comm


_STATE: Optional[_State] = None
_INITIALIZED = False
_O_PROJ_RESULT_BUFFERS: dict[int, torch.Tensor] = {}
_RS_FALLBACK_LOGGED = False
_RS_DISPATCH_LOGGED = False
_AG_DISPATCH_LOGGED = False
_O_PROJ_OUTPUT_ROWS: ContextVar[Optional[int]] = ContextVar(
    "k3_sp_o_proj_output_rows", default=None
)


def _init_state() -> Optional[_State]:
    global _STATE, _INITIALIZED
    if _INITIALIZED:
        return _STATE
    _INITIALIZED = True

    explicit = envs.SGLANG_K3_SP_COLLECTIVE.is_set()
    if explicit and not envs.SGLANG_K3_SP_COLLECTIVE.get():
        return None

    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.utils.common import get_device_sm

    a2a = get_exec().moe.moe_a2a_backend
    group = get_parallel().attn_tp_group
    comm = group.ca_comm
    device_sm = get_device_sm()
    is_custom_allreduce = isinstance(comm, CustomAllReduceV2)
    if is_custom_allreduce:
        comm_disabled = comm.disabled
        has_multicast = comm.has_multicast
    else:
        comm_disabled = None
        has_multicast = None
    if (
        device_sm not in (100, 103)
        or group.world_size not in _SUPPORTED_WORLD_SIZES
        or a2a not in ("megamoe", "deepep")
        or not is_custom_allreduce
        or comm_disabled
        or not has_multicast
    ):
        message = (
            "K3 SP collective requires SM100/SM103, TP4/TP8/TP16, "
            "MegaMoE/DeepEP, and CustomAllReduceV2 with multicast; using NCCL "
            "(sm=%s, world_size=%s, a2a=%s, comm=%s, disabled=%s, "
            "has_multicast=%s)."
            % (
                device_sm,
                group.world_size,
                a2a,
                type(comm).__name__ if comm is not None else None,
                comm_disabled,
                has_multicast,
            )
        )
        (logger.warning if explicit else logger.info)(message)
        return None

    from sglang.kernels.ops.kimi_k3 import attn_res, sp_collective

    # Refuse to enable without a checked-in table for this exact device.
    if (
        sp_collective.get_dispatch(
            "reduce_scatter",
            group.world_size,
            _HIDDEN_SIZE,
            1,
            torch.device("cuda"),
        )
        is None
    ):
        (logger.warning if explicit else logger.info)(
            "K3 SP collective has no tuning table for %s; using NCCL.",
            torch.cuda.get_device_name(),
        )
        return None

    sp_collective.register_comm(comm.obj)
    attn_res.register_comm(comm.obj)
    _STATE = _State(group, comm)
    logger.info(
        "K3 SP collective enabled (TP%d, fused RS residual + AG)",
        group.world_size,
    )
    if envs.SGLANG_K3_SP_ATTN_RES.get():
        logger.info(
            "K3 SP attention-residual carry enabled "
            "(local agg/bank write + normalized AG)"
        )
    return _STATE


def enabled() -> bool:
    return _init_state() is not None


def _symm_buffer(
    state: _State, name: str, rows: int, width: int, dtype: torch.dtype
) -> torch.Tensor:
    """Named persistent symmetric buffer over the group this module reduces on."""
    return k3_ar_fusion.symm_buffer(
        name, rows, width, dtype, group_name=state.group.cpu_group.group_name
    )


def requires_symmetric_rs(num_tokens: int, device: torch.device) -> bool:
    """Whether standalone or fused RS reads o_proj through its NVLS alias."""
    state = _init_state()
    if state is None:
        return False
    from sglang.kernels.ops.kimi_k3 import sp_collective

    dispatch = sp_collective.get_dispatch(
        "reduce_scatter",
        state.group.world_size,
        _HIDDEN_SIZE,
        num_tokens,
        device,
    )
    if dispatch is not None and dispatch.strategy == "pull":
        return True
    if not envs.SGLANG_K3_SP_ATTN_RES.get():
        return False
    fusion = sp_collective.get_fusion_dispatch(
        "reduce_scatter_attn_res",
        state.group.world_size,
        _HIDDEN_SIZE,
        num_tokens,
        device,
    )
    return fusion is not None and fusion.strategy == "fused_pull"


def get_o_proj_output_buffer(
    num_tokens: int, dtype: torch.dtype, hidden_size: int = _HIDDEN_SIZE
) -> torch.Tensor:
    """Persistent symmetric storage for a TP-partial o_proj output."""
    state = _init_state()
    assert state is not None, "K3 SP collective is not initialized"
    return _symm_buffer(state, _O_PROJ, num_tokens, hidden_size, dtype)


@contextmanager
def o_proj_output_rows(num_rows: int):
    """Tell o_proj to target a padded full-batch symmetric output."""
    token = _O_PROJ_OUTPUT_ROWS.set(num_rows)
    try:
        yield
    finally:
        _O_PROJ_OUTPUT_ROWS.reset(token)


def get_o_proj_output_rows(default: int) -> int:
    num_rows = _O_PROJ_OUTPUT_ROWS.get()
    return default if num_rows is None else num_rows


def register_o_proj_output(result: torch.Tensor, output: torch.Tensor) -> None:
    """Associate a model-visible o_proj result/view with its symmetric backing."""
    if (
        result.ndim != output.ndim
        or result.shape[1:] != output.shape[1:]
        or result.shape[0] > output.shape[0]
        or result.dtype != output.dtype
    ):
        raise RuntimeError(
            "K3 o_proj result does not match its persistent symmetric output"
        )
    _O_PROJ_RESULT_BUFFERS[result.data_ptr()] = output


def finish_padded_o_proj_output(
    result: torch.Tensor, num_padded: int
) -> Optional[torch.Tensor]:
    """Zero the padding tail and return the full persistent symmetric buffer."""
    output = _O_PROJ_RESULT_BUFFERS.get(result.data_ptr())
    if output is None or output.shape[0] != num_padded:
        return None
    output[result.shape[0] :].zero_()
    _O_PROJ_RESULT_BUFFERS[output.data_ptr()] = output
    return output


def _resolve_symmetric_o_proj_input(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Recover the persistent o_proj buffer if a model wrapper rebound its view.

    The second element is 0 when neither is symmetric, and the caller falls back.
    """
    input_mc_ptr = k3_ar_fusion.find_mc_ptr(tensor)
    if input_mc_ptr is not None:
        return tensor, input_mc_ptr
    output = _O_PROJ_RESULT_BUFFERS.get(tensor.data_ptr())
    if output is None:
        return tensor, 0
    return output, k3_ar_fusion.find_mc_ptr(output) or 0


def _eligible(
    state: _State,
    tensor: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> bool:
    if (
        tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or tensor.ndim != 2
        or tensor.shape[1] != _HIDDEN_SIZE
        or tensor.shape[0] <= 0
        or tensor.shape[0] % state.group.world_size != 0
    ):
        return False
    if residual is not None:
        local_shape = (tensor.shape[0] // state.group.world_size, tensor.shape[1])
        if (
            residual.shape not in (tensor.shape, local_shape)
            or residual.dtype != tensor.dtype
            or not residual.is_contiguous()
        ):
            return False
    # Pull/direct strategies use separately allocated symmetric tensors. Push
    # workspace capacity is strategy-specific and checked after table dispatch.
    return True


def reduce_scatter_res(
    tensor: torch.Tensor, residual: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Return a fused local shard, or None when the NCCL fallback should run."""
    global _RS_DISPATCH_LOGGED, _RS_FALLBACK_LOGGED
    state = _init_state()
    if state is None:
        return None
    if not _eligible(state, tensor, residual):
        if not _RS_FALLBACK_LOGGED:
            _RS_FALLBACK_LOGGED = True
            logger.warning(
                "K3 SP reduce-scatter ineligible: tensor(shape=%s, dtype=%s, "
                "contiguous=%s), residual(shape=%s, dtype=%s, contiguous=%s), "
                "world_size=%s",
                tuple(tensor.shape),
                tensor.dtype,
                tensor.is_contiguous(),
                None if residual is None else tuple(residual.shape),
                None if residual is None else residual.dtype,
                None if residual is None else residual.is_contiguous(),
                state.group.world_size,
            )
        return None
    from sglang.kernels.ops.kimi_k3 import sp_collective

    dispatch = sp_collective.get_dispatch(
        "reduce_scatter",
        state.group.world_size,
        tensor.shape[1],
        tensor.shape[0],
        tensor.device,
    )
    if dispatch is None:
        if not _RS_FALLBACK_LOGGED:
            _RS_FALLBACK_LOGGED = True
            logger.warning(
                "K3 SP reduce-scatter has no custom dispatch for "
                "world_size=%s, hidden_size=%s, num_tokens=%s, device=%s",
                state.group.world_size,
                tensor.shape[1],
                tensor.shape[0],
                tensor.device,
            )
        return None
    if dispatch.strategy == "push":
        local_bytes = tensor.numel() * tensor.element_size() // state.group.world_size
        if local_bytes > state.comm.max_push_size:
            if not _RS_FALLBACK_LOGGED:
                _RS_FALLBACK_LOGGED = True
                logger.warning(
                    "K3 SP reduce-scatter push requires %s bytes per rank, but "
                    "the communicator workspace has %s; using NCCL.",
                    local_bytes,
                    state.comm.max_push_size,
                )
            return None
    if not _RS_DISPATCH_LOGGED:
        _RS_DISPATCH_LOGGED = True
        logger.info(
            "K3 SP reduce-scatter dispatch: strategy=%s, world_size=%s, "
            "num_tokens=%s, num_blocks=%s, block_size=%s",
            dispatch.strategy,
            state.group.world_size,
            tensor.shape[0],
            dispatch.tuning.num_blocks,
            dispatch.tuning.block_size,
        )
    output = torch.empty(
        (tensor.shape[0] // state.group.world_size, tensor.shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if dispatch.strategy == "push":
        return sp_collective.reduce_scatter_res(
            state.group.world_size,
            tensor,
            output,
            residual,
            tuning=dispatch.tuning,
        )
    if dispatch.strategy == "pull":
        tensor, input_mc_ptr = _resolve_symmetric_o_proj_input(tensor)
        if input_mc_ptr == 0:
            logger.warning("K3 pull RS input is not symmetric; using NCCL.")
            return None
        return sp_collective.reduce_scatter_pull(
            state.group.world_size,
            tensor,
            output,
            residual,
            input_mc_ptr=input_mc_ptr,
            tuning=dispatch.tuning,
        )
    raise AssertionError(f"unknown K3 reduce-scatter strategy: {dispatch.strategy}")


def reduce_scatter_attn_res(
    tensor: torch.Tensor,
    residual: Optional[torch.Tensor],
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    nvb: int,
    eps: float,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Fused NVLS pull RS + local residual + attention-residual aggregation."""
    state = _init_state()
    if (
        state is None
        or not envs.SGLANG_K3_SP_ATTN_RES.get()
        or not _eligible(state, tensor, residual)
    ):
        return None
    local_tokens = tensor.shape[0] // state.group.world_size
    if (
        bank.shape[0] != local_tokens
        or bank.ndim != 3
        or bank.shape[2] != _HIDDEN_SIZE
        or not bank.is_contiguous()
        or cw.shape != (_HIDDEN_SIZE,)
        or ow.shape != (_HIDDEN_SIZE,)
    ):
        return None
    from sglang.kernels.ops.kimi_k3 import attn_res, sp_collective

    dispatch = sp_collective.get_fusion_dispatch(
        "reduce_scatter_attn_res",
        state.group.world_size,
        tensor.shape[1],
        tensor.shape[0],
        tensor.device,
    )
    if dispatch is None or dispatch.strategy != "fused_pull":
        return None
    tensor, input_mc_ptr = _resolve_symmetric_o_proj_input(tensor)
    if input_mc_ptr == 0:
        logger.warning("K3 fused pull RS input is not symmetric; using separate path.")
        return None
    normed = torch.empty(
        (local_tokens, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device
    )
    prefix = torch.empty_like(normed)
    attn_res.attn_res_fused_pull_rs(
        state.group.world_size,
        tensor,
        residual,
        bank,
        cw,
        ow,
        normed,
        prefix,
        nvb,
        eps,
        input_mc_ptr=input_mc_ptr,
        max_blocks=dispatch.max_blocks,
    )
    return normed, prefix


def all_gather(tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the reassembled full batch, or None for the NCCL fallback."""
    global _AG_DISPATCH_LOGGED
    state = _init_state()
    if state is None:
        return None
    global_tokens = tensor.shape[0] * state.group.world_size
    if (
        tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or tensor.ndim != 2
        or tensor.shape[1] != _HIDDEN_SIZE
        or tensor.shape[0] <= 0
    ):
        return None
    from sglang.kernels.ops.kimi_k3 import sp_collective

    dispatch = sp_collective.get_dispatch(
        "all_gather",
        state.group.world_size,
        tensor.shape[1],
        global_tokens,
        tensor.device,
    )
    if dispatch is None:
        return None
    if (
        dispatch.strategy == "push"
        and tensor.numel() * tensor.element_size() > state.comm.max_push_size
    ):
        return None
    if not _AG_DISPATCH_LOGGED:
        _AG_DISPATCH_LOGGED = True
        logger.info(
            "K3 SP all-gather dispatch: strategy=%s, world_size=%s, "
            "num_tokens=%s, num_blocks=%s, block_size=%s",
            dispatch.strategy,
            state.group.world_size,
            global_tokens,
            dispatch.tuning.num_blocks,
            dispatch.tuning.block_size,
        )
    output_shape = (global_tokens, tensor.shape[1])
    if dispatch.strategy == "push":
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        return sp_collective.all_gather(
            state.group.world_size,
            tensor,
            output,
            tuning=dispatch.tuning,
        )
    if dispatch.strategy == "direct":
        output = _symm_buffer(
            state, _ALL_GATHER, global_tokens, tensor.shape[1], tensor.dtype
        )
        return sp_collective.all_gather_direct(
            state.group.world_size,
            tensor,
            output,
            output_mc_ptr=k3_ar_fusion.get_mc_ptr(output),
            tuning=dispatch.tuning,
        )
    raise AssertionError(f"unknown K3 all-gather strategy: {dispatch.strategy}")


def attn_res_all_gather(
    prefix: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
) -> Optional[torch.Tensor]:
    """Fused local attention-residual aggregation + direct multicast AG."""
    state = _init_state()
    if (
        state is None
        or not envs.SGLANG_K3_SP_ATTN_RES.get()
        or prefix.dtype != torch.bfloat16
        or not prefix.is_contiguous()
        or prefix.ndim != 2
        or prefix.shape[1] != _HIDDEN_SIZE
        or prefix.shape[0] <= 0
        or bank.ndim != 3
        or bank.shape[0] != prefix.shape[0]
        or bank.shape[2] != _HIDDEN_SIZE
        or not bank.is_contiguous()
        or cw.shape != (_HIDDEN_SIZE,)
        or ow.shape != (_HIDDEN_SIZE,)
    ):
        return None
    from sglang.kernels.ops.kimi_k3 import attn_res, sp_collective

    global_tokens = prefix.shape[0] * state.group.world_size
    dispatch = sp_collective.get_fusion_dispatch(
        "attn_res_all_gather",
        state.group.world_size,
        prefix.shape[1],
        global_tokens,
        prefix.device,
    )
    if dispatch is None or dispatch.strategy != "fused_direct":
        return None
    output = _symm_buffer(
        state, _ATTN_RES_AG, global_tokens, prefix.shape[1], prefix.dtype
    )
    attn_res.attn_res_fused_direct_ag(
        state.group.world_size,
        prefix,
        bank,
        cw,
        ow,
        output,
        nvb,
        eps,
        output_mc_ptr=k3_ar_fusion.get_mc_ptr(output),
        max_blocks=dispatch.max_blocks,
        write_prefix=write_prefix,
    )
    return output

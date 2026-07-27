"""K3 SP-MoE MNNVL reduce-scatter/all-gather dispatch.

The reduce-scatter folds the pending attention residual into its reduction
epilogue.  The matching all-gather reassembles the token shards after MoE.
Both reuse CustomAllReduceV2's push workspace and fall back as a pair outside
the checked-in GB300 tuning envelope.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 7168
_SUPPORTED_WORLD_SIZES = (4, 8)


class _State:
    def __init__(self, group: GroupCoordinator, comm: CustomAllReduceV2):
        self.group = group
        self.comm = comm


_STATE: Optional[_State] = None
_INITIALIZED = False
_O_PROJ_BUFFERS: dict[tuple, torch.Tensor] = {}
_O_PROJ_RESULT_BUFFERS: dict[int, torch.Tensor] = {}
_SYMM_MC_PTRS: dict[int, int] = {}
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
    from sglang.srt.runtime_context import get_parallel, get_server_args
    from sglang.srt.utils.common import get_device_sm

    server_args = get_server_args()
    a2a = server_args.moe_a2a_backend
    group = get_parallel().attn_tp_group
    comm = group.ca_comm
    if (
        get_device_sm() != 103
        or group.world_size not in _SUPPORTED_WORLD_SIZES
        or a2a not in ("megamoe", "deepep")
        or not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
        or comm.mc_base_ptr == 0
    ):
        message = (
            "K3 SP collective requires SM103, TP4/TP8, MegaMoE/DeepEP, and "
            "CustomAllReduceV2 with multicast; using NCCL."
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

    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    attn_res.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
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


def symm_alloc():
    """Allocate a tensor from torch's multicast-bound symmetric pool."""
    from sglang.srt.layers.k3_ar_fusion import symm_alloc as _symm_alloc

    return _symm_alloc()


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


def _find_mc_ptr(state: _State, tensor: torch.Tensor) -> int:
    cached = _SYMM_MC_PTRS.get(tensor.data_ptr())
    if cached is not None:
        return cached

    import torch.distributed._symmetric_memory as torch_symm_mem

    handle = torch_symm_mem.rendezvous(tensor, state.group.cpu_group.group_name)
    if handle is None or handle.multicast_ptr == 0:
        return 0
    ptr = handle.multicast_ptr + tensor.data_ptr() - handle.buffer_ptrs[state.comm.rank]
    _SYMM_MC_PTRS[tensor.data_ptr()] = ptr
    return ptr


def get_o_proj_output_buffer(
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    hidden_size: int = _HIDDEN_SIZE,
) -> torch.Tensor:
    """Return persistent symmetric storage for a TP-partial o_proj output."""
    state = _init_state()
    if state is None:
        raise RuntimeError("K3 SP collective is not initialized")
    device = torch.device(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    key = (device.type, device_index, dtype, num_tokens, hidden_size)
    output = _O_PROJ_BUFFERS.get(key)
    if output is None:
        with symm_alloc():
            output = torch.empty(
                (num_tokens, hidden_size),
                dtype=dtype,
                device=device,
            )
        if _find_mc_ptr(state, output) == 0:
            raise RuntimeError("failed to multicast-bind K3 o_proj output buffer")
        _O_PROJ_BUFFERS[key] = output
        logger.info(
            "Allocated persistent symmetric K3 o_proj buffer "
            "(tokens=%d, hidden=%d, ptr=%#x)",
            num_tokens,
            hidden_size,
            output.data_ptr(),
        )
    return output


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


def _resolve_symmetric_o_proj_input(
    state: _State, tensor: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Recover the persistent o_proj buffer if a model wrapper rebound its view."""
    input_mc_ptr = _find_mc_ptr(state, tensor)
    if input_mc_ptr != 0:
        return tensor, input_mc_ptr
    output = _O_PROJ_RESULT_BUFFERS.get(tensor.data_ptr())
    if output is None:
        return tensor, 0
    return output, _find_mc_ptr(state, output)


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
    local_bytes = tensor.numel() * tensor.element_size() // state.group.world_size
    return local_bytes <= state.comm.max_push_size


def reduce_scatter_res(
    tensor: torch.Tensor, residual: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Return a fused local shard, or None when the NCCL fallback should run."""
    state = _init_state()
    if state is None or not _eligible(state, tensor, residual):
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
        return None
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
        tensor, input_mc_ptr = _resolve_symmetric_o_proj_input(state, tensor)
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
    tensor, input_mc_ptr = _resolve_symmetric_o_proj_input(state, tensor)
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
        or tensor.numel() * tensor.element_size() > state.comm.max_push_size
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
    output_shape = (global_tokens, tensor.shape[1])
    if dispatch.strategy == "push":
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        return sp_collective.all_gather(
            state.group.world_size,
            tensor,
            output,
            ws_mc_base=state.comm.mc_base_ptr,
            tuning=dispatch.tuning,
        )
    if dispatch.strategy == "direct":
        import torch.distributed._symmetric_memory as torch_symm_mem

        with symm_alloc():
            output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        handle = torch_symm_mem.rendezvous(output, state.group.cpu_group.group_name)
        assert handle is not None and handle.multicast_ptr != 0
        output_mc_ptr = (
            handle.multicast_ptr
            + output.data_ptr()
            - handle.buffer_ptrs[state.comm.rank]
        )
        return sp_collective.all_gather_direct(
            state.group.world_size,
            tensor,
            output,
            output_mc_ptr=output_mc_ptr,
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
    output_shape = (global_tokens, prefix.shape[1])
    with symm_alloc():
        output = torch.empty(output_shape, dtype=prefix.dtype, device=prefix.device)
    output_mc_ptr = _find_mc_ptr(state, output)
    if output_mc_ptr == 0:
        logger.warning(
            "K3 fused direct AG output is not symmetric; using separate path."
        )
        return None
    attn_res.attn_res_fused_direct_ag(
        state.group.world_size,
        prefix,
        bank,
        cw,
        ow,
        output,
        nvb,
        eps,
        output_mc_ptr=output_mc_ptr,
        max_blocks=dispatch.max_blocks,
        write_prefix=write_prefix,
    )
    return output

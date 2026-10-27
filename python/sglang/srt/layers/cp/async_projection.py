"""Opt-in CP input gather/projection overlap for GLM KDA.

The mixer keeps its inputs CP-sharded until the first projection. Output
reduction, mHC residual placement and recurrent state ownership are unchanged.
Small batches and unsupported layouts retain the original path. Graph support
reserves its symmetric workspace before capture, using the same feature switch.
"""

import logging

import torch

from sglang.srt.environ import envs

_GRAPH_WORKSPACE_CAPACITY: dict[tuple[int, str], int] = {}
_CAPTURE_LOGGED = False
logger = logging.getLogger(__name__)


def prepare_kda_cp_async_graph(max_local_tokens: int, hidden_size: int, device):
    """Reserve a fixed workspace outside capture; never grow a captured pool."""
    if not kda_cp_async_tp_enabled():
        return
    from torch.distributed import _symmetric_memory as symm_mem

    from sglang.srt.runtime_context import get_parallel

    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("KDA Async-TP workspace must be prepared before capture")
        group_name = get_parallel().attn_cp_group.device_group.group_name
        capacity = max_local_tokens * hidden_size * 2  # BF16 input shard
        symm_mem.get_symm_mem_workspace(group_name, min_size=capacity)
        # Create the communication stream before graph capture as well.
        symm_mem._get_backend_stream()
        _GRAPH_WORKSPACE_CAPACITY[(torch.cuda.current_device(), group_name)] = capacity


@torch.library.custom_op(
    "sglang::kda_cp_async_graph_projection", mutates_args=(), device_types="cuda"
)
def _kda_cp_async_graph_projection(
    shard: torch.Tensor, weight: torch.Tensor, group_name: str, cp_size: int
) -> torch.Tensor:
    """Opaque compiler boundary retaining PyTorch's multi-stream Async-TP."""
    global _CAPTURE_LOGGED
    from torch.distributed import _symmetric_memory as symm_mem  # noqa: F401

    capacity = _GRAPH_WORKSPACE_CAPACITY.get((shard.device.index, group_name), 0)
    if shard.numel() * shard.element_size() > capacity:
        raise RuntimeError(
            "KDA Async-TP graph workspace is not prepared for this shape"
        )
    if not _CAPTURE_LOGGED and torch.cuda.is_current_stream_capturing():
        logger.info(
            "KDA Async-TP BCG capture: local_rows=%s, hidden=%s, output=%s, workspace_bytes=%s",
            shard.shape[0],
            shard.shape[1],
            weight.shape[0],
            capacity,
        )
        _CAPTURE_LOGGED = True
    _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        shard, [weight.t()], 0, group_name, return_A=False
    )
    return outputs[0]


@_kda_cp_async_graph_projection.register_fake
def _kda_cp_async_graph_projection_fake(shard, weight, group_name, cp_size):
    return shard.new_empty((shard.shape[0] * cp_size, weight.shape[0]))


def kda_cp_async_tp_enabled() -> bool:
    if not envs.SGLANG_ENABLE_KDA_CP_ASYNC_TP.get() or torch.version.cuda is None:
        return False
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    return (
        p.enable_prefill_cp
        and p.enable_cp_tp_group_sharing
        and p.cp_strategy == "interleave"
        and p.attn_cp_size == p.tp_size
        and p.attn_cp_size > 1
        and p.attn_dp_size == 1
    )


def try_kda_cp_async_projection(shard: torch.Tensor, weight: torch.Tensor):
    """Return rank-major projected rows, or None for the original path.

    Only the unquantized, bias-free BF16 merged projection is audited. The
    caller restores sequence order *after* projection, then runs the same
    KDA gate projections, convolution and recurrent update as before.
    """
    if (
        not shard.is_cuda
        or shard.ndim != 2
        or shard.dtype != torch.bfloat16
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or weight.shape[1] != shard.shape[1]
        or shard.shape[0] < 4096
        or not shard.is_contiguous()
        or not weight.is_contiguous()
    ):
        return None
    # Importing the module registers the Python CUDA implementations of the op.
    from torch.distributed import _symmetric_memory as _symm_mem  # noqa: F401

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().attn_cp_group.device_group
    if (
        torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
        or (shard.device.index, group.group_name) in _GRAPH_WORKSPACE_CAPACITY
    ):
        capacity = _GRAPH_WORKSPACE_CAPACITY.get(
            (shard.device.index, group.group_name), 0
        )
        if shard.numel() * shard.element_size() > capacity:
            # Larger eager requests must not invalidate workspace retained by
            # existing graphs. The caller falls back to its ordinary gather.
            return None
        return _kda_cp_async_graph_projection(
            shard, weight, group.group_name, get_parallel().attn_cp_size
        )
    # PyTorch owns the symmetric workspace and shard/GEMM stream dependencies.
    # All CP ranks must invoke this with the same physical shard shape.
    _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
        shard, [weight.t()], 0, group.group_name, return_A=False
    )
    return outputs[0]

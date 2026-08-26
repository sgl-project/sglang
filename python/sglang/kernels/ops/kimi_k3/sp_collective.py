"""K3 SP-MoE bf16 reduce-scatter and all-gather over MNNVL push memory."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator


class Tuning(NamedTuple):
    num_blocks: int
    block_size: int


class Dispatch(NamedTuple):
    strategy: str
    tuning: Tuning


class FusionDispatch(NamedTuple):
    strategy: str
    max_blocks: int


# Safe seed values. GB300 production values are loaded by the layer glue from
# the checked-in JSON after the sweep settles.
DEFAULT_TUNING = Tuning(num_blocks=16, block_size=256)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "sp_collective")
_TABLES: dict[str, Optional[dict]] = {}


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def _table(world_size: int, hidden_size: int, device: torch.device) -> Optional[dict]:
    path = os.path.join(
        _CONFIG_DIR,
        (
            f"world={world_size},H={hidden_size},"
            f"device_name={_device_name(device)}.json"
        ),
    )
    if path not in _TABLES:
        if os.path.exists(path):
            with open(path) as f:
                _TABLES[path] = json.load(f)
        else:
            _TABLES[path] = None
    return _TABLES[path]


def get_dispatch(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
) -> Optional[Dispatch]:
    """Return the tuned strategy, or None when the table selects NCCL."""
    table = _table(world_size, hidden_size, device)
    if table is None:
        return None
    raw_configs = table["configs"].get(kind)
    if raw_configs is None:
        return None
    configs = {int(k): v for k, v in raw_configs.items()}
    bucket = min(configs)
    for candidate in sorted(configs):
        if candidate <= num_tokens:
            bucket = candidate
        else:
            break
    config = configs[bucket]
    if config["strategy"] == "nccl":
        return None
    return Dispatch(
        config["strategy"],
        Tuning(config["num_blocks"], config["block_size"]),
    )


def get_tuning(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
) -> Optional[Tuning]:
    """Compatibility helper for callers that only support staging push."""
    dispatch = get_dispatch(kind, world_size, hidden_size, num_tokens, device)
    if dispatch is None or dispatch.strategy != "push":
        return None
    return dispatch.tuning


def get_fusion_dispatch(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
) -> Optional[FusionDispatch]:
    """Return a measured fused strategy, or None for the separate path."""
    table = _table(world_size, hidden_size, device)
    if table is None:
        return None
    raw_configs = table["configs"].get(kind)
    if raw_configs is None:
        return None
    configs = {int(k): v for k, v in raw_configs.items()}
    bucket = min(configs)
    for candidate in sorted(configs):
        if candidate <= num_tokens:
            bucket = candidate
        else:
            break
    config = configs[bucket]
    if config["strategy"] == "separate":
        return None
    return FusionDispatch(config["strategy"], config["max_blocks"])


@cache_once
def _jit_module(world_size: int) -> Module:
    args = make_cpp_args(world_size, is_arch_support_pdl())
    cls = f"SPCollectiveKernel<{args}>"
    return load_jit(
        "kimi_k3_sp_collective",
        *args,
        cuda_files=["kimi_k3/comm/sp_collective.cuh"],
        cuda_wrappers=[
            ("reduce_scatter_res", f"{cls}::reduce_scatter_res"),
            ("reduce_scatter_pull", f"{cls}::reduce_scatter_pull"),
            ("all_gather", f"{cls}::all_gather"),
            ("all_gather_direct", f"{cls}::all_gather_direct"),
        ],
        extra_cuda_cflags=["-O3"],
    )


_COMM_MAP: dict[int, Communicator] = {}


def register_comm(comm: Communicator) -> None:
    # One communicator per world_size per process -- see the note in
    # kimi_k3/all_reduce.py::register_comm. The ops key only on world_size, so an
    # overwrite here would hand the old group's callers the new group's peer
    # pointers.
    prev = _COMM_MAP.get(comm.world_size)
    assert prev is None or prev is comm, (
        f"a different communicator is already registered for world_size="
        f"{comm.world_size}"
    )
    _COMM_MAP[comm.world_size] = comm


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_res_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
    residual_is_local: bool,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).reduce_scatter_res(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        None if residual is None else residual.view(-1),
        residual_is_local,
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _reduce_scatter_pull_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor],
    residual_is_local: bool,
    input_mc_ptr: int,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).reduce_scatter_pull(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        None if residual is None else residual.view(-1),
        residual_is_local,
        input_mc_ptr,
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _all_gather_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).all_gather(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        num_blocks,
        block_size,
    )


@register_custom_op(mutates_args=["output"])
def _all_gather_direct_op(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    output_mc_ptr: int,
    num_blocks: int,
    block_size: int,
) -> None:
    _jit_module(world_size).all_gather_direct(
        _COMM_MAP[world_size],
        input.view(-1),
        output.view(-1),
        output_mc_ptr,
        num_blocks,
        block_size,
    )


def reduce_scatter_res(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    residual_is_local = residual is not None and residual.numel() == output.numel()
    _reduce_scatter_res_op(
        world_size,
        input,
        output,
        residual,
        residual_is_local,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def reduce_scatter_pull(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    input_mc_ptr: int,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    residual_is_local = residual is not None and residual.numel() == output.numel()
    _reduce_scatter_pull_op(
        world_size,
        input,
        output,
        residual,
        residual_is_local,
        input_mc_ptr,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def all_gather(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    _all_gather_op(
        world_size,
        input,
        output,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output


def all_gather_direct(
    world_size: int,
    input: torch.Tensor,
    output: torch.Tensor,
    *,
    output_mc_ptr: int,
    tuning: Tuning = DEFAULT_TUNING,
) -> torch.Tensor:
    _all_gather_direct_op(
        world_size,
        input,
        output,
        output_mc_ptr,
        tuning.num_blocks,
        tuning.block_size,
    )
    return output

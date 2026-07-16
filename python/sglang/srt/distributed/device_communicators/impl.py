from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Callable, ContextManager, Dict, List, Optional

import msgspec
import torch

from sglang.srt.compilation.compilation_config import register_split_op
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from .base import BaseCommunicator


class CommunicatorImpl(msgspec.Struct, kw_only=True, weakref=True):
    unique_name: str
    world_size: int
    capture_comms: List[BaseCommunicator]
    all_reduce_comms: List[BaseCommunicator]
    all_gather_comms: List[BaseCommunicator]
    reduce_scatter_comms: List[BaseCommunicator]

    # NOTE: never use this, this is only kept for compatibility with
    # python/sglang/srt/model_executor/mindspore_runner.py
    device_group: torch.distributed.ProcessGroup
    ranks: List[int]
    local_rank: int

    def __post_init__(self):
        _register_group(self)

    def graph_capture_context(self) -> List[ContextManager]:
        ctx_list = []
        for comm in self.capture_comms:
            if (comm_ctx := comm.graph_capture_context()) is not None:
                ctx_list.append(comm_ctx)
        return ctx_list

    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        for i, comm in enumerate(self.all_reduce_comms):
            if comm.disabled:
                continue
            mode = comm.get_all_reduce_mode(input_)
            if mode is not None and mode.supports(inplace):
                if not comm.should_use_custom_op():
                    return comm.all_reduce(input_, inplace=inplace)
                # prefer in-place when the caller does not care
                use_inplace = mode.can_inplace() if inplace is None else inplace
                if use_inplace:
                    inplace_all_reduce(input_, self.unique_name, i)
                    return input_
                else:
                    return outplace_all_reduce(input_, self.unique_name, i)
        raise ValueError(f"No compatible all-reduce communicator found: {inplace = }")

    def reduce_scatter_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: torch.Tensor,  # for now, out is never None
    ) -> torch.Tensor:
        for i, comm in enumerate(self.reduce_scatter_comms):
            if comm.disabled:
                continue
            if not comm.should_use_custom_op():
                return comm.reduce_scatter_tensor(input_, out=out)
            inplace_reduce_scatter(input_, self.unique_name, i, out=out)
            return out
        raise ValueError("No compatible reduce-scatter communicator found")

    def all_gather_into_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, comm in enumerate(self.all_gather_comms):
            if comm.disabled:
                continue
            if not comm.should_use_custom_op():
                return comm.all_gather_into_tensor(input_, out=out)
            if out is not None:
                inplace_all_gather(input_, self.unique_name, i, out=out)
                return out
            else:
                return outplace_all_gather(input_, self.unique_name, i)
        raise ValueError("No compatible all-gather communicator found")


# NOTE: never use any of the following functions/variable outside this module
# the only exception is
# python/sglang/srt/model_executor/mindspore_runner.py
# we keep backward compatibility for this file

_GROUPS: Dict[str, Callable[[], Optional[CommunicatorImpl]]] = {}


def _register_group(group: CommunicatorImpl) -> None:
    _GROUPS[group.unique_name] = weakref.ref(group)


def _get_group(group_name: str) -> CommunicatorImpl:
    assert group_name in _GROUPS, f"Group {group_name} is not found."
    group = _GROUPS[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group


def _fake_all_gather(input: torch.Tensor, group_name: str) -> torch.Tensor:
    from sglang.srt.distributed.device_communicators.base import allocate_all_gather

    return allocate_all_gather(input, _get_group(group_name).world_size)


@register_custom_op(mutates_args=["input_"])
@register_split_op()
def inplace_all_reduce(input_: torch.Tensor, group_name: str, method: int) -> None:
    group = _get_group(group_name)
    group.all_reduce_comms[method].all_reduce(input_, inplace=True)


@register_custom_op(out_shape="input_")
def outplace_all_reduce(
    input_: torch.Tensor, group_name: str, method: int
) -> torch.Tensor:
    group = _get_group(group_name)
    return group.all_reduce_comms[method].all_reduce(input_, inplace=False)


@register_custom_op(mutates_args=["out"])
def inplace_reduce_scatter(
    input_: torch.Tensor, group_name: str, method: int, *, out: torch.Tensor
) -> None:
    group = _get_group(group_name)
    group.reduce_scatter_comms[method].reduce_scatter_tensor(input_, out=out)


@register_custom_op(mutates_args=["out"])
def inplace_all_gather(
    input_: torch.Tensor, group_name: str, method: int, *, out: torch.Tensor
) -> None:
    group = _get_group(group_name)
    group.all_gather_comms[method].all_gather_into_tensor(input_, out=out)


@register_custom_op(fake_impl=_fake_all_gather)
def outplace_all_gather(
    input_: torch.Tensor, group_name: str, method: int
) -> torch.Tensor:
    group = _get_group(group_name)
    return group.all_gather_comms[method].all_gather_into_tensor(input_)


# NOTE(dark): we don't make them class method due to conflict with piecewise cuda graph

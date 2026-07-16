from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.dp_attention import is_allocation_symmetric

from .base import AllReduceMode, BaseCommunicator
from .pynccl_allocator import (
    debug_check_symmetric_mempool,
    is_symmetric_memory_enabled,
    use_symmetric_memory,
)

if TYPE_CHECKING:
    from sglang.srt.distributed import GroupCoordinator
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator


class PyNcclSymmMemCommunicator(BaseCommunicator):
    name = "pynccl_symm_mem"

    def __init__(self, group_coordinator: GroupCoordinator, pynccl: PyNcclCommunicator):
        self.group_coordinator = group_coordinator
        self.pynccl = pynccl
        super().__init__(self.pynccl.world_size)

    @property
    def disabled(self) -> bool:
        return self._disabled or not is_symmetric_memory_enabled()

    def should_use_custom_op(self) -> bool:
        return True

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        # always inplace
        return AllReduceMode.INPLACE

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        self.assert_inplace("all_reduce", inplace)
        debug_check_symmetric_mempool(
            self.group_coordinator, {"input": input_}, "all_reduce"
        )
        with self.pynccl.change_state(enable=True):
            self.pynccl.all_reduce(input_, inplace=True)
        return input_

    @BaseCommunicator.validate
    def all_gather_into_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            with self._allocation_context():
                out = self.allocate_all_gather(input_)
        debug_check_symmetric_mempool(
            self.group_coordinator, {"output": out}, "all_gather_into_tensor"
        )
        with self.pynccl.change_state(enable=True):
            self.pynccl.all_gather_into_tensor(input_, out=out)
        return out

    @BaseCommunicator.validate
    def reduce_scatter_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            with self._allocation_context():
                out = self.allocate_reduce_scatter(input_)
        debug_check_symmetric_mempool(
            self.group_coordinator,
            {"output": out, "input": input_},
            "reduce_scatter_tensor",
        )
        with self.pynccl.change_state(enable=True):
            self.pynccl.reduce_scatter_tensor(input_, out=out)
        return out

    def _allocation_context(self):
        return use_symmetric_memory(
            group_coordinator=self.group_coordinator,
            disabled=not is_allocation_symmetric(),
        )

from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.utils import is_npu

from .base import AllReduceMode, BaseCommunicator

_is_npu = is_npu()


class TorchDefaultCommunicator(BaseCommunicator):
    name = "torch_native"

    def __init__(
        self,
        rank_in_group: int,
        ranks: List[int],
        device_group: dist.ProcessGroup,
    ) -> None:
        self.ranks = ranks
        self.rank_in_group = rank_in_group
        self.device_group = device_group
        super().__init__(len(ranks))

    def change_state(self, enable: bool):
        assert enable, "TorchDefaultCommunicator cannot be disabled"
        return super().change_state(enable)

    def should_use_custom_op(self) -> bool:
        # Route through the registered custom ops so Dynamo treats these
        # collectives as opaque calls instead of decomposing them into
        # _c10d_functional primitives (which breaks XPU graph capture and
        # perturbs piecewise CUDA graph splitting). NPU keeps direct calls,
        # matching the pre-refactor dispatch.
        return not _is_npu

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        return AllReduceMode.INPLACE

    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        self.assert_inplace("all_reduce", inplace)
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather_into_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = self.allocate_all_gather(input_)
        dist.all_gather_into_tensor(out, input_, group=self.device_group)
        return out

    def all_gather(
        self,
        input_: torch.Tensor,
        *,
        out_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        if out_list is None:
            out_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(out_list, input_, group=self.device_group)
        return out_list

    def reduce_scatter_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = self.allocate_reduce_scatter(input_)
        dist.reduce_scatter_tensor(out, input_, group=self.device_group)
        return out

    def reduce_scatter(
        self,
        input_list: List[torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input_list[self.rank_in_group])
        dist.reduce_scatter(out, input_list, group=self.device_group)
        return out

    def gather(
        self,
        input_: torch.Tensor,
        dst: int,
        *,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        gather_list = None
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.gather(input_, gather_list, dst=self.ranks[dst], group=self.device_group)
        return None if gather_list is None else torch.cat(gather_list, dim=dim)

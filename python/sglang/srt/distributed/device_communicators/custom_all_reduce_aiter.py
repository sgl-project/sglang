from typing import Optional

import torch
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from torch.distributed import ProcessGroup

from sglang.srt.environ import envs

from .base import AllReduceMode, BaseCommunicator


class AiterCustomAllReduce(BaseCommunicator):
    name = "custom_all_reduce_aiter"

    def __init__(self, group: ProcessGroup, *args, **kwargs):
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self.comm = CustomAllreduce(
            group, *args, **kwargs, enable_register_for_capturing=not tms_cudagraph
        )

    def graph_capture_context(self):
        return self.comm.capture()

    def should_use_custom_op(self) -> bool:
        return True

    @property
    def disabled(self) -> bool:
        return self._disabled or self.comm.disabled

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        can_use = self.comm.should_custom_ar(input_)
        return AllReduceMode.OUTPLACE if can_use else None

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        self.assert_outplace("all_reduce", inplace)
        out = self.comm.custom_all_reduce(input_)
        assert out is not None
        return out

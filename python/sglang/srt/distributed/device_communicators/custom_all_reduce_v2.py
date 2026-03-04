import logging
from contextlib import contextmanager
from typing import List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_weak_contiguous,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device, None] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from sglang.jit_kernel.all_reduce import get_custom_all_reduce_cls

        if max_size is None:
            max_size = 32 * 1024 * 1024  # default to 32MB

        self.max_size = max_size
        self.group = group
        self.disabled = True

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."
        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes."
            )
            return

        MAX_GRAPH_INPUTS = 131072
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        cls = get_custom_all_reduce_cls()
        self.obj = cls(self.rank, self.world_size, self.max_size, MAX_GRAPH_INPUTS)
        self._post_init_obj()
        self.disabled = False
        log_info_on_rank0(logger, "Custom allreduce v2 initialized successfully")

    @contextmanager
    def capture(self):
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
        if not self.disabled:
            pairs = self.obj.share_graph_inputs()
            handles = [handle for _, handle in pairs]
            offsets = [offset for offset, _ in pairs]
            handles_all = self._share_list(handles)
            offsets_all = self._share_list(offsets)
            result = [list(zip(o, h)) for o, h in zip(offsets_all, handles_all)]
            self.obj.register_inputs(result)
            log_info_on_rank0(logger, f"Registering {len(pairs)} cuda graph addresses")

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        return inp_size <= self.max_size

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """Main allreduce API compatible with GroupCoordinator.

        Returns the reduced tensor, or None if custom AR is disabled/unsuitable.
        """
        if self.disabled or not self.should_custom_ar(input):
            return None
        return self._all_reduce(input)

    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self.obj.free()

    def _all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the actual all-reduce via JIT kernel."""
        input_bytes = input.numel() * input.element_size()
        # HARDCODED: 128KB threshold for 1-shot vs 2-shot
        THRESHOLD_2_SHOT = 128 * 1024
        shots = 1 if input_bytes <= THRESHOLD_2_SHOT or self.world_size == 2 else 2
        return torch.from_dlpack(self.obj.all_reduce(input, shots))

    def _post_init_obj(self):
        handles = [self.obj.share_storage()]
        result = self._share_list(handles)
        assert all(len(r) == 1 for r in result)
        result = [h[0] for h in result]
        self.obj.post_init(result)

    def _share_list(self, input: List[T]) -> List[List[T]]:
        input_tensor = torch.tensor(input, dtype=torch.int32, device="cpu")
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.group)
        return [g.tolist() for g in gather_list]

    def __del__(self):
        self.close()

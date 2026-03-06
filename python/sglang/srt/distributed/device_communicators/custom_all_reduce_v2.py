import logging
from contextlib import contextmanager
from typing import List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

T = TypeVar("T")

INF = 1 << 60


# NOTE(dark): This is tuned on hopper GPUs, also need tuning on blackwell/ampere
THRESHOLD_2_SHOT_MAP = {
    2: INF,
    3: 512 * 1024,  # 512KB
    4: 256 * 1024,  # 256KB
    5: 256 * 1024,  # 256KB
    6: 256 * 1024,  # 256KB
    7: 192 * 1024,  # 192KB
    8: 192 * 1024,  # 192KB
}


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: Optional[int] = None,
    ) -> None:
        if max_size is None:
            max_size = 16 * 1024 * 1024  # default to 16MB

        self.max_size = max_size
        self.group = group
        self.disabled = True
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.threshold = THRESHOLD_2_SHOT_MAP[self.world_size]

        full_nvlink = can_use_custom_all_reduce_with_nvlink(
            group=group,
            device=device,
            supported_world_size=list(THRESHOLD_2_SHOT_MAP.keys()),
            cls_name="CustomAllReduceV2",
        )
        if full_nvlink != True:
            return
        from sglang.jit_kernel.all_reduce import get_custom_all_reduce_cls

        MAX_GRAPH_INPUTS = 131072
        cls = get_custom_all_reduce_cls()
        self.obj = cls(self.rank, self.world_size, self.max_size, MAX_GRAPH_INPUTS)
        self._post_init_obj()
        self.disabled = False
        log_info_on_rank0(logger, "Custom allreduce v2 initialized successfully")

    def override_shot(self, shot: int):
        self.threshold = INF if shot == 1 else 0

    @contextmanager
    def capture(self):
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
        if not self.disabled:
            # cannot call when graph is capturing
            assert (
                torch.cuda.is_current_stream_capturing() == False
            ), "Cannot register graph inputs while capturing CUDA graph"
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
        if is_in_piecewise_cuda_graph():  # disable inplace optimization
            try:
                self.obj.set_cuda_graph_capture(False)
                return self._all_reduce(input)
            finally:
                self.obj.set_cuda_graph_capture(True)
        return self._all_reduce(input)

    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self.obj.free(self.group)

    def _all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the actual all-reduce via JIT kernel."""
        input_bytes = input.numel() * input.element_size()
        shots = 1 if input_bytes <= self.threshold else 2
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

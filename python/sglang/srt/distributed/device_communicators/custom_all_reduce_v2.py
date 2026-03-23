import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.jit_kernel.all_reduce import AllReduceAlgo, get_custom_all_reduce_cls
from sglang.srt.distributed import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.utils import is_sm100_supported, log_info_on_rank0

logger = logging.getLogger(__name__)

T = TypeVar("T")

INF = 1 << 60


@dataclass(frozen=True)
class ModeConfig:
    one_shot_push_threshold: int  # below this, use one-shot push
    one_shot_pull_threshold: int  # below this, use one-shot pull


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
    ) -> None:
        _init_config()
        self.disabled = True
        full_nvlink = can_use_custom_all_reduce_with_nvlink(
            group=group,
            device=device,
            supported_world_size=list(THRESHOLD_2_SHOT_MAP.keys()),
            cls_name="CustomAllReduceV2",
        )
        if full_nvlink != True:
            return

        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.override_shot(None)
        if max_pull_size is None:
            max_pull_size = 16 * 1024 * 1024  # default to 16MB
        if max_push_size is None:
            max_push_size = self.config.one_shot_push_threshold
        max_push_size = min(max_push_size, max_pull_size)
        self.max_pull_size = max_pull_size
        self.max_push_size = max_push_size
        self.override_algo: Optional[AllReduceAlgo] = None
        self.obj = get_custom_all_reduce_cls()(
            rank=self.rank,
            world_size=self.world_size,
            pull_buffer_bytes=self.max_pull_size,
            push_buffer_bytes=self.max_push_size,
            graph_input_count=131072,
        )
        self._post_init_obj()
        self.disabled = False
        log_info_on_rank0(logger, "Custom allreduce v2 initialized successfully")

    def override_shot(self, shot: int | None):
        if shot is None:
            self.config = THRESHOLD_2_SHOT_MAP[self.world_size]
        else:
            assert shot in (1, 2)
            threshold = INF if shot == 1 else 0
            self.config = replace(self.config, one_shot_pull_threshold=threshold)

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
        return inp_size <= self.max_pull_size

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
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
        algo = self._determine_algo(input)
        return torch.from_dlpack(self.obj.all_reduce(input, algo))

    def _determine_algo(self, input: torch.Tensor) -> AllReduceAlgo:
        if self.override_algo is not None:
            return self.override_algo
        input_bytes = input.numel() * input.element_size()
        if input_bytes <= self.config.one_shot_push_threshold:
            return AllReduceAlgo.ONE_SHOT_PUSH
        if input_bytes <= self.config.one_shot_pull_threshold:
            return AllReduceAlgo.ONE_SHOT_PULL
        else:
            return AllReduceAlgo.TWO_SHOT_PULL

    def _post_init_obj(self):
        handles = [self.obj.share_storage()]
        result = self._share_list(handles)
        assert all(len(r) == 1 for r in result)
        result = [h[0] for h in result]
        self.obj.post_init(result)

    def _share_list(self, input: List[T]) -> List[List[T]]:
        input_tensor = torch.tensor(input, dtype=torch.int64, device="cpu")
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.group)
        return [g.tolist() for g in gather_list]

    def __del__(self):
        self.close()


def _init_config():
    global THRESHOLD_2_SHOT_MAP
    KB, MB = 1024, 1024 * 1024

    if is_sm100_supported():
        # NOTE: This result is based on benchmarks on B200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(4 * MB, INF),
            3: ModeConfig(4 * MB, 4 * MB),
            4: ModeConfig(2 * MB, 2 * MB),
            5: ModeConfig(2 * MB, 2 * MB),
            6: ModeConfig(1 * MB, 1 * MB),
            7: ModeConfig(896 * KB, 896 * KB),
            8: ModeConfig(720 * KB, 720 * KB),
        }
    else:
        # NOTE: This result is based on benchmarks on H200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(2 * MB, INF),
            3: ModeConfig(512 * KB, 512 * KB),
            4: ModeConfig(384 * KB, 256 * KB),
            5: ModeConfig(256 * KB, 256 * KB),
            6: ModeConfig(192 * KB, 192 * KB),
            7: ModeConfig(192 * KB, 192 * KB),
            8: ModeConfig(160 * KB, 160 * KB),
        }
    # TODO: tune on more GPUs, e.g A100


THRESHOLD_2_SHOT_MAP: Dict[int, ModeConfig] = {}

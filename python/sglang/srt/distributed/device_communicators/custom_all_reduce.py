# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/distributed/device_communicators/custom_all_reduce.py
import ctypes
import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Callable, List, Optional, TypeVar, Union

import pynvml
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

from sglang.srt import _custom_ops as ops
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    gpu_p2p_access_check,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import cuda_device_count_stateless, is_cuda

try:
    import sgl_kernel

    custom_ar = True
except Exception:
    # For AMD GPUs and CPUs
    custom_ar = False

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


@with_nvml_context
def is_full_nvlink(physical_device_ids: List[int]) -> bool:
    """
    query if the set of gpus are fully connected by nvlink (1 hop)
    """
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
    for i, handle in enumerate(handles):
        for j, peer_handle in enumerate(handles):
            if i < j:
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                    )
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return False
                except pynvml.NVMLError:
                    logger.exception(
                        "NVLink detection failed. This is normal if your"
                        " machine has no NVLink equipped."
                    )
                    return False
    return True


def _can_p2p(rank: int, world_size: int) -> bool:
    # SGLANG_SKIP_P2P_CHECK can be set to False in sglang
    SGLANG_SKIP_P2P_CHECK = os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1"
    for i in range(world_size):
        if i == rank:
            continue
        if SGLANG_SKIP_P2P_CHECK:
            logger.info("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=8192 * 1024,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        self.group = group

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

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu") for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        assert is_cuda()

        full_nvlink = is_full_nvlink(physical_device_ids)
        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        if not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        # From TensorRT-LLM getMaxRequiredWorkspaceSize
        self.max_required_workspace_size = [16 * 1000 * 1000, 8 * 1000 * 1000]

        # sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
        self.barrier_max_size = 8 * (24 + 2) * 8

        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        self.barrier_in_ptrs = self.create_shared_buffer(
            self.barrier_max_size, group=group
        )
        self.barrier_out_ptrs = self.create_shared_buffer(
            self.barrier_max_size, group=group
        )

        self._ptr = ops.init_custom_ar(
            rank,
            world_size,
            self.buffer_ptrs,
            self.barrier_in_ptrs,
            self.barrier_out_ptrs,
        )
        self.disabled = False

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2:
            return (
                inp_size < self.max_size
                and inp_size < self.max_required_workspace_size[0]
            )

        if self.full_nvlink:
            return (
                inp_size < self.max_size
                and inp_size < self.max_required_workspace_size[1]
            )

        return False

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                output = torch.empty_like(input)
                ops.all_reduce(self._ptr, input, output)
                return output
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            output = torch.empty_like(input)
            ops.all_reduce(self._ptr, input, output)
            return output

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            self.free_shared_buffer(self.buffer_ptrs)
            self.free_shared_buffer(self.barrier_in_ptrs)
            self.free_shared_buffer(self.barrier_out_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()

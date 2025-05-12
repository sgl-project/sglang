# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from typing import List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _can_p2p,
    is_full_nvlink,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


try:
    import sgl_kernel

    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def support_custom_all_to_all():
    try:
        from sgl_kernel import all_to_all

        return True
    except Exception:
        return False


class CustomAlltoAll:
    # todo: multi-node all to all
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
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
        if not _is_cuda:
            logger.warning("Custom alltoall is disabled because cuda is not available.")
            return
        if not support_custom_all_to_all():
            logger.warning(
                "Custom alltoall is not available, please use the latest sgl-kernel."
            )
            return

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "Custom alltoall should be attached to a non-NCCL group."

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom alltoall is disabled because this process group"
                " spans across nodes."
            )
            return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom alltoall for single GPU case.
            return

        if world_size not in CustomAlltoAll._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom alltoall is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s.",
                world_size,
                str(CustomAlltoAll._SUPPORTED_WORLD_SIZES),
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
            device_ids = list(range(torch.cuda.device_count()))

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
        full_nvlink = is_full_nvlink(physical_device_ids, world_size)

        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom alltoall is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly."
            )
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not _can_p2p(rank, world_size):
            logger.warning(
                "Custom alltoall is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return
        self.peer_output_size = 256  # (3+2*8)*size8, other reserved
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate alltoall results.
        self.meta_ptrs = self.create_shared_buffer(
            sgl_kernel.meta_size() + self.peer_output_size, group=group
        )
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self._ptr = sgl_kernel.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.full_nvlink
        )
        self.disabled = False
        self.buffer_ptrs = {}

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

    def register_output_buffer(self, output: torch.Tensor, buffer_tag: str):
        lib = CudaRTLibrary()
        pointer = output.untyped_storage().data_ptr()
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=self.group)
        rank = dist.get_rank(group=self.group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=self.group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore
        sgl_kernel.register_buffer(self._ptr, pointers)
        assert buffer_tag not in self.buffer_ptrs
        self.buffer_ptrs[buffer_tag] = pointers

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        return True

    def check_inputs(
        self,
        output: torch.Tensor,  # out_dim0, head_dim
        input: torch.Tensor,  # in_dim0, head_dim
        output_split_sizes: Union[torch.Tensor, List[int]],
        input_split_sizes: Union[torch.Tensor, List[int]],
        chunk_size: int,
    ):
        """Performs an out-of-place all to all.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if isinstance(output_split_sizes, list):
            output_split_sizes = torch.tensor(
                output_split_sizes, dtype=torch.int64, device=input.device
            )
        else:
            output_split_sizes = output_split_sizes.to(torch.int64)
        if isinstance(input_split_sizes, list):
            input_split_sizes = torch.tensor(
                input_split_sizes, dtype=torch.int64, device=input.device
            )
        else:
            input_split_sizes = input_split_sizes.to(torch.int64)
        assert (
            len(output_split_sizes.shape) == 1
            and output_split_sizes.shape[0] == self.world_size
        ), f"output_split_sizes shape {output_split_sizes.shape} should be world_size {self.world_size}"
        assert (
            len(input_split_sizes.shape) == 1
            and input_split_sizes.shape[0] == self.world_size
        ), f"input_split_sizes shape {output_split_sizes.shape} should be world_size {self.world_size}"
        assert (
            len(output.shape) == 3
        ), f"output must be a 3D(dim0,dim1,chunk_size) tensor, got {output.shape}"
        assert (
            len(input.shape) == 3
        ), f"input must be a 3D(dim0,dim1,chunk_size) tensor, got {input.shape}"
        assert (
            input.shape[-1] == chunk_size
        ), f"input last dim should be chunk_size {chunk_size}, got shape {input.shape}"
        assert (
            output.shape[-1] == chunk_size
        ), f"output last dim should be chunk_size {chunk_size}, got shape {output.shape}"
        return output_split_sizes, input_split_sizes

    @torch.compiler.disable
    def custom_all_to_all(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        plan_meta: torch.Tensor,
        buffer_tag: str,
    ):
        if self.disabled:
            return
        if self.disabled or not self.should_custom_ar(input):
            return None
        assert buffer_tag in self.buffer_ptrs, "buffer_ptrs is not initialized"
        buffer_ptrs = self.buffer_ptrs[buffer_tag]
        sgl_kernel.all_to_all(
            self._ptr, output, input, plan_meta, buffer_ptrs[self.rank]
        )

    def custom_all_to_all_plan(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: Union[torch.Tensor, List[int]],
        input_split_sizes: Union[torch.Tensor, List[int]],
        chunk_size: int,
        output_split_offsets: Optional[torch.Tensor],
        input_split_offsets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.disabled or not self.should_custom_ar(input):
            return
        output_split_sizes, input_split_sizes = self.check_inputs(
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            chunk_size,
        )
        # 256,16,576; 64,64,512; 32,128,512
        meta_extra = (input.numel() * input.dtype.itemsize // 512 + 7) // 8
        plan_meta = torch.zeros(
            128 + meta_extra, dtype=torch.int64, device=input.device
        )
        sgl_kernel.all_to_all_plan(
            self._ptr,
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            chunk_size,
            output_split_offsets,
            input_split_offsets,
            plan_meta,
        )
        return plan_meta

    def close(self):
        if not self.disabled and self._ptr:
            if sgl_kernel is None:
                return
            sgl_kernel.dispose(self._ptr)
            if _is_cuda:
                self.free_shared_buffer(self.meta_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()

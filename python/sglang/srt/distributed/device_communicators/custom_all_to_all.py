# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Callable, List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing_extensions import ParamSpec

from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _can_p2p,
    is_full_nvlink,
    is_weak_contiguous,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.utils import cuda_device_count_stateless, is_cuda, is_hip

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
        from sgl_kernel import custom_all_to_all

        return True
    except Exception:
        return False


class CustomAlltoAll:
    # todo: multi-node all to all
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    # tp2dp: 2048 max chunk size per dp, dp heads 128//dp_mla_tp_size, head_dim(576), type size
    # dp2tp: dp_size*2048 max chunk size per dp, tp heads 128/tp_size=16, head_dim(512), type size
    # todo: larger chunked-prefill-size, per dp chunk size = chunked-prefill-size/dp_size,
    #  currently chunked-prefill-size should <= 2048*8(dp=8) or 4096*4(dp=4)
    _MAX_CAR_SIZE = 2048 * 128 * 576 * 2

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
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
        self.peer_output_size = 128  # 8*8*2
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        # sizeof(uint32_t) * (MAX_ALL_TO_ALL_BLOCKS + 2) * MAX_RANKS_PER_NODE;
        self.barrier_max_size = 8 * (64 + 2) * 8

        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        self.buffer_meta_ptrs = self.create_shared_buffer(
            self.peer_output_size, group=group
        )
        self.rank_data_base = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.barrier_in_ptrs = self.create_shared_buffer(
            self.barrier_max_size, group=group
        )
        self.barrier_out_ptrs = self.create_shared_buffer(
            self.barrier_max_size, group=group
        )

        self._ptr = sgl_kernel.init_custom_reduce(
            rank,
            world_size,
            self.rank_data_base,
            self.buffer_ptrs,
            self.buffer_meta_ptrs,
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
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = sgl_kernel.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        sgl_kernel.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(
        self,
        inp: torch.Tensor,
        input_split_sizes: Union[torch.Tensor, List[int]],
        input_split_offsets: Union[torch.Tensor, List[int]],
    ):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.full_nvlink:
            if inp_size >= self.max_size:
                if input_split_offsets is None:
                    block_count = (
                        sum(input_split_sizes)
                        if isinstance(input_split_sizes, list)
                        else torch.sum(input_split_sizes)
                    )
                else:
                    block_count = (
                        max(
                            [
                                off + len
                                for off, len in zip(
                                    input_split_offsets, input_split_sizes
                                )
                            ]
                        )
                        if isinstance(input_split_offsets, list)
                        else torch.max(input_split_offsets + input_split_sizes)
                    )
                inp_size = block_count * inp[0].numel() * inp.element_size()
            return inp_size < self.max_size

        return False

    def check_inputs(
        self,
        output: torch.Tensor,  # out_dim0, head_dim
        input: torch.Tensor,  # in_dim0, head_dim
        output_split_sizes: Union[torch.Tensor, List[int]],
        input_split_sizes: Union[torch.Tensor, List[int]],
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
        ), f"output_split_sizes must be a list  or a tensor of length world_size"
        assert (
            len(input_split_sizes.shape) == 1
            and input_split_sizes.shape[0] == self.world_size
        ), f"input_split_sizes must be a list  or a tensor of length world_size"
        assert len(output.shape) == 2, f"output must be a 2D tensor, got {output.shape}"
        assert len(input.shape) == 2, f"input must be a 2D tensor, got {input.shape}"
        assert (
            input.shape[-1] == output.shape[-1]
        ), f"output and input must have the same last dimension, got {input.shape} and {output.shape}"
        return output_split_sizes, input_split_sizes

    def custom_all_to_all(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        plan_meta: torch.Tensor,
    ):
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled:
            return
        block_size = input.shape[-1]
        sgl_kernel.custom_all_to_all(
            self._ptr,
            output,
            input,
            plan_meta,
            block_size,
        )

    def custom_all_to_all_plan(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: Union[torch.Tensor, List[int]],
        input_split_sizes: Union[torch.Tensor, List[int]],
        output_split_offsets: Optional[torch.Tensor],
        input_split_offsets: Optional[torch.Tensor],
        plan_meta: torch.Tensor,
    ) -> torch.Tensor:
        if self.disabled or not self.should_custom_ar(
            input, input_split_sizes, input_split_offsets
        ):
            return
        output_split_sizes, input_split_sizes = self.check_inputs(
            output, input, output_split_sizes, input_split_sizes
        )
        block_size = input.shape[-1]
        sgl_kernel.custom_all_to_all_plan(
            self._ptr,
            output,
            input,
            output_split_sizes,
            input_split_sizes,
            output_split_offsets,
            input_split_offsets,
            plan_meta,
            block_size,
        )

    def close(self):
        if not self.disabled and self._ptr:
            sgl_kernel.custom_dispose(self._ptr)
            if _is_cuda:
                self.free_shared_buffer(self.buffer_ptrs)
                self.free_shared_buffer(self.buffer_meta_ptrs)
                self.free_shared_buffer(self.barrier_in_ptrs)
                self.free_shared_buffer(self.barrier_out_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()

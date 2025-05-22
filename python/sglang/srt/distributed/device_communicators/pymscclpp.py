import logging
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, List, Optional, TypeVar, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt import _custom_ops as ops
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

mscclpp_is_available = False
if _is_hip:
    # TODO(zyksir): mscclpp is untested on AMD and therefore disabled.
    mscclpp_is_available = False
if _is_cuda:
    try:
        import sgl_kernel

        mscclpp_is_available = True
    except:
        mscclpp_is_available = False


class MscclContextSelection(IntEnum):
    MSCCL1SHOT1NODELL = 1
    MSCCL1SHOT2NODELL = 2


def mscclpp_is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


class PyMscclppCommunicator:
    _SUPPORTED_WORLD_SIZES = [8, 16]
    _MAX_CAR_SIZE = 1024 * 1024
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

    # max_size: max supported mscclpp allreduce size
    # in A100 mscclpp is faster than nccl only under condition of msg size smaller than1MB
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

        if not mscclpp_is_available:
            # disable because of missing mscclpp library
            # e.g. in a non-cuda environment
            return

        self.group = group

        assert (
            dist.get_backend(group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize mscclpp for single GPU case.
            return

        if world_size not in PyMscclppCommunicator._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "PyMscclpp is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_mscclpp=True explicitly.",
                world_size,
                str(PyMscclppCommunicator._SUPPORTED_WORLD_SIZES),
            )
            return

        self.ranks = torch.distributed.get_process_group_ranks(group)
        torch.cuda.device_count()
        # for now mscclpp with stride in the communicator is not tested
        if not (abs(self.ranks[-1] - self.ranks[0]) == world_size - 1):
            logger.warning(
                "PyMscclpp is disabled due to an unsupported group %s."
                "Please ensure all ranks in the group are consecutive."
                "To silence this warning, specify disable_mscclpp=True explicitly.",
                str(self.ranks),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size

        if dist.get_rank(group) == 0:
            unique_id = [ops.mscclpp_generate_unique_id()]
        else:
            unique_id = [None]
        dist.broadcast_object_list(unique_id, src=self.ranks[0], group=self.group)
        self.unique_id = unique_id[0]
        self.rank_to_node, self.rank_to_ib = list(range(world_size)), list(
            range(world_size)
        )
        for r in range(world_size):
            self.rank_to_node[r] = r // 8
            self.rank_to_ib[r] = r % 8

        self._context = None
        if not _is_hip:
            self.scratch = torch.empty(
                PyMscclppCommunicator._MAX_CAR_SIZE * 8,
                dtype=torch.uint8,
                device=self.device,
            )
            self.put_buffer = torch.empty(
                PyMscclppCommunicator._MAX_CAR_SIZE * 8 // 8,
                dtype=torch.uint8,
                device=self.device,
            )
            if world_size == 8:
                selection = int(MscclContextSelection.MSCCL1SHOT1NODELL)
            elif world_size == 16:
                selection = int(MscclContextSelection.MSCCL1SHOT2NODELL)
            self._context = ops.mscclpp_init_context(
                self.unique_id,
                self.rank,
                self.world_size,
                self.scratch,
                self.put_buffer,
                self.rank_to_node,
                self.rank_to_ib,
                selection,
            )
        else:
            raise NotImplementedError("HIP Mscclpp is not supported yet.")

        # PyMscclpp is enabled only in cuda graph
        self.disabled = True

    def post_process_graph_input(self) -> bool:
        return
        # TODO: find the best thread and block size using this config
        # for tensor_dtype, tensor_numel in self.graph_input_set:
        #     if (tensor_dtype, tensor_numel) not in self.allreduce_algo.msg_sz2param:
        #         randn_input = torch.randn((tensor_numel, ), dtype=tensor_dtype, device=self.device)
        #         best_config = find_best_config(randn_input, self.allreduce_algo)
        #         objects = [best_config] if self.rank == 0 else [None]
        #         dist.broadcast_object_list(objects, src=0, group=self.cpu_group)
        #         best_config = objects[0]
        #         self.allreduce_algo.msg_sz2param[(tensor_dtype, tensor_numel)] = best_config

    def should_msccl_allreduce(
        self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ) -> bool:
        if self.disabled or self._context is None:
            return False
        if inp.dtype not in PyMscclppCommunicator._SUPPORTED_DTYPE:
            return False
        if not mscclpp_is_weak_contiguous(inp):
            return False
        # only support sum op
        if op != ReduceOp.SUM:
            return False
        if inp.numel() * inp.element_size() > self.max_size:
            return False
        return True

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                self.graph_input_set.add((tensor.dtype, tensor.numel()))
        result = torch.empty_like(tensor)
        ops.mscclpp_allreduce(self._context, tensor, result)
        return result

    @contextmanager
    def change_state(
        self,
        enable: Optional[bool] = None,
    ):
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable

        yield

        self.disabled = old_disable
        if enable:
            self.post_process_graph_input()

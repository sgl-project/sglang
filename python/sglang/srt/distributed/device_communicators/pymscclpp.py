import bisect
import logging
import math
import os
from contextlib import contextmanager
from enum import IntEnum
from typing import Optional, Union

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
        import sgl_kernel  # noqa: F401

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


def mscclpp_convert_to_bytes(size_str):
    """
    Converts a human-readable size string (e.g., "1MB", "2.5kb", "3 GB")
    into the equivalent number of bytes using binary units.

    Args:
        size_str (str): A string representing size with unit (KB, MB, GB).

    Returns:
        int: Number of bytes.
    """
    size_str = size_str.strip().lower()

    if not size_str:
        raise ValueError("Empty input string")

    # Extract numeric part and unit
    for i in range(len(size_str)):
        if not size_str[i].isdigit() and size_str[i] != ".":
            break
    num_str = size_str[:i]
    unit = size_str[i:].strip()

    try:
        num = float(num_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value in '{size_str}'")

    # Conversion factors
    if unit == "b":
        return int(num)
    elif unit == "kb":
        return int(num * 1024)
    elif unit == "mb":
        return int(num * 1024 * 1024)
    elif unit == "gb":
        return int(num * 1024 * 1024 * 1024)
    else:
        raise ValueError(f"Unsupported unit: {unit}, support B, KB, MB, GB only")


def mscclpp_bench_time(func, test_niter: int = 10, warmup_niter: int = 2):
    # warmup
    for _ in range(warmup_niter):
        func()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start_event.record()
    for _ in range(test_niter):
        func()
    end_event.record()
    end_event.synchronize()
    func_cost_us = start_event.elapsed_time(end_event) / test_niter * 1000
    return func_cost_us


class PyMscclppCommunicator:
    _SUPPORTED_WORLD_SIZES = [8, 16]
    _MAX_BYTES = mscclpp_convert_to_bytes(os.getenv("SGLANG_MSCCLPP_MAX_BYTES", "1MB"))
    _SUPPORTED_DTYPE = [torch.float, torch.float16, torch.bfloat16]

    # max_bytes: max supported mscclpp allreduce size
    # in A100 mscclpp is faster than nccl only under condition of msg size smaller than1MB
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_bytes=_MAX_BYTES,
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
        self.nranks_per_node = torch.cuda.device_count()
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

        self.max_bytes = max_bytes
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
            self.rank_to_ib[r] = self.rank % 8

        self._context = None
        self.context_selection = None
        self.msg_size_for_finetune = [
            2**i for i in range(10, math.floor(math.log2(self.max_bytes)) + 1)
        ]
        self.msg_size2best_config = {}
        if world_size == 8:
            self.context_selection = MscclContextSelection.MSCCL1SHOT1NODELL
        elif world_size == 16:
            self.context_selection = MscclContextSelection.MSCCL1SHOT2NODELL
        if not _is_hip:
            self.scratch = torch.empty(
                self.max_bytes * 8,
                dtype=torch.uint8,
                device=self.device,
            )
            self.put_buffer = torch.empty(
                self.max_bytes * 8 // self.nranks_per_node,
                dtype=torch.uint8,
                device=self.device,
            )
            self._context = ops.mscclpp_init_context(
                self.unique_id,
                self.rank,
                self.world_size,
                self.scratch,
                self.put_buffer,
                self.nranks_per_node,
                self.rank_to_node,
                self.rank_to_ib,
                int(self.context_selection),
            )
        else:
            raise NotImplementedError("HIP Mscclpp is not supported yet.")

        self.msg_size2best_config = {}
        self.pre_tune_config()
        if dist.get_rank(group) == 0:
            msg_size2best_config = [self.msg_size2best_config]
        else:
            msg_size2best_config = [None]
        dist.broadcast_object_list(
            msg_size2best_config, src=self.ranks[0], group=self.group
        )
        self.msg_size2best_config = msg_size2best_config[0]

        # PyMscclpp is enabled only in cuda graph
        self.disabled = True

    def pre_tune_config(self, dtype=torch.bfloat16) -> bool:
        logger.debug(f"start to pre-tune configs for rank {self.rank}")
        nthreads_to_try = [256, 512, 1024]
        nblocks_to_try = [21, 42, 84]
        inp_randn = torch.ones(
            self.msg_size_for_finetune[-1] // dtype.itemsize, dtype=dtype, device="cuda"
        )
        oup_randn = torch.empty_like(inp_randn)
        for msg_size in self.msg_size_for_finetune:
            mock_inp, mock_outp = (
                inp_randn[: msg_size // dtype.itemsize],
                oup_randn[: msg_size // dtype.itemsize],
            )
            best_config, best_time = None, None
            for nthreads in nthreads_to_try:
                for nblocks in nblocks_to_try:
                    cur_cost = mscclpp_bench_time(
                        lambda: ops.mscclpp_allreduce(
                            self._context, mock_inp, mock_outp, nthreads, nblocks
                        )
                    )
                    if best_time is None or cur_cost < best_time:
                        best_config = (nthreads, nblocks)
                        best_time = cur_cost
            self.msg_size2best_config[msg_size] = best_config
            if self.rank == 0:
                logger.debug(
                    f"for msg_size {msg_size}, best_config: {best_config}, best_time: {best_time}us"
                )

    def should_mscclpp_allreduce(
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
        if inp.numel() * inp.element_size() > self.max_bytes:
            return False
        return True

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                self.graph_input_set.add((tensor.dtype, tensor.numel()))
        msg_size = tensor.numel() * tensor.itemsize
        index = bisect.bisect_left(self.msg_size_for_finetune, msg_size)
        msg_size_finetune = self.msg_size_for_finetune[index]
        nthreads, nblocks = self.msg_size2best_config[msg_size_finetune]
        result = torch.empty_like(tensor)
        ops.mscclpp_allreduce(self._context, tensor, result, nthreads, nblocks)
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

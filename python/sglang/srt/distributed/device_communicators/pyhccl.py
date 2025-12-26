# Adapted from https://github.com/vllm-project/vllm-ascend/blob/v0.11.0-dev/vllm_ascend/distributed/device_communicators/pyhccl.py

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary,
    aclrtStream_t,
    buffer_type,
    current_stream,
    hcclComm_t,
    hcclDataTypeEnum,
    hcclRedOpTypeEnum,
    hcclUniqueId,
)
from sglang.srt.distributed.utils import StatelessProcessGroup

logger = logging.getLogger(__name__)


class PyHcclCommunicator:

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the PyHcclCommunicator to. If None,
                it will be bind to f"npu:{local_rank}".
            library_path: the path to the HCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """

        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.HCCL
            ), "PyHcclCommunicator should be attached to a non-HCCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        try:
            self.hccl = HCCLLibrary(library_path)
        except Exception:
            # disable because of missing HCCL library
            # e.g. in a non-NPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("vLLM is using pyhccl")

        if isinstance(device, int):
            device = torch.device(f"npu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        if self.rank == 0:
            # get the unique id from HCCL
            with torch.npu.device(device):
                self.unique_id = self.hccl.hcclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = hcclUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        # hccl communicator and stream will use this device
        # `torch.npu.device` is a context manager that changes the
        # current npu device to the specified one
        with torch.npu.device(device):
            self.comm: hcclComm_t = self.hccl.hcclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def all_reduce(
        self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # hccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.hccl.hcclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            hcclDataTypeEnum.from_torch(in_tensor.dtype),
            hcclRedOpTypeEnum.from_torch(op),
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )
        return out_tensor

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        if src == self.rank:
            buffer = buffer_type(tensor.data_ptr())
        else:
            buffer = buffer_type(tensor.data_ptr())
        self.hccl.hcclBroadcast(
            buffer,
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )

    def all_gather(
        self, out_tensor: torch.Tensor, in_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        assert out_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {out_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        self.hccl.hcclAllGather(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            hcclDataTypeEnum.from_torch(in_tensor.dtype),
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )

    def reduce_scatter(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        assert in_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        assert out_tensor.device == self.device, (
            f"this hccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {out_tensor.device}"
        )

        if stream is None:
            stream = current_stream()

        self.hccl.hcclReduceScatter(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            out_tensor.numel(),
            hcclDataTypeEnum.from_torch(in_tensor.dtype),
            hcclRedOpTypeEnum.from_torch(op),
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )

    def barrier(self, stream=None):
        if self.disabled:
            return
        if stream is None:
            stream = current_stream()

        self.hccl.hcclBarrier(
            self.comm,
            aclrtStream_t(stream.npu_stream),
        )

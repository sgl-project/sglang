# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import logging
from typing import Optional, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils.common import get_current_device_stream_fast

from .base import AllReduceMode, BaseCommunicator
from .pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)

logger = logging.getLogger(__name__)


class PyNcclCommunicator(BaseCommunicator):
    name = "pynccl"

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
            device: the device to bind the PyNcclCommunicator to. If None,
                it will be bind to f"cuda:{local_rank}".
            library_path: the path to the NCCL library. If None, it will
                use the default library path.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert (
                dist.get_backend(group) != dist.Backend.NCCL
            ), "PyNcclCommunicator should be attached to a non-NCCL group."
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size
        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        super().__init__(world_size=self.world_size, disabled=True)
        self.group = group

        # if world_size == 1, no need to create communicator
        assert self.world_size > 1

        self.nccl = NCCLLibrary(library_path)
        self.available = True
        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            logger.info("sglang is using nccl==%s", self.nccl.ncclGetVersion())

        if self.rank == 0:
            # get the unique id from NCCL
            self.unique_id = self.nccl.ncclGetUniqueId()
        else:
            # construct an empty unique id
            self.unique_id = ncclUniqueId()

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
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # nccl communicator and stream will use this device
        # `torch.cuda.device` is a context manager that changes the
        # current cuda device to the specified one
        with torch.cuda.device(device), self.change_state(enable=True):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )
            warmup_stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            with torch.cuda.stream(warmup_stream):
                data = torch.zeros(1, device=device)
                self.all_reduce(data)
            warmup_stream.synchronize()
            del data

    def should_use_custom_op(self) -> bool:
        return True

    @property
    def disabled(self) -> bool:
        # TODO(dark): this is temporary work around
        return self._disabled and not is_in_tc_piecewise_cuda_graph()

    def _resolve_stream(self) -> torch.cuda.Stream:
        """Return the current device stream used for NCCL calls."""
        return get_current_device_stream_fast()

    def graph_capture_context(self):
        return self.change_state(enable=True)

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        return AllReduceMode.BOTH

    @BaseCommunicator.validate
    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_.device}"
        )
        output = input_
        if inplace is False:  # default to inplace
            output = torch.empty_like(input_)

        stream = self._resolve_stream()
        self.nccl.ncclAllReduce(
            buffer_type(input_.data_ptr()),
            buffer_type(output.data_ptr()),
            input_.numel(),
            ncclDataTypeEnum.from_torch(input_.dtype),
            ncclRedOpTypeEnum.from_torch(ReduceOp.SUM),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return output

    @BaseCommunicator.validate
    def all_gather_impl(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: Optional[list[int]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = self._resolve_stream()

        if sizes is not None:
            split_offset = 0

            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                dst_slice = output_tensor[split_offset : split_offset + split_size]
                self.nccl.ncclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(dst_slice.data_ptr()),
                    dst_slice.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            self.nccl.ncclAllGather(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                input_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

    def all_gather_into_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = self.allocate_all_gather(input_)
        self.all_gather_impl(out, input_)
        return out

    def cp_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        """
        Currently, it is mainly used in context parallelism,
        primarily leveraging pynccl to implement non-blocking allgather communication.
        """
        with self.change_state(enable=True):
            return self.all_gather_impl(output_tensor, input_tensor, stream=stream)

    @BaseCommunicator.validate
    def reduce_scatter_impl(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        sizes: Optional[list[int]] = None,
    ) -> None:
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        stream = self._resolve_stream()

        if sizes is not None:
            split_offset = 0
            self.nccl.ncclGroupStart()
            for root, split_size in enumerate(sizes):
                chunk = input_tensor[split_offset : split_offset + split_size, ...]

                self.nccl.ncclReduce(
                    buffer_type(chunk.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    chunk.numel(),
                    ncclDataTypeEnum.from_torch(input_tensor.dtype),
                    ncclRedOpTypeEnum.from_torch(op),
                    root,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                split_offset += split_size
            self.nccl.ncclGroupEnd()
        else:
            self.nccl.ncclReduceScatter(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                output_tensor.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )

    def reduce_scatter_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = self.allocate_reduce_scatter(input_)
        self.reduce_scatter_impl(out, input_)
        return out

    @BaseCommunicator.validate
    def send(self, tensor: torch.Tensor, dst: int):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    @BaseCommunicator.validate
    def recv(self, tensor: torch.Tensor, src: int):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    @BaseCommunicator.validate
    def broadcast(self, tensor: torch.Tensor, src: int):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()

        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            # NCCL requires the sender also to have a receive buffer
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())
        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def register_comm_window_raw(self, ptr: int, size: int):
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

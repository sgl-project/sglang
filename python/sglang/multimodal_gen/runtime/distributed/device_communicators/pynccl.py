# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/pynccl.py

# ===================== import region =====================
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from sglang.multimodal_gen.runtime.distributed.utils import StatelessProcessGroup
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import current_stream

logger = init_logger(__name__)


class PyNcclCommunicator:

    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
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

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return
        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            # e.g. in a non-GPU environment
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        logger.info("sglang-diffusion is using nccl==%s", self.nccl.ncclGetVersion())

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
        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=device)
            self.all_reduce(data)
            if stream is not None:
                stream.synchronize()
            del data

    def all_reduce(
        self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None
    ) -> torch.Tensor:
        if self.disabled:
            return None
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    def all_to_all_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        output_split_sizes: list[int] | None = None,
        input_split_sizes: list[int] | None = None,
        stream=None,
    ) -> None:
        """Equal-split all-to-all, the dist.all_to_all_single equivalent.

        Exists because ProcessGroupNCCL's collectives cannot be recorded into a
        CUDA graph: their host-side per-op bookkeeping advances once at capture,
        so replays leave the ranks disagreeing about which collective is in
        flight and they hang. Raw ncclSend/ncclRecv inside a group carries no
        such state, so a captured region can hold the exchange.
        """
        if self.disabled:
            raise RuntimeError(
                "pynccl all_to_all_single called while the communicator is "
                "disabled; wrap it in change_state(enable=True)"
            )
        assert output.dtype == input_.dtype, (output.dtype, input_.dtype)
        assert input_.is_contiguous() and output.is_contiguous()
        if input_split_sizes is None and output_split_sizes is None:
            assert output.numel() == input_.numel(), (
                output.numel(),
                input_.numel(),
            )
            assert input_.numel() % self.world_size == 0, (
                f"all_to_all_single without split sizes needs an equal split, "
                f"got {input_.numel()} elements over {self.world_size} ranks"
            )
        if stream is None:
            stream = current_stream()
        # dist.all_to_all_single defines split sizes along dim 0; convert rows
        # to element counts so n-D tensors split identically to torch
        in_row = input_.numel() // input_.size(0) if input_.dim() else 1
        out_row = output.numel() // output.size(0) if output.dim() else 1
        chunk = input_.numel() // self.world_size
        if input_split_sizes is None:
            send_counts = [chunk] * self.world_size
        else:
            assert sum(input_split_sizes) == input_.size(0)
            send_counts = [n * in_row for n in input_split_sizes]
        if output_split_sizes is None:
            recv_counts = [chunk] * self.world_size
        else:
            assert sum(output_split_sizes) == output.size(0)
            recv_counts = [n * out_row for n in output_split_sizes]
        assert len(send_counts) == len(recv_counts) == self.world_size
        send = input_.view(-1)
        recv = output.view(-1)
        dtype = ncclDataTypeEnum.from_torch(input_.dtype)
        itemsize = input_.element_size()
        send_off = recv_off = 0
        self.nccl.ncclGroupStart()
        for peer in range(self.world_size):
            self.nccl.ncclSend(
                buffer_type(send.data_ptr() + send_off * itemsize),
                send_counts[peer],
                dtype,
                peer,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            self.nccl.ncclRecv(
                buffer_type(recv.data_ptr() + recv_off * itemsize),
                recv_counts[peer],
                dtype,
                peer,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            send_off += send_counts[peer]
            recv_off += recv_counts[peer]
        self.nccl.ncclGroupEnd()

    @contextmanager
    def change_state(self, enable: bool | None = None):
        """Enable the communicator for the duration of the block.

        The graph-capture path is the only caller that needs it, so ordinary
        traffic keeps going through the process group and this stays a scoped
        override rather than a mode switch.
        """
        if enable is None:
            enable = self.available
        old_disabled = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disabled

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Protocol, Union, cast

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.utils.common import get_current_device_stream_fast

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nccl.core.communicator import Communicator as _Nccl4PyCommunicator
    from nccl.core.typing import NcclRedOp as _Nccl4PyRedOp
    from nccl.core.utils import UniqueId as _Nccl4PyUniqueId
    from nccl.core.utils import Version as _Nccl4PyVersion


class _Nccl4PyModule(Protocol):
    SUM: "_Nccl4PyRedOp"
    PROD: "_Nccl4PyRedOp"
    MAX: "_Nccl4PyRedOp"
    MIN: "_Nccl4PyRedOp"
    AVG: "_Nccl4PyRedOp"
    Communicator: type["_Nccl4PyCommunicator"]
    UniqueId: type["_Nccl4PyUniqueId"]

    def get_version(self) -> "_Nccl4PyVersion": ...

    def get_unique_id(self, empty: bool = False) -> "_Nccl4PyUniqueId": ...

    def group_start(self) -> None: ...

    def group_end(self) -> None: ...


class _NcclCommHandle:
    """Small compatibility shim for users that read ``comm.value``."""

    def __init__(self, value: int):
        self.value = value


def _import_nccl4py() -> _Nccl4PyModule:
    import nccl.core as nccl4py

    return cast(_Nccl4PyModule, nccl4py)


def _nccl4py_reduce_op(nccl4py: _Nccl4PyModule, op: ReduceOp) -> "_Nccl4PyRedOp":
    if op == ReduceOp.SUM:
        return nccl4py.SUM
    if op == ReduceOp.PRODUCT:
        return nccl4py.PROD
    if op == ReduceOp.MAX:
        return nccl4py.MAX
    if op == ReduceOp.MIN:
        return nccl4py.MIN
    if op == ReduceOp.AVG:
        return nccl4py.AVG
    raise ValueError(f"Unsupported op: {op}")


class PyNcclCommunicator:

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

        self.group = group
        self.use_nccl4py = False
        self.nccl4py = cast(_Nccl4PyModule, None)

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        if library_path is None:
            try:
                nccl4py = _import_nccl4py()
            except Exception as e:
                logger.info(
                    "nccl4py is unavailable, falling back to legacy PyNCCL: %s", e
                )
            else:
                self._init_nccl4py(nccl4py, group, device)
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
        with torch.cuda.device(device):
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

        # by default it is disabled, e.g. in profiling models and prefill phase.
        # to use it, use under `with obj.change_state(enable=True)`, usually
        # when we are using CUDA graph.
        self.disabled = True

    def _init_nccl4py(
        self,
        nccl4py: _Nccl4PyModule,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
    ) -> None:
        self.nccl4py = nccl4py
        self.use_nccl4py = True
        self.nccl = None  # type: ignore[assignment]
        self.available = True
        self.disabled = False

        version = nccl4py.get_version()
        v = version.nccl_version
        self.nccl_version = v.major * 10000 + v.minor * 100 + v.micro
        if self.rank == 0:
            logger.info(
                "sglang is using nccl4py==%s (nccl==%s)",
                version.nccl4py_version,
                version.nccl_version,
            )

        if self.rank == 0:
            self.unique_id = nccl4py.get_unique_id()
        else:
            self.unique_id = nccl4py.get_unique_id(empty=True)

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.as_bytes))
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
            self.unique_id = nccl4py.UniqueId.from_bytes(bytes(tensor.tolist()))
        else:
            unique_id_bytes = group.broadcast_obj(self.unique_id.as_bytes, src=0)
            self.unique_id = nccl4py.UniqueId.from_bytes(unique_id_bytes)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        with torch.cuda.device(device):
            self.nccl_comm: "_Nccl4PyCommunicator" = nccl4py.Communicator.init(
                nranks=self.world_size, rank=self.rank, unique_id=self.unique_id
            )
            self.comm = _NcclCommHandle(self.nccl_comm.ptr)
            warmup_stream = torch.cuda.Stream()

            # A small all_reduce for warmup.
            with torch.cuda.stream(warmup_stream):
                data = torch.zeros(1, device=device)
                self.all_reduce(data)
            warmup_stream.synchronize()
            del data

        # Keep the same default behavior as the legacy communicator: disabled
        # outside explicit graph/symmetric-memory communication contexts.
        self.disabled = True

    def _resolve_stream(self) -> torch.cuda.Stream:
        """Return the current device stream used for NCCL calls."""
        return get_current_device_stream_fast()

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        if self.disabled:
            return
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()
        if self.use_nccl4py:
            self.nccl_comm.allreduce(
                tensor,
                tensor,
                _nccl4py_reduce_op(self.nccl4py, op),
                stream=stream.cuda_stream,
            )
            return
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def outplace_all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: Optional[torch.Tensor] = None,
        op: ReduceOp = ReduceOp.SUM,
    ) -> Optional[torch.Tensor]:
        if self.disabled:
            return None
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )

        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)

        stream = self._resolve_stream()
        if self.use_nccl4py:
            self.nccl_comm.allreduce(
                in_tensor,
                out_tensor,
                _nccl4py_reduce_op(self.nccl4py, op),
                stream=stream.cuda_stream,
            )
            return out_tensor
        self.nccl.ncclAllReduce(
            buffer_type(in_tensor.data_ptr()),  # sendbuff
            buffer_type(out_tensor.data_ptr()),  # recvbuff - DIFFERENT pointer
            in_tensor.numel(),
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_gather(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: Optional[list[int]] = None,
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
        stream = self._resolve_stream()

        if self.use_nccl4py:
            if sizes is not None:
                split_offset = 0
                self.nccl4py.group_start()
                try:
                    for root, split_size in enumerate(sizes):
                        dst_slice = output_tensor[
                            split_offset : split_offset + split_size
                        ]
                        self.nccl_comm.broadcast(
                            input_tensor[:split_size],
                            dst_slice,
                            root,
                            stream=stream.cuda_stream,
                        )
                        split_offset += split_size
                finally:
                    self.nccl4py.group_end()
            else:
                self.nccl_comm.allgather(
                    input_tensor, output_tensor, stream=stream.cuda_stream
                )
            return

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

    def cp_all_gather_into_tensor(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        sizes: Optional[list[int]] = None,
    ):
        """
        Currently, it is mainly used in context parallelism,
        primarily leveraging pynccl to implement non-blocking allgather communication.
        """
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert input_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        if self.use_nccl4py:
            self.nccl_comm.allgather(
                input_tensor, output_tensor, stream=stream.cuda_stream
            )
            return
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
        sizes: Optional[list[int]] = None,
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
        stream = self._resolve_stream()

        if self.use_nccl4py:
            if sizes is not None:
                split_offset = 0
                self.nccl4py.group_start()
                try:
                    for root, split_size in enumerate(sizes):
                        chunk = input_tensor[
                            split_offset : split_offset + split_size, ...
                        ]
                        self.nccl_comm.reduce(
                            chunk,
                            output_tensor,
                            _nccl4py_reduce_op(self.nccl4py, op),
                            root=root,
                            stream=stream.cuda_stream,
                        )
                        split_offset += split_size
                finally:
                    self.nccl4py.group_end()
            else:
                self.nccl_comm.reduce_scatter(
                    input_tensor,
                    output_tensor,
                    _nccl4py_reduce_op(self.nccl4py, op),
                    stream=stream.cuda_stream,
                )
            return

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

    def send(self, tensor: torch.Tensor, dst: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()
        if self.use_nccl4py:
            self.nccl_comm.send(tensor, peer=dst, stream=stream.cuda_stream)
            return
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()
        if self.use_nccl4py:
            self.nccl_comm.recv(tensor, peer=src, stream=stream.cuda_stream)
            return
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        stream = self._resolve_stream()

        if self.use_nccl4py:
            self.nccl_comm.broadcast(
                tensor, tensor, root=src, stream=stream.cuda_stream
            )
            return

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
        if self.use_nccl4py:
            from nccl.core.constants import WindowFlag
            from nccl.core.resources import RegisteredWindowHandle

            self.nccl_comm.register_window()
            return RegisteredWindowHandle(
                self.nccl_comm.ptr, ptr, size, WindowFlag.CollSymmetric
            )
        return self.nccl.ncclCommWindowRegister(self.comm, buffer_type(ptr), size, 1)

    def deregister_comm_window(self, window):
        if self.use_nccl4py:
            window.close()
            return
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def group_start(self):
        if self.use_nccl4py:
            self.nccl4py.group_start()
            return
        self.nccl.ncclGroupStart()

    def group_end(self):
        if self.use_nccl4py:
            self.nccl4py.group_end()
            return
        self.nccl.ncclGroupEnd()

    @contextmanager
    def change_state(self, enable: Optional[bool] = None):
        """
        A context manager to change the enabled state of the communicator.
        """
        if enable is None:
            # guess a default value when not specified
            enable = self.available

        old_disable = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disable

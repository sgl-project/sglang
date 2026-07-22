# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import ctypes
import logging
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple, Union

# ===================== import region =====================
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.device_communicators.pynccl_wrapper import (
    NCCL_API_MAGIC,
    NCCL_CONFIG_UNDEF_INT,
    NCCL_WIN_COLL_SYMMETRIC,
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclConfig_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.utils.common import get_current_device_stream_fast

logger = logging.getLogger(__name__)

# ncclPutSignal / one-sided RMA shipped in NCCL 2.30; raw ncclGetVersion() for
# 2.30.x is MAJOR*10000+MINOR*100+PATCH (e.g. 2.30.7 -> 23007).
_NCCL_RMA_MIN_VERSION = 23000


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

    def register_comm_window_raw(
        self, ptr: int, size: int, win_flags: int = NCCL_WIN_COLL_SYMMETRIC
    ):
        """Register ptr..ptr+size as an NCCL symmetric-memory window and return
        the opaque window handle (ncclWindow_t). On RMA-unavailable HW (no
        NVLink, vGPU) NCCL returns success but a NULL handle; callers using
        the handle for one-sided RMA must treat NULL as unavailable."""
        return self.nccl.ncclCommWindowRegister(
            self.comm, buffer_type(ptr), size, win_flags
        )

    def deregister_comm_window(self, window):
        return self.nccl.ncclCommWindowDeregister(self.comm, window)

    def supports_rma(self) -> bool:
        """Whether the loaded NCCL exposes one-sided RMA at a usable version.
        A capability gate only; does not guarantee the HW can serve RMA."""
        return getattr(self.nccl, "has_rma", False) and (
            self.nccl_version >= _NCCL_RMA_MIN_VERSION
        )

    # ---- one-sided RMA primitives (NCCL 2.29-2.30+) -------------------------

    def put_signal(
        self,
        local_ptr: int,
        count: int,
        dtype: int,
        peer: int,
        peer_win,
        peer_win_offset: int = 0,
        sig_idx: int = 0,
        ctx: int = 0,
        flags: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Put count elements of dtype from local_ptr into the peer's window
        peer_win at peer_win_offset (bytes) and send a signal. peer_win must be
        the peer's handle, allgathered across ranks."""
        stream = stream or self._resolve_stream()
        return self.nccl.ncclPutSignal(
            buffer_type(local_ptr),
            count,
            dtype,
            peer,
            peer_win,
            peer_win_offset,
            sig_idx,
            ctx,
            flags,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def signal(
        self,
        peer: int,
        sig_idx: int = 0,
        ctx: int = 0,
        flags: int = 0,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        stream = stream or self._resolve_stream()
        return self.nccl.ncclSignal(
            peer, sig_idx, ctx, flags, self.comm, cudaStream_t(stream.cuda_stream)
        )

    def wait_signal(
        self,
        descs_ptr: int,
        n_desc: int,
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Wait for signals described by n_desc contiguous ncclWaitSignalDesc_t
        at descs_ptr (built via make_wait_descs)."""
        stream = stream or self._resolve_stream()
        return self.nccl.ncclWaitSignal(
            n_desc,
            ctypes.c_void_p(descs_ptr),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def win_get_user_ptr(self, window):
        out = buffer_type()
        self.nccl.ncclWinGetUserPtr(self.comm, window, ctypes.byref(out))
        return out.value

    def nccl_mem_alloc(self, size: int) -> int:
        """Allocate a symmetric-memory buffer (ncclMemAlloc); eligible for
        window registration. Free with nccl_mem_free."""
        ptr = buffer_type()
        self.nccl.ncclMemAlloc(ctypes.byref(ptr), size)
        return ptr.value

    def nccl_mem_free(self, ptr: int) -> None:
        if ptr:
            self.nccl.ncclMemFree(buffer_type(ptr))

    def get_peer_device_pointer(self, window, offset: int, peer: int):
        out = buffer_type()
        self.nccl.ncclGetPeerDevicePointer(window, offset, peer, ctypes.byref(out))
        return out.value

    @staticmethod
    def make_wait_descs(peers_opcnt: Sequence[Tuple[int, int]]):
        """Build a contiguous ncclWaitSignalDesc_t[] buffer for wait_signal.
        peers_opcnt: [(peer_rank, op_cnt), ...]; returns (buffer_ptr, n_desc)."""
        import numpy as np

        n_desc = len(peers_opcnt)
        dtype = np.dtype(
            [("op_cnt", "i4"), ("peer", "i4"), ("sig_idx", "i4"), ("ctx", "i4")]
        )
        arr = np.zeros(n_desc, dtype=dtype)
        for i, (peer, op_cnt) in enumerate(peers_opcnt):
            arr[i]["peer"] = peer
            arr[i]["op_cnt"] = op_cnt
        return np.ascontiguousarray(arr).ctypes.data, n_desc

    def make_nccl_config(
        self, num_rma_ctx: int = NCCL_CONFIG_UNDEF_INT
    ) -> "ncclConfig_t":
        """Build an ncclConfig_t matching NCCL_CONFIG_INITIALIZER for the loaded
        library. num_rma_ctx defaults to UNDEF (inherit NCCL default); setting
        it is not required for RMA to work but sizes the RMA contexts."""
        cfg = ncclConfig_t()
        cfg.size = ctypes.sizeof(ncclConfig_t)
        cfg.magic = NCCL_API_MAGIC
        cfg.version = self.nccl_version
        for fld in (
            "blocking",
            "cgaClusterSize",
            "minCTAs",
            "maxCTAs",
            "splitShare",
            "trafficClass",
            "collnetEnable",
            "CTAPolicy",
            "shrinkShare",
            "nvlsCTAs",
            "nChannelsPerNetPeer",
            "nvlinkCentricSched",
            "graphUsageMode",
            "numRmaCtx",
            "maxP2pPeers",
            "graphStreamOrdering",
        ):
            setattr(cfg, fld, NCCL_CONFIG_UNDEF_INT)
        cfg.netName = None
        cfg.commName = None
        cfg.numRmaCtx = num_rma_ctx
        return cfg

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
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

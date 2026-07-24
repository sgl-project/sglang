# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM's vllm/distributed/device_communicators/pyxccl_comm.py

# ``PyXcclCommunicator`` is the XPU counterpart of ``PyNcclCommunicator``: a
# direct binding to Intel's oneCCL library that bypasses ``torch.distributed``
# for collective operations (AllReduce, AllGather, ReduceScatter, Broadcast,
# Send/Recv).  It mirrors the ``pynccl.py`` pattern for CUDA, adapted for Intel
# XPU + oneCCL so that ``GroupCoordinator`` can drive it beside ``pynccl_comm``.
#
# The ctypes/library-loading layer lives in the vendored ``pyxccl_wrapper``
# module (the XPU analogue of ``pynccl_wrapper``), which binds the ``oneccl*``
# NCCL-compatible C API exported by ``libccl.so``.  This module only
# orchestrates rendezvous, device binding, and buffer/stream marshalling.  When
# the oneCCL library cannot be loaded the communicator marks itself disabled and
# the caller falls back to ``torch.distributed``.

import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from sglang.srt.distributed.device_communicators.pyxccl_wrapper import (
    ONECCLLibrary,
    buffer_type,
    onecclComm_t,
    onecclDataTypeEnum,
    onecclRedOpTypeEnum,
    onecclUniqueId,
    xpuStream_t,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class PyXcclCommunicator:
    """oneCCL-backed communicator for Intel XPU devices.

    Mirrors the interface of :class:`PyNcclCommunicator` so that
    ``GroupCoordinator`` can use it as a drop-in replacement for the
    ``torch.distributed`` XPU collective calls. It is opt-in
    (``SGLANG_ENABLE_PYXCCL=1``) and intended for environments where the torch
    build has no usable XCCL backend.

    Unlike ``PyNcclCommunicator`` (which starts ``disabled`` and is only enabled
    inside CUDA-graph capture via ``change_state``), pyxccl is enabled as soon as
    it initializes: it is the live collective path in both eager and graph modes
    on XPU, replacing the ``torch.distributed`` calls.
    """

    def __init__(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[int, str, torch.device],
        library_path: Optional[str] = None,
    ):
        """
        Args:
            group: the process group to work on. pyxccl broadcasts the oneCCL
                unique id over this group during rendezvous, so it should be a
                CPU (gloo) group or a StatelessProcessGroup, mirroring how
                PyNcclCommunicator is attached to ``cpu_group``.
            device: the XPU device this communicator is bound to.
            library_path: optional explicit path to ``libccl.so.1``. When None,
                ``SGLANG_PYXCCL_SO_PATH`` is tried first, then the dynamic-linker
                default (``"libccl.so.1"``).
        """
        if isinstance(group, StatelessProcessGroup):
            self.rank = group.rank
            self.world_size = group.world_size
        else:
            assert dist.is_initialized()
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)

        self.group = group

        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        # pyxccl is opt-in (off by default); skip when world_size == 1 or when it
        # has not been explicitly enabled via SGLANG_ENABLE_PYXCCL.
        if self.world_size == 1 or not envs.SGLANG_ENABLE_PYXCCL.get():
            self.available = False
            self.disabled = True
            return

        # Resolve library path: explicit arg > env var > ldconfig default.
        so_path = library_path or envs.SGLANG_PYXCCL_SO_PATH.get() or "libccl.so.1"

        try:
            self.oneccl = ONECCLLibrary(so_path)
        except Exception as exc:
            logger.warning(
                "Failed to load oneCCL library from %s: %s. "
                "The XPU oneCCL fast-path communicator is disabled.",
                so_path,
                exc,
            )
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False
        # Opt-in per-collective logging (SGLANG_DEBUG_PYXCCL=1) to confirm at
        # runtime that collectives are dispatched through this oneCCL path.
        self._debug = envs.SGLANG_DEBUG_PYXCCL.get()

        if self.rank == 0:
            logger.info(
                "sglang XPU is using oneCCL==%s from %s",
                self.oneccl.onecclGetVersion(),
                so_path,
            )

        try:
            self._init_comm()
        except Exception as exc:
            logger.warning(
                "oneCCL communicator initialization failed: %s. This is often "
                "caused by missing transport configuration; check that "
                "CCL_ATL_TRANSPORT=ofi and FI_PROVIDER_PATH are set correctly "
                "(see docs/platforms/xpu.pyxccl.md).",
                exc,
            )
            self.available = False
            self.disabled = True

    def _init_comm(self) -> None:
        """Rendezvous + bind the oneCCL communicator (called from __init__)."""
        # --- broadcast unique_id so all ranks share the same rendezvous ---
        if self.rank == 0:
            self.unique_id = self.oneccl.onecclGetUniqueId()
        else:
            self.unique_id = onecclUniqueId()

        if isinstance(self.group, StatelessProcessGroup):
            self.unique_id = self.group.broadcast_obj(self.unique_id, src=0)
        else:
            # Pack the opaque id into a ByteTensor for torch.distributed.
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(self.group)
            # arg `src` in `broadcast` is the global rank
            dist.broadcast(tensor, src=ranks[0], group=self.group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte

        # Bind the communicator to the correct XPU device.
        self.oneccl.onecclSetDevice(self.device.index)

        self.comm: onecclComm_t = self.oneccl.onecclCommInitRank(
            self.world_size, self.unique_id, self.rank
        )

        # Warmup: a tiny all_reduce to force runtime init before first use, and
        # to fail fast (→ disabled, torch.distributed fallback) if the transport
        # is misconfigured.
        data = torch.zeros(1, device=self.device)
        self.all_reduce(data)
        torch.xpu.synchronize()
        del data

    def _stream(self) -> "xpuStream_t":
        """Return the current XPU SYCL queue as the oneCCL stream handle."""
        return xpuStream_t(torch.xpu.current_stream().sycl_queue)

    def _debug_log(self, coll: str, tensor: torch.Tensor, **extra) -> None:
        """When SGLANG_DEBUG_PYXCCL=1, log that a collective went via oneCCL."""
        if not self._debug:
            return
        detail = "".join(f" {k}={v}" for k, v in extra.items())
        logger.info(
            "[pyxccl] rank=%d/%d oneCCL.%s numel=%d dtype=%s dev=%s%s",
            self.rank,
            self.world_size,
            coll,
            tensor.numel(),
            tensor.dtype,
            tensor.device,
            detail,
        )

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM):
        """In-place all-reduce (send buffer == recv buffer)."""
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        self._debug_log("AllReduce", tensor, op=op)
        self.oneccl.onecclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            onecclRedOpTypeEnum.from_torch(op),
            self.comm,
            self._stream(),
        )

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        self._debug_log("AllGather", input_tensor)
        self.oneccl.onecclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            onecclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            self._stream(),
        )

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {input_tensor.device}"
        )
        self._debug_log("ReduceScatter", input_tensor, op=op)
        self.oneccl.onecclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            onecclDataTypeEnum.from_torch(input_tensor.dtype),
            onecclRedOpTypeEnum.from_torch(op),
            self.comm,
            self._stream(),
        )

    def broadcast(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        self._debug_log("Broadcast", tensor, src=src)
        self.oneccl.onecclBroadcast(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            self._stream(),
        )

    def send(self, tensor: torch.Tensor, dst: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        self._debug_log("Send", tensor, dst=dst)
        self.oneccl.onecclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            self._stream(),
        )

    def recv(self, tensor: torch.Tensor, src: int):
        if self.disabled:
            return
        assert tensor.device == self.device, (
            f"this pyxccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        self._debug_log("Recv", tensor, src=src)
        self.oneccl.onecclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            onecclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            self._stream(),
        )

    def destroy(self):
        """Release the oneCCL communicator handle."""
        if self.available and not self.disabled:
            try:
                self.oneccl.onecclCommDestroy(self.comm)
            except Exception as exc:
                logger.debug("onecclCommDestroy raised: %s", exc)
            self.available = False
            self.disabled = True

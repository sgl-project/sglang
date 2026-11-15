# SPDX-License-Identifier: Apache-2.0
"""Direct multicast allgather using the existing background-transfer gate."""

from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.gated_transfer import (
    CopyChunk,
    Link,
    TransferChain,
    chunk_spans,
)
from sglang.srt.utils.cuda_vmm_utils import check_drv

try:
    from cuda.bindings import driver as cu
except ImportError:
    cu = None


def has_multicast_support(device: torch.device) -> bool:
    """Whether ``device`` supports CUDA multicast and cuda-python is present
    to drive the copies. A rendezvous can still return no multicast mapping
    (for example, when the fabric manager cannot bind it).

    Asks the driver: torch's own check would fix this process's
    symmetric-memory backend before the caller selects one."""
    if cu is None:
        return False
    check_drv(cu.cuInit(0), "cuInit")
    attribute = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
    return bool(
        check_drv(cu.cuDeviceGetAttribute(attribute, device.index), "multicast")
    )


class MulticastUnavailableError(RuntimeError):
    """Some rank's symmetric memory has no full-group multicast mapping.
    ``MulticastAllGather`` raises it on every rank alike."""


def _multicast_address(tensor, handle, rank, world_size):
    """Address of a tensor view in a full-group symmetric multicast mapping."""
    if (
        handle is None
        or handle.rank != rank
        or handle.world_size != world_size
        or not handle.multicast_ptr
        or len(handle.buffer_ptrs) != world_size
        or not all(handle.buffer_ptrs)
    ):
        raise ValueError("a full-group symmetric multicast mapping is required")
    offset = tensor.storage_offset() * tensor.element_size()
    start = int(handle.offset) + offset
    if start < 0 or start + tensor.nbytes > handle.buffer_size:
        raise ValueError("tensor view is outside its symmetric allocation")
    # The backend already includes handle.offset in multicast_ptr.
    return int(handle.multicast_ptr) + offset


class MulticastAllGather:
    """Gather directly into a bound symmetric output, one copy per chunk.

    Construction is collective: allocate ``output`` with PyTorch symmetric
    memory, then construct on every rank in matching order before any gate
    can park. The selected
    backend must expose a multicast mapping spanning the whole group, or
    every rank raises ``MulticastUnavailableError``; backend
    selection remains with the caller. For example, select the NCCL backend
    before allocating any symmetric tensors::

        symm_mem.set_backend("NCCL")
        output = symm_mem.empty(world * count, device=device, dtype=dtype)
        chain = TransferChain(Link.NVLINK)
        gather = MulticastAllGather(group, output, chain, chunk_bytes=8 << 20)
        ticket = gather.run(input_, after=producer_stream)
        # Foreground work can use paused_background_transfers(Link.NVLINK).
        chain.wait(ticket, stream=consumer_stream)

    Output is contiguous and rank-major. A call gathers into the bound output,
    or into any contiguous view inside it (``run(output=...)``), so one
    mapping serves gathers whose size varies. Input has
    output.numel()/world_size elements and may be exactly this rank's output
    block. Calls must use the
    same shapes, output offsets, and order across ranks. One chain serializes
    copies and both barriers, so a completed ticket means every rank's data
    has arrived. ``TransferChain(None)`` runs the same copies without gating.

    Only the copies are gated. A paused rank also holds its peers at the exit
    barrier. Warm foreground communication before pausing, and follow the
    chain's eager-only and stream-ordering requirements. Keep the input and
    this object alive until its ticket completes. Output/control allocation
    and rendezvous happen during construction; run() reuses mappings and barriers.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        output: torch.Tensor,
        chain: TransferChain,
        *,
        chunk_bytes: int = 8 << 20,
    ):
        from torch.distributed import _symmetric_memory as symm_mem

        if cu is None:
            raise RuntimeError("multicast allgather requires cuda-python")
        if not output.is_cuda or not output.is_contiguous():
            raise ValueError("output must be a contiguous CUDA tensor")
        if chunk_bytes <= 0 or chunk_bytes % output.element_size():
            raise ValueError("chunk_bytes must be a positive multiple of element size")
        if chain.device != output.device.index or chain.link not in (None, Link.NVLINK):
            raise ValueError("chain must use the output device and NVLINK or no gate")
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        # cuStreamBatchMemOp accepts fewer than 256 operations. run() checks
        # that each output divides evenly; the bound output need not.
        if not 0 < self.world_size <= 127:
            raise ValueError("multicast allgather supports 1-127 ranks")
        self.output, self.chain, self.chunk_bytes = output, chain, chunk_bytes
        with torch.cuda.device(output.device):
            self._output_handle = symm_mem.rendezvous(output, group.group_name)
            stride = (self.world_size * 4 + 63) // 64 * 64
            self._control = symm_mem.empty(
                2 * stride, dtype=torch.uint8, device=output.device
            )
            self._control_handle = symm_mem.rendezvous(self._control, group.group_name)
            error = None
            try:
                self._multicast_ptr = _multicast_address(
                    output, self._output_handle, self.rank, self.world_size
                )
                multicast = _multicast_address(
                    self._control, self._control_handle, self.rank, self.world_size
                )
            except ValueError as exc:
                error = str(exc)
            # Agree before failing: a rank that raised alone would leave its
            # peers waiting in the barrier below.
            errors = [None] * self.world_size
            dist.all_gather_object(errors, error, group=group)
            failed = [(rank, error) for rank, error in enumerate(errors) if error]
            if failed:
                raise MulticastUnavailableError(
                    "; ".join(f"rank {rank}: {error}" for rank, error in failed)
                )
            self._one = torch.ones(1, dtype=torch.int32, pin_memory=True)
            self._signals = tuple(
                CopyChunk(
                    multicast + channel * stride + self.rank * 4,
                    self._one.data_ptr(),
                    4,
                )
                for channel in (0, 1)
            )
            self._ops = tuple(
                self._wait_then_clear(self._control.data_ptr() + channel * stride)
                for channel in (0, 1)
            )
            self._control.zero_()
            torch.cuda.synchronize(output.device)
            dist.barrier(group=group)
        self._entry = functools.partial(self._barrier, 0)
        self._exit = functools.partial(self._barrier, 1)

    def _wait_then_clear(self, address):
        waits, clears = [], []
        for rank in range(self.world_size):
            wait, clear = cu.CUstreamBatchMemOpParams(), cu.CUstreamBatchMemOpParams()
            wait.operation = cu.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WAIT_VALUE_32
            wait.waitValue.address = address + rank * 4
            wait.waitValue.value = 1
            wait.waitValue.flags = cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ
            clear.operation = cu.CUstreamBatchMemOpType.CU_STREAM_MEM_OP_WRITE_VALUE_32
            clear.writeValue.address = address + rank * 4
            clear.writeValue.value = 0
            clear.writeValue.flags = (
                cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
            )
            waits.append(wait)
            clears.append(clear)
        return waits + clears

    def _barrier(self, channel, stream):
        # Signals use multicast copies; waits/clears use this tensor's LOCAL VA.
        self._signals[channel](stream)
        ops = self._ops[channel]
        check_drv(
            cu.cuStreamBatchMemOp(stream.cuda_stream, len(ops), ops, 0),
            "multicast allgather barrier",
        )

    def run(
        self,
        input_: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        after: Optional[torch.cuda.Stream] = None,
    ) -> int:
        """Submit a gather; ticket completion includes every rank's exit signal.

        ``output`` (the bound output by default) is a contiguous view inside
        the bound output. ``after`` (the current stream by default) must follow
        input writes and all previous readers of ``output`` on this rank.
        """
        bound = self.output
        output = bound if output is None else output
        if not input_.is_contiguous() or not output.is_contiguous():
            raise ValueError("input and output must be contiguous")
        if input_.device != bound.device or input_.dtype != output.dtype:
            raise ValueError("input and output must have the same dtype/device")
        if input_.numel() * self.world_size != output.numel():
            raise ValueError("output must contain world_size * input elements")
        src, dst, nbytes = input_.data_ptr(), output.data_ptr(), input_.nbytes
        base = bound.data_ptr()
        if output.device != bound.device or not (
            base <= dst and dst + output.nbytes <= base + bound.nbytes
        ):
            raise ValueError("output must be a view inside the bound output")
        overlaps = max(src, dst) < min(src + nbytes, dst + output.nbytes)
        if overlaps and src != dst + self.rank * nbytes:
            raise ValueError(
                "input may overlap only its own complete output rank block"
            )
        target = self._multicast_ptr + (dst - base) + self.rank * nbytes
        chunks = [
            CopyChunk(target + offset, src + offset, size)
            for offset, size in chunk_spans(nbytes, self.chunk_bytes)
        ]
        return self.chain.run(
            chunks, setup=(self._entry,), finish=(self._exit,), after=after
        )

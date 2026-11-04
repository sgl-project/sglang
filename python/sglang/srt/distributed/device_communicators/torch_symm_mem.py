# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
import socket
from dataclasses import dataclass, field
from functools import cache
from typing import Optional, Union
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import is_cuda, is_hip

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _is_cuda = is_cuda()
    _is_hip = is_hip()

    torch_symm_mem_available = False
    if _is_cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


logger = logging.getLogger(__name__)


@dataclass
class _SymmetricMemoryWorkspace:
    """Owned mappings and per-operation counters; no transport implementation."""

    communicator: "TorchSymmMemCommunicator"
    buffers: torch.Tensor
    data_handle: object
    peers: list[torch.Tensor]
    stream: torch.cuda.Stream
    signals: torch.Tensor | None = None
    signal_handle: object = None
    signal_peers: list[torch.Tensor] = field(default_factory=list)
    signal_ptrs: tuple[int, ...] = ()
    peer_ptrs: torch.Tensor | None = None
    comm_stream: torch.cuda.Stream | None = None
    generations: list[int] = field(default_factory=lambda: [0, 0])
    barrier_buffer: torch.Tensor | None = None

    def close(self):
        self.communicator.release_workspace(self, synchronize=True)

    def __del__(self):
        self.communicator.release_workspace(self)


class TorchSymmMemCommunicator:
    """
    Torch symmetric-memory collectives and intra-node peer transfers.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    All-reduce eligibility is independent of RMA. Peer transfers reserve
    separate workspaces and do not require BF16 payloads or multicast support.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [4, 6, 8],
    }

    _instances = WeakValueDictionary()

    @staticmethod
    def _local_multicast_unavailable_reason():
        """Probe prerequisites without allocating symmetric communication buffers."""
        if torch.cuda.get_device_capability()[0] < 9:
            return "CUDA multicast requires SM90 or newer"
        try:
            import torch.distributed._symmetric_memory as symm_mem
            from cuda.bindings import driver

            device = torch.device("cuda", torch.cuda.current_device())
            if symm_mem.get_backend(device) != "CUDA":
                return "Torch CUDA symmetric allocator is unavailable"
            if not symm_mem._SymmetricMemory.has_multicast_support(
                symm_mem.DeviceType.CUDA, device.index
            ):
                return "CUDA multicast is unavailable"
            if not all(
                hasattr(driver, name)
                for name in (
                    "cuStreamWaitValue32",
                    "cuStreamWriteValue32",
                    "cuMemcpyDtoDAsync",
                )
            ):
                return "CUDA stream memory operations are unavailable"
            # The legacy CAN_USE_STREAM_MEM_OPS_V1 attribute can be false even
            # when the current APIs work. Probe an ordered local write/wait instead;
            # the wait is submitted only after a successful write on the same stream.
            with torch.inference_mode(False):
                signal = torch.zeros(1, dtype=torch.int32, device=device)
            stream = torch.cuda.current_stream(device)
            result = driver.cuStreamWriteValue32(
                stream.cuda_stream, signal.data_ptr(), 1, 0
            )
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                return f"CUDA stream memory write is unavailable: {result[0]}"
            result = driver.cuStreamWaitValue32(
                stream.cuda_stream, signal.data_ptr(), 1, 0
            )
            stream.synchronize()
            if result[0] != driver.CUresult.CUDA_SUCCESS:
                return f"CUDA stream memory wait is unavailable: {result[0]}"
        except (ImportError, AttributeError, RuntimeError) as exc:
            return f"symmetric state communication capability probe failed: {exc}"
        return None

    @staticmethod
    def multicast_unavailable_reason(group, *, enabled=True):
        """Agree once at startup; no capability collectives in the forward path."""
        world = dist.get_world_size(group) if group is not None else 1
        if world not in (2, 4, 8):
            return "symmetric state communication requires 2, 4 or 8 ranks"
        local = (
            socket.gethostname(),
            enabled,
            TorchSymmMemCommunicator._local_multicast_unavailable_reason(),
        )
        peers = [None] * world
        with torch.inference_mode(False):
            dist.all_gather_object(peers, local, group=group)
        if len({peer[1] for peer in peers}) != 1:
            raise ValueError(
                "Peer ranks disagree on symmetric state communication selection"
            )
        if not enabled:
            return "symmetric state communication disabled for operator comparison"
        if len({peer[0] for peer in peers}) != 1:
            return "symmetric state communication requires peer ranks on one host"
        for rank, (_, _, reason) in enumerate(peers):
            if reason is not None:
                return f"Peer rank {rank}: {reason}"
        return None

    @staticmethod
    def _device(device):
        device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    @classmethod
    def get(cls, group, device, *, enable_all_reduce=False):
        """Share group/device resources without coupling RMA to all-reduce support."""
        device = cls._device(device)
        key = (group, device)
        communicator = cls._instances.get(key)
        if communicator is None:
            communicator = cls(group, device, enable_all_reduce=False)
            cls._instances[key] = communicator
        if enable_all_reduce:
            communicator.initialize_all_reduce()
        return communicator

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        *,
        enable_all_reduce=True,
    ):
        self.disabled = True  # Eligibility of the optional all-reduce path only.
        self.buffer = None
        self.max_size = 0
        self._all_reduce_initialized = False
        self._peer_access_initialized = False
        self.device = self._device(device)
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if enable_all_reduce:
            self.initialize_all_reduce()

    def initialize_all_reduce(self):
        if self._all_reduce_initialized:
            return
        self._all_reduce_initialized = True
        if not torch_symm_mem_available:
            return
        torch.cuda.set_device(self.device)
        self.dtype = torch.bfloat16
        self.device_capability = torch.cuda.get_device_capability(self.device)[0]
        supported_max_sizes = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(
            self.device_capability
        )
        if supported_max_sizes is None:
            logger.warning(
                "TorchSymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in supported_max_sizes:
            logger.warning(
                "TorchSymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = supported_max_sizes[self.world_size]
        # Keep the JIT all-reduce buffer above the largest prefill payload
        # ([16384, 6144] bf16 = 192 MiB), including room for tail regions.
        if envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR.get():
            self.max_size = max(self.max_size, 256 * 1024 * 1024)

            if (
                get_exec().comm.enable_scattered_sconv
                or envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV.get()
            ):
                # Fused extend kernels are out-of-place, so OUT must hold the
                # same maximum prefill payload as IN, including tail regions.
                self.max_size = max(self.max_size, 512 * 1024 * 1024)
        with torch.inference_mode(False), torch.no_grad():
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
        handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if handle.multicast_ptr == 0:
            logger.warning(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported."
            )
            self.buffer = None
            self.disabled = True
            return
        self.disabled = False

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor; if omitted, a new tensor is allocated.

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Writes the result into 'out' and returns it.
        """
        if not self.should_torch_symm_mem_allreduce(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM.get(
            self.device_capability, ()
        ):
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out

    def _initialize_peer_access(self):
        if self._peer_access_initialized:
            return
        if not torch_symm_mem_available or self.device.type != "cuda":
            raise ValueError("Peer operations require Torch CUDA symmetric memory")
        from cuda.bindings import driver

        hosts = [None] * self.world_size
        with torch.inference_mode(False), torch.no_grad():
            dist.all_gather_object(hosts, socket.gethostname(), group=self.group)
        if len(set(hosts)) != 1:
            raise ValueError(
                "Torch symmetric peer operations require ranks on one host; use nccl across hosts"
            )
        if torch_symm_mem.get_backend(self.device) != "CUDA":
            raise ValueError(
                "Peer operations require the Torch CUDA symmetric allocator"
            )
        self.driver = driver
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1) % self.world_size
        self._peer_access_initialized = True

    def _allocate_workspace(self, shape, dtype, *, signal_shape=None):
        self._initialize_peer_access()
        # Persistent storage remains mutable across warmup/inference scopes.
        with torch.inference_mode(False), torch.no_grad():
            buffers = torch_symm_mem.empty(shape, device=self.device, dtype=dtype)
        handle = torch_symm_mem.rendezvous(buffers, group=self.group)
        workspace = _SymmetricMemoryWorkspace(
            communicator=self,
            buffers=buffers,
            data_handle=handle,
            peers=[
                handle.get_buffer(rank, shape, dtype) for rank in range(self.world_size)
            ],
            stream=torch.cuda.current_stream(self.device),
        )
        if signal_shape is not None:
            with torch.inference_mode(False), torch.no_grad():
                workspace.signals = torch_symm_mem.empty(
                    signal_shape, device=self.device, dtype=torch.int32
                ).zero_()
            workspace.signal_handle = torch_symm_mem.rendezvous(
                workspace.signals, group=self.group
            )
            workspace.signal_peers = [
                workspace.signal_handle.get_buffer(rank, signal_shape, torch.int32)
                for rank in range(self.world_size)
            ]
            workspace.signal_ptrs = tuple(
                peer.data_ptr() for peer in workspace.signal_peers
            )
            workspace.stream.synchronize()
            dist.barrier(group=self.group)
        return workspace

    def allocate_ring_workspace(self, shape, dtype):
        """Reserve independent arrival/credit counters for a ring exchange."""
        return self._allocate_workspace(shape, dtype, signal_shape=(2, 2))

    def allocate_broadcast_workspace(self, capacity_bytes):
        """Reserve byte storage for multicast, independently of all-reduce support."""
        if self.world_size not in (2, 4, 8):
            raise ValueError("CUDA multicast broadcast requires 2, 4 or 8 ranks")
        workspace = self._allocate_workspace(
            (capacity_bytes,), torch.uint8, signal_shape=(2, self.world_size)
        )
        if not workspace.data_handle.multicast_ptr:
            raise ValueError("CUDA multicast broadcast requires a multicast mapping")
        workspace.comm_stream = torch.cuda.Stream(device=self.device)
        return workspace

    def allocate_peer_workspace(
        self, shape, dtype, *, signal_shape=None, rdma_compatible=False
    ):
        """Reserve shared peer inputs and the device pointer table used by kernels."""
        if rdma_compatible:
            # The CUDA symmetric allocator's VMM payload cannot be registered by
            # all GPUDirect RDMA transports. Keep ordinary Torch storage for the
            # payload and share its IPC mapping; use symm only for GPU barriers.
            from torch.multiprocessing.reductions import reduce_tensor

            workspace = self._allocate_workspace(
                (1,), torch.uint8, signal_shape=signal_shape
            )
            workspace.barrier_buffer = workspace.buffers
            with torch.inference_mode(False), torch.no_grad():
                payload = torch.empty(shape, dtype=dtype, device=self.device)
                # Each receiver needs its own IPC reference-count token. A single
                # reduction broadcast to several consumers would release it early.
                outgoing = [
                    reduce_tensor(payload) if rank != self.rank else None
                    for rank in range(self.world_size)
                ]
                incoming = [None] * self.world_size
                dist.all_gather_object(incoming, outgoing, group=self.group)
                peers = []
                for rank, handles in enumerate(incoming):
                    if rank == self.rank:
                        peers.append(payload)
                    else:
                        rebuild, args = handles[self.rank]
                        # Open the IPC mapping in this rank's CUDA context, not
                        # in the sender's device recorded by reduce_tensor.
                        peers.append(rebuild(*args[:6], self.device.index, *args[7:]))
            workspace.buffers = payload
            workspace.peers = peers
        else:
            workspace = self._allocate_workspace(
                shape, dtype, signal_shape=signal_shape
            )
        with torch.inference_mode(False), torch.no_grad():
            workspace.peer_ptrs = torch.tensor(
                [peer.data_ptr() for peer in workspace.peers],
                device=self.device,
                dtype=torch.uint64,
            )
        return workspace

    def release_workspace(self, workspace, *, synchronize=False):
        if workspace.buffers is None:
            return
        if synchronize:
            workspace.stream.synchronize()
            dist.barrier(group=self.group)
        # Release all peer views and mappings before their local backing storage.
        workspace.signal_peers = []
        workspace.signal_handle = None
        workspace.signals = None
        workspace.peers = []
        workspace.peer_ptrs = None
        workspace.data_handle = None
        workspace.barrier_buffer = None
        workspace.buffers = None

    def publish(self, workspace, value):
        """Publish read-only peer inputs after all preceding consumers finish."""
        if workspace.communicator is not self or workspace.buffers is None:
            raise ValueError(
                "Peer workspace is closed or belongs to another communicator"
            )
        if (
            torch.cuda.current_stream(self.device) != workspace.stream
            or torch.cuda.is_current_stream_capturing()
        ):
            raise ValueError(
                "Peer publication requires its construction stream and eager execution"
            )
        if (
            value.shape != workspace.buffers.shape
            or value.dtype != workspace.buffers.dtype
            or value.device != self.device
        ):
            raise ValueError("Peer publication shape/dtype/device mismatch")
        workspace.data_handle.barrier(channel=0, timeout_ms=30000)
        workspace.buffers.copy_(value)
        workspace.data_handle.barrier(channel=1, timeout_ms=30000)

    def _checked(self, result):
        if result[0] != self.driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Torch symmetric-memory RMA failed: {result[0]}")

    def exchange_ring(
        self, workspace, *, send_slot, recv_slot, signal_slot, byte_ranges=None
    ):
        """Put byte ranges into the next peer's slot and wait for the previous peer.

        The caller owns slot reuse and compute/communication stream ordering.
        Byte ranges are relative to a payload slot and have identical source and
        destination offsets; no attention layout or token count is interpreted.
        """
        if workspace.communicator is not self or workspace.buffers is None:
            raise ValueError(
                "RMA workspace is closed or belongs to another communicator"
            )
        if not (
            0 <= send_slot < workspace.buffers.shape[0]
            and 0 <= recv_slot < workspace.buffers.shape[0]
        ):
            raise ValueError("RMA payload slot exceeds its workspace")
        source = workspace.buffers[send_slot]
        destination = workspace.peers[self.next_rank][recv_slot]
        slot_bytes = source.numel() * source.element_size()
        ranges = ((0, slot_bytes),) if byte_ranges is None else tuple(byte_ranges)
        if signal_slot not in (0, 1) or any(
            offset < 0 or size < 0 or offset + size > slot_bytes
            for offset, size in ranges
        ):
            raise ValueError("RMA signal slot or byte range exceeds its workspace")
        workspace.generations[signal_slot] = (
            workspace.generations[signal_slot] + 1
        ) & 0xFFFFFFFF
        generation = workspace.generations[signal_slot]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        driver = self.driver
        # Default writes include a fence; cyclic GE waits handle uint32 wrap.
        self._checked(
            driver.cuStreamWriteValue32(
                stream,
                workspace.signal_ptrs[self.prev_rank] + 4 * (signal_slot * 2 + 1),
                generation,
                0,
            )
        )
        self._checked(
            driver.cuStreamWaitValue32(
                stream,
                workspace.signal_ptrs[self.rank] + 4 * (signal_slot * 2 + 1),
                generation,
                0,
            )
        )
        for offset, size in ranges:
            if size:
                self._checked(
                    driver.cuMemcpyDtoDAsync(
                        destination.data_ptr() + offset,
                        source.data_ptr() + offset,
                        size,
                        stream,
                    )
                )
        self._checked(
            driver.cuStreamWriteValue32(
                stream,
                workspace.signal_ptrs[self.next_rank] + 4 * signal_slot * 2,
                generation,
                0,
            )
        )
        self._checked(
            driver.cuStreamWaitValue32(
                stream,
                workspace.signal_ptrs[self.rank] + 4 * signal_slot * 2,
                generation,
                0,
            )
        )

    def exchange_peer_rows(self, workspace, value, ranges, *, slot):
        """Copy preplanned strided rows directly into each peer's target layout.

        Ranges specify peer, source row, target row, count and both row strides.
        All ranks grant overwrite credit and publish arrival even for empty data.
        The caller orders readiness and previous consumers on this stream.
        """
        if (
            workspace.communicator is not self
            or workspace.buffers is None
            or workspace.signals is None
            or tuple(workspace.signals.shape) != (2, self.world_size, 2)
            or slot not in (0, 1)
            or torch.cuda.current_stream(self.device) != workspace.stream
        ):
            raise ValueError("Peer row exchange requires its own workspace and stream")
        from sglang.kernels.ops.communication.ipc_a2a import load_ipc_a2a_sync

        _, capacity, features = workspace.buffers.shape
        row_bytes = features * value.element_size()
        copies = []
        for peer, source, target, count, source_step, target_step in ranges:
            if not (
                0 <= peer < self.world_size
                and min(source, target) >= 0
                and min(count, source_step, target_step) > 0
                and source + (count - 1) * source_step < value.shape[0]
                and target + (count - 1) * target_step < capacity
            ):
                raise ValueError("Peer row range exceeds its reserved storage")
            copies.extend(
                (
                    value.data_ptr() + source * row_bytes,
                    workspace.peers[peer].data_ptr()
                    + (slot * capacity + target) * row_bytes,
                    row_bytes,
                    count,
                    source_step * row_bytes,
                    target_step * row_bytes,
                )
            )
        generation = (workspace.generations[slot] + 1) & 0xFFFFFFFF
        workspace.generations[slot] = generation
        base = slot * self.world_size * 2
        signals = []
        for peer in range(self.world_size):
            if peer != self.rank:
                credit_write = workspace.signal_ptrs[peer] + 4 * (base + 2 * self.rank)
                credit_wait = workspace.signal_ptrs[self.rank] + 4 * (base + 2 * peer)
                signals.extend(
                    (credit_write, credit_wait, credit_write + 4, credit_wait + 4)
                )
        load_ipc_a2a_sync().copy_peer_strided(
            signals, copies, workspace.stream.cuda_stream, generation
        )

    def broadcast(self, workspace, value, owner):
        if workspace.communicator is not self or workspace.buffers is None:
            raise ValueError(
                "Broadcast workspace is closed or belongs to another communicator"
            )
        if value.device != self.device:
            raise ValueError("Broadcast payload device differs from its communicator")
        if torch.cuda.current_stream(value.device) != workspace.stream:
            raise ValueError(
                "CUDA multicast broadcast must use its construction stream"
            )
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA multicast broadcast does not support graph capture")
        size = value.numel() * value.element_size()
        if (
            not value.is_contiguous()
            or value.data_ptr() % 16
            or size % 16
            or size > workspace.buffers.numel()
        ):
            raise ValueError(
                "CUDA multicast broadcast requires an aligned contiguous payload within capacity"
            )
        if not 0 <= owner < self.world_size:
            raise ValueError(
                "CUDA multicast broadcast owner is outside the process group"
            )
        workspace.generations[0] = (workspace.generations[0] + 1) & 0xFFFFFFFF
        generation = workspace.generations[0]
        ready = workspace.stream.record_event()
        with torch.cuda.stream(workspace.comm_stream):
            workspace.comm_stream.wait_event(ready)
            stream = workspace.comm_stream.cuda_stream
            write = self.driver.cuStreamWriteValue32
            wait = self.driver.cuStreamWaitValue32
            # Every receiver has consumed the preceding generation before it
            # grants permission to overwrite the replicated symmetric buffer.
            self._checked(
                write(
                    stream, workspace.signal_ptrs[owner] + 4 * self.rank, generation, 0
                )
            )
            if self.rank == owner:
                for rank in range(self.world_size):
                    self._checked(
                        wait(
                            stream,
                            workspace.signal_ptrs[owner] + 4 * rank,
                            generation,
                            0,
                        )
                    )
                _get_multicast_kernel()[(min(32, max(4, (size + 32767) // 32768)),)](
                    value,
                    workspace.data_handle.multicast_ptr,
                    size // 16,
                    BLOCK=256,
                    num_warps=8,
                )
                # Default write flags include a system memory fence. Publish
                # arrival only after the multicast kernel completes.
                for rank in range(self.world_size):
                    if rank != owner:
                        self._checked(
                            write(
                                stream,
                                workspace.signal_ptrs[rank]
                                + 4 * (self.world_size + owner),
                                generation,
                                0,
                            )
                        )
            else:
                self._checked(
                    wait(
                        stream,
                        workspace.signal_ptrs[self.rank]
                        + 4 * (self.world_size + owner),
                        generation,
                        0,
                    )
                )
            received = workspace.comm_stream.record_event()
        workspace.stream.wait_event(received)
        value.view(torch.uint8).reshape(-1).copy_(workspace.buffers[:size])
        return value


@cache
def _get_multicast_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _multimem_st_128(multicast_ptrs, x, y, z, w, mask):
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.eq.s32 %p0, $6, 1;
                @!%p0 bra end;
                multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
                end:
            }
            """,
            "=r,l,r,r,r,r,r",
            args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
            dtype=(tl.uint32),
            is_pure=False,
            pack=1,
        )

    @triton.jit
    def _local_ld_128(in_ptr, mask):
        return tl.inline_asm_elementwise(
            """
            {
                .reg .pred %p0;
                setp.eq.s32 %p0, $5, 1;
                @!%p0 bra end;
                ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
                end:
            }
            """,
            "=r,=r,=r,=r,l,r",
            args=[in_ptr, mask.to(tl.int32)],
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _get_flat_tid():
        tid_x, tid_y, tid_z = _get_tid()
        ntid_x, ntid_y, _ = _get_ntid()
        return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x

    @triton.jit
    def _get_tid():
        return tl.inline_asm_elementwise(
            """
            mov.u32 $0, %tid.x;
            mov.u32 $1, %tid.y;
            mov.u32 $2, %tid.z;
            """,
            "=r,=r,=r",
            [],
            dtype=(tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _get_ntid():
        return tl.inline_asm_elementwise(
            """
            mov.u32 $0, %ntid.x;
            mov.u32 $1, %ntid.y;
            mov.u32 $2, %ntid.z;
            """,
            "=r,=r,=r",
            [],
            dtype=(tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _multicast_bytes(
        source,
        multicast_ptr,
        vectors,
        BLOCK: tl.constexpr,
    ):
        # Only the owner launches this kernel. Stream memory operations provide
        # readiness and publication ordering without spinning on receiver SMs.
        block_start = tl.program_id(0) * BLOCK
        while block_start < vectors:
            index = block_start + _get_flat_tid()
            x, y, z, w = _local_ld_128(
                source.to(tl.pointer_type(tl.uint64)) + index * 2, index < vectors
            )
            target = (
                multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + index * 2
            )
            _multimem_st_128(target, x, y, z, w, index < vectors)
            block_start += tl.num_programs(0) * BLOCK

    return _multicast_bytes

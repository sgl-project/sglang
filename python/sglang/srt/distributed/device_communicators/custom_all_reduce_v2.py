"""JIT custom all-reduce (v2) over a decoupled storage plane.

The CUDA side is split into two independent pieces:

- ``Communicator``: a thin pointer holder over symmetric-memory workspaces
  (push buffers, pull buffer, semaphores) plus a local push counter. All
  storage is allocated and owned here, in Python.
- the all-reduce kernel: a pure function of ``(input, Communicator, algo,
  pull_arg)`` with three algorithms (1shot_push / 1shot_pull / 2shot_pull)
  and three pull data sources (eager workspace / CUDA-graph pointer table /
  multicast address).

CUDA-graph inputs are exchanged from Python after capture (cudaIpc handles
for cudaMalloc-backed pointers, fabric/posix-fd VMM mapping for expandable
segments) and written into a device-side pointer table (``graph_params``);
the kernel captured in the graph dereferences its row at replay time.
"""

import enum
import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.jit_kernel.all_reduce import (
    AllReduceAlgo,
    Communicator,
    IPCManager,
    custom_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)

from .configs.custom_all_reduce_v2 import get_all_reduce_config
from .custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from .vmm_utils import (
    VmmGraphInputManager,
    compute_graph_capture_bases,
    is_vmm_pointer,
)

logger = logging.getLogger(__name__)

MB = 1024 * 1024

_ALIGN_BYTES = 1024
_SEMAPHORE_BYTES = 128
_MAX_GRAPH_INPUTS = 131072
# resolved once at import time; explicit constructor sizes take precedence
_DEFAULT_MAX_SIZE = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024


class _PullMode(enum.Enum):
    EAGER = enum.auto()  # pull_arg = False (also used for 1shot_push)
    MULTICAST = enum.auto()  # pull_arg = True
    GRAPH = enum.auto()  # pull_arg = a graph_params row


def _ceil_align(nbytes: int, align: int) -> int:
    return (nbytes + align - 1) // align * align


def _allocate_symmetric_memory(nbytes: int, device: torch.device, group: ProcessGroup):
    from torch._C._distributed_c10d import _SymmetricMemory

    if torch.__version__ < "2.11.0":
        import torch.distributed._symmetric_memory as torch_symm_mem

        torch_symm_mem.enable_symm_mem_for_group(group.group_name)
    tensor = _SymmetricMemory.empty_strided_p2p(
        (nbytes,),
        [1],
        torch.uint8,
        device,
        group.group_name,
    )
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    return tensor, symm_mem


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
        *,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
        max_pull_blocks: Optional[int] = None,
        max_push_blocks: Optional[int] = None,
    ) -> None:
        """
        :param max_size: direction-agnostic memory cap. Each workspace is
                         sized to what the tuned config wants, clipped to
                         this bound. Defaults to
                         ``SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB`` (16 MB).
        :param max_pull_size: explicit pull workspace size; overrides both
                              the tuned size and ``max_size``.
        :param max_push_size: explicit per-buffer push workspace size;
                              overrides both the tuned size and ``max_size``.
        """
        self.disabled = True
        if not can_use_custom_all_reduce_v2(group=group, device=device):
            return

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        base_config = get_all_reduce_config(self.world_size)
        if max_pull_size is None:
            max_pull_size = min(base_config.max_pull_bytes, max_size)
        if max_push_size is None:
            max_push_size = min(base_config.max_push_bytes, max_size)
        # a minimal workspace keeps the Communicator valid even when a caller
        # only uses one direction (e.g. push-only fused qk-norm instances)
        self.max_pull_size = _ceil_align(max(max_pull_size, _ALIGN_BYTES), _ALIGN_BYTES)
        self.max_push_size = _ceil_align(max(max_push_size, _ALIGN_BYTES), _ALIGN_BYTES)
        self.max_size = max(self.max_pull_size, self.max_push_size)
        num_pull_blocks = base_config.num_pull_blocks
        num_push_blocks = base_config.num_push_blocks
        if max_pull_blocks is not None:
            num_pull_blocks = max(min(num_pull_blocks, max_pull_blocks), 1)
        if max_push_blocks is not None:
            num_push_blocks = max(max_push_blocks, 1)
        self.config = base_config.clip(
            max_push_bytes=self.max_push_size, max_pull_bytes=self.max_pull_size
        )._replace(num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks)
        self.override_algo: Optional[AllReduceAlgo] = None
        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()

        # device-side pointer table: one row of world_size pointers per
        # graph-captured all-reduce input (at most 8 MB at world_size = 8)
        self.graph_params = torch.zeros(
            (_MAX_GRAPH_INPUTS, self.world_size),
            dtype=torch.uint64,
            device=self.device,
        )
        self._init_workspace()
        self._ipc_manager = IPCManager()
        self._vmm_graph_input_manager = VmmGraphInputManager(
            obj=self,
            group=self.group,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._graph_inputs: List[Tuple[int, int]] = []  # (data_ptr, nbytes)
        self._graph_counter = 0
        self._graph_mode_allowed = False
        self.disabled = False

    def _init_workspace(self) -> None:
        """Slice one symmetric-memory allocation into all shared buffers.

        Layout per rank: ``[2 * world_size push buffers | pull buffer |
        pull semaphores]``. The push counter is rank-local, so it lives in a
        plain CUDA tensor instead.
        """
        cfg = self.config
        push_num_bufs = 2 * self.world_size  # 2 phases x world_size peers
        push_ws_bytes = push_num_bufs * self.max_push_size
        pull_ws_bytes = self.max_pull_size
        pull_sem_bytes = _SEMAPHORE_BYTES * cfg.num_pull_blocks
        total_bytes = push_ws_bytes + pull_ws_bytes + pull_sem_bytes
        pull_ws_offset = push_ws_bytes
        pull_sem_offset = push_ws_bytes + pull_ws_bytes

        self._symm_tensor, symm_mem = _allocate_symmetric_memory(
            total_bytes, device=self.device, group=self.group
        )
        workspaces = [
            symm_mem.get_buffer(i, [total_bytes], torch.uint8)
            for i in range(self.world_size)
        ]
        workspaces[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

        def slice_ws(rank: int, shape: List[int], offset: int) -> torch.Tensor:
            nbytes = 1
            for s in shape:
                nbytes *= s
            assert offset + nbytes <= total_bytes
            return workspaces[rank][offset : offset + nbytes].view(shape)

        push_workspaces = [
            slice_ws(i, [push_num_bufs, self.max_push_size], 0)
            for i in range(self.world_size)
        ]
        pull_workspaces = [
            slice_ws(i, [pull_ws_bytes], pull_ws_offset) for i in range(self.world_size)
        ]
        pull_semaphores = [
            slice_ws(i, [cfg.num_pull_blocks, _SEMAPHORE_BYTES], pull_sem_offset)
            for i in range(self.world_size)
        ]
        self._push_counter = torch.zeros(
            (cfg.num_push_blocks,), dtype=torch.uint32, device=self.device
        )

        multicast_ptr = int(symm_mem.multicast_ptr)
        can_multicast = multicast_ptr != 0
        pull_mc_workspace = multicast_ptr + pull_ws_offset if can_multicast else None
        if not can_multicast or cfg.num_mc_blocks is None:
            self.config = self.config._replace(num_mc_blocks=None)

        self.obj = Communicator(
            rank=self.rank,
            world_size=self.world_size,
            push_workspaces=push_workspaces,
            pull_workspaces=pull_workspaces,
            pull_semaphores=pull_semaphores,
            push_counter=self._push_counter.view(-1, 1).view(torch.uint8),
            pull_mc_workspace=pull_mc_workspace,
        )
        if self.config.num_mc_blocks is not None:
            self.obj.config(num_multicast_blocks=self.config.num_mc_blocks)
        if self.rank == 0:
            logger.info(
                "All Reduce config: symmetric_memory = %.2f MB, "
                "local_buffer = %.2f MB, multicast = %s",
                total_bytes / MB,
                (self.graph_params.nbytes + self._push_counter.nbytes) / MB,
                self.config.num_mc_blocks is not None,
            )
        dist.barrier(group=self.group)

    # ------------------------------------------------------------------
    # Algo selection
    # ------------------------------------------------------------------

    def uncap_pull_thresholds(self) -> None:
        """Raise the 2-shot ceiling to the workspace capacity.

        The tuned config caps ``2shot_pull`` at the size where NCCL takes
        over; benchmarks and tests that must keep every sweep size on the
        custom-AR path can lift that cap up to ``max_pull_size``.
        """

        def uncap(heuristic):
            return heuristic._replace(two_shot_pull_threshold=self.max_pull_size)

        self.config = self.config._replace(
            graph=uncap(self.config.graph),
            eager=uncap(self.config.eager),
        )

    def _can_use_graph(self) -> bool:
        # `_graph_mode_allowed` is only set inside `capture()`, so the eager
        # hot path never reaches the cudart capture query. During capture,
        # warm-up runs execute immediately and must not consume a
        # graph_params row (it would be dereferenced before registration).
        return (
            self._graph_mode_allowed
            and not is_in_tc_piecewise_cuda_graph()
            and torch.cuda.is_current_stream_capturing()
        )

    def _pick_algo(
        self, nbytes: int, can_use_graph: bool
    ) -> Tuple[Optional[AllReduceAlgo], _PullMode]:
        heuristic = self.config.graph if can_use_graph else self.config.eager
        default_mode = _PullMode.GRAPH if can_use_graph else _PullMode.EAGER
        use_multicast = self.config.num_mc_blocks is not None
        if nbytes <= heuristic.one_shot_push_threshold:
            return AllReduceAlgo.ONE_SHOT_PUSH, _PullMode.EAGER
        if nbytes <= heuristic.one_shot_pull_threshold:
            return AllReduceAlgo.ONE_SHOT_PULL, default_mode
        if use_multicast and heuristic.mc.contains(nbytes):
            return AllReduceAlgo.TWO_SHOT_PULL, _PullMode.MULTICAST
        if nbytes <= heuristic.two_shot_pull_threshold:
            return AllReduceAlgo.TWO_SHOT_PULL, default_mode
        return None, _PullMode.EAGER

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled or inp.numel() == 0:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.override_algo is not None:
            return inp_size <= self.max_size
        algo, _ = self._pick_algo(inp_size, can_use_graph=self._can_use_graph())
        return algo is not None

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        can_use_graph = self._can_use_graph()
        if self.override_algo is not None:
            algo = self.override_algo
            use_graph = can_use_graph and not algo.is_push()
            mode = _PullMode.GRAPH if use_graph else _PullMode.EAGER
        else:
            algo, mode = self._pick_algo(nbytes, can_use_graph=can_use_graph)
            assert algo is not None, f"No algo for {nbytes} bytes"
        if mode == _PullMode.GRAPH:
            pull_arg: torch.Tensor | bool = self._allocate_graph_row(input, nbytes)
        else:
            pull_arg = mode == _PullMode.MULTICAST
        return torch.from_dlpack(custom_all_reduce(self.obj, input, algo, pull_arg))

    def _allocate_graph_row(self, input: torch.Tensor, nbytes: int) -> torch.Tensor:
        index = self._graph_counter + len(self._graph_inputs)
        assert (
            index < _MAX_GRAPH_INPUTS
        ), "Graph input table overflow, increase _MAX_GRAPH_INPUTS!"
        self._graph_inputs.append((input.data_ptr(), nbytes))
        return self.graph_params[index]

    # ------------------------------------------------------------------
    # CUDA-graph input registration
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        try:
            self._graph_mode_allowed = not self.tms_cudagraph
            yield
        finally:
            self._graph_mode_allowed = False
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "Cannot register graph inputs while capturing CUDA graph"
        self._register_graph_inputs()

    def _register_graph_inputs(self) -> None:
        if not self._graph_inputs:
            return
        first_ptr = self._graph_inputs[0][0]
        if is_vmm_pointer(first_ptr):
            # calls back into get_graph_capture_bases / register_peer_mapped_inputs
            self._vmm_graph_input_manager.register_graph_inputs()
        else:
            self._register_graph_inputs_ipc()

    def _register_graph_inputs_ipc(self) -> None:
        """Register graph capture inputs via cudaIpc handles.

        This is the fast path for cudaMalloc-backed allocations. Fails on
        VMM pointers (expandable_segments), which use the VMM path instead.
        """
        ptrs = [ptr for ptr, _ in self._graph_inputs]
        handles = self._ipc_manager.batch_get_handles(ptrs)
        local = [(list(handle), int(offset)) for handle, offset in handles]
        gathered: List[Optional[list]] = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.group)
        ptrs_per_rank: List[List[int]] = []
        for rank, remote in enumerate(gathered):
            if rank == self.rank:
                ptrs_per_rank.append(ptrs)
            else:
                ptrs_per_rank.append(list(self._ipc_manager.batch_open_handles(remote)))
        peer_ptrs = [
            [ptrs_per_rank[rank][i] for rank in range(self.world_size)]
            for i in range(len(ptrs))
        ]
        self.register_peer_mapped_inputs(peer_ptrs)

    def get_graph_capture_bases(self):
        """VMM base allocations of pending graph inputs (VmmGraphInputManager hook)."""
        return compute_graph_capture_bases(self._graph_inputs)

    def register_peer_mapped_inputs(self, peer_ptrs: List[List[int]]) -> None:
        """Write per-input peer pointers into the device-side pointer table."""
        assert len(peer_ptrs) == len(self._graph_inputs)
        count = len(peer_ptrs)
        rows = torch.tensor(peer_ptrs, dtype=torch.uint64, device=self.device)
        self.graph_params[self._graph_counter : self._graph_counter + count].copy_(rows)
        # the rows must be visible before any (PDL-chained) graph replay
        torch.cuda.synchronize()
        self._graph_counter += count
        self._graph_inputs.clear()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self._ipc_manager.destroy()
            dist.barrier(group=self.group)
            del self.obj  # drop the pointer holder before the workspace tensors
        if hasattr(self, "_vmm_graph_input_manager"):
            self._vmm_graph_input_manager.close()

    def __del__(self):
        self.close()


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(range(2, 9)),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True

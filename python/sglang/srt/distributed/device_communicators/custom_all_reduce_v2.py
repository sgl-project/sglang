"""JIT custom all-reduce (v2) over a decoupled storage plane.

The CUDA side is split into independent pieces:

- ``PushPlane``: the zero-filled symmetric push buffers plus a rank-local
  phase counter.
- ``PullPlane``: the symmetric pull buffers plus the per-block semaphores
  guarding them. The buffers exist only because this class hands the
  all-reduce tensors that are not symmetric memory, so it stages them
  through; the K3 fused collectives bring their own symmetric input and
  borrow the semaphores alone.
- ``Communicator``: the two planes above (either may be absent), the handle
  every kernel takes, plus the pull launch widths.
- the all-reduce kernel: a pure function of ``(Communicator, input, algo,
  graph_params, use_multicast)`` with three algorithms (1shot_push /
  1shot_pull / 2shot_pull) and three pull data sources (eager pull buffer /
  CUDA-graph pointer table / multicast address).

All storage is allocated and owned here, in Python.

CUDA-graph inputs are exchanged from Python after capture (cudaIpc handles
for cudaMalloc-backed pointers, fabric/posix-fd VMM mapping for expandable
segments) and written into a device-side pointer table (``graph_params``);
the kernel captured in the graph dereferences its row at replay time.
"""

import logging
import math
from contextlib import contextmanager
from typing import Any, Final, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.ops.communication.all_reduce import (
    AllReduceAlgo,
    Communicator,
    IPCManager,
    PullPlane,
    PushPlane,
    custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils.cuda_vmm_utils import (
    VmmGraphInputManager,
    compute_graph_capture_bases,
    is_vmm_pointer,
)

from .config import Dispatch, get_supported_world_sizes, get_tuning
from .custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_one_nvlink_clique,
    is_weak_contiguous,
)

logger = logging.getLogger(__name__)


def _kb_to_b(size_kb: Optional[int]) -> Optional[int]:
    return None if size_kb is None else int(size_kb) * 1024


_DEFAULT_MAX_SIZE = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024
_FORCED_PULL_SIZE = _kb_to_b(envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB.get())
_FORCED_PUSH_SIZE = _kb_to_b(envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB.get())

_ALIGN_BYTES = 1024
_SEMAPHORE_BYTES = 128
_MAX_GRAPH_INPUTS = 131072


def _align_up(nbytes: int) -> int:
    return (nbytes + _ALIGN_BYTES - 1) // _ALIGN_BYTES * _ALIGN_BYTES


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


def _probe_multicast(group: ProcessGroup, device: torch.device) -> bool:
    _, symm_mem = _allocate_symmetric_memory(_ALIGN_BYTES, device=device, group=group)
    return int(symm_mem.multicast_ptr) != 0


def _allocate_workspace(
    *,
    group: ProcessGroup,
    device: torch.device,
    rank: int,
    world_size: int,
    max_push_size: int,
    max_pull_size: int,
    num_push_blocks: int,
    num_pull_blocks: int,
    num_multicast_blocks: Optional[int],
) -> tuple[Communicator, int, int, Any]:
    """Slice one symmetric-memory allocation into every shared buffer.

    Layout per rank: ``[2 * world_size push slots | pull buffer | pull
    semaphores]``. One allocation rather than three keeps it to a single
    rendezvous and a single multicast base to offset from. The push counter is
    rank-local, so it lives in a plain CUDA tensor.

    Collective: every rank must call it.
    """
    push_num_slots = 2 * world_size  # 2 phases x world_size peers
    push_bytes = push_num_slots * max_push_size
    pull_offset = push_bytes
    pull_bytes = max_pull_size
    sem_offset = push_bytes + pull_bytes
    sem_bytes = _SEMAPHORE_BYTES * num_pull_blocks

    total_bytes = push_bytes + pull_bytes + sem_bytes
    symm_slab, symm_mem = _allocate_symmetric_memory(total_bytes, device, group)
    slabs = [
        symm_mem.get_buffer(i, [total_bytes], torch.uint8) for i in range(world_size)
    ]
    # The push slots (lamport pos-zero markers) and the semaphores must start
    # zeroed; the pull buffer need not, but it rides along in the one-shot memset.
    slabs[rank].zero_()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    def slice_all(shape: List[int], offset: int) -> List[torch.Tensor]:
        """The same sub-range of every rank's slab, one view per rank."""
        nbytes = math.prod(shape)
        assert offset + nbytes <= total_bytes
        return [slab[offset : offset + nbytes].view(shape) for slab in slabs]

    def slice_mc_ptr(offset: int) -> Optional[int]:
        return multicast_ptr + offset if multicast_ptr != 0 else None

    multicast_ptr = int(symm_mem.multicast_ptr)
    push_counter = torch.zeros((num_push_blocks,), dtype=torch.uint32, device=device)
    push_plane = PushPlane(
        rank,
        world_size,
        workspaces=slice_all([push_num_slots, max_push_size], 0),
        counter=push_counter.view(-1, 1).view(torch.uint8),
        mc_workspace=slice_mc_ptr(0),
    )
    # A push-only caller passes max_pull_size=0 and num_pull_blocks=0, which
    # leaves both halves empty; the plane is then an inert placeholder.
    pull_plane = PullPlane(
        rank,
        world_size,
        workspaces=slice_all([pull_bytes], pull_offset),
        semaphores=slice_all([num_pull_blocks, _SEMAPHORE_BYTES], sem_offset),
        mc_workspace=slice_mc_ptr(pull_offset),
        mc_semaphore=slice_mc_ptr(sem_offset),
    )
    comm = Communicator(push=push_plane, pull=pull_plane)
    if num_multicast_blocks is not None:
        comm.set_pull_multicast_blocks(min(num_multicast_blocks, num_pull_blocks))
    dist.barrier(group=group)
    return comm, total_bytes, push_counter.nbytes, (symm_slab, push_counter)


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
        *,
        max_pull_size: Optional[int] = _FORCED_PULL_SIZE,
        max_push_size: Optional[int] = _FORCED_PUSH_SIZE,
        num_pull_blocks: Optional[int] = None,
        num_push_blocks: Optional[int] = None,
        uncap_pull: bool = False,
    ) -> None:
        """
        :param max_size: direction-agnostic memory cap. Each workspace is
                         sized to what the tuned config wants, clipped to
                         this bound. Defaults to
                         ``SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB`` (16 MB).
        :param max_pull_size: explicit pull buffer size; overrides both
                              the tuned size and ``max_size``. ``0`` builds a
                              push-only instance.
        :param max_push_size: explicit per-slot push workspace size;
                              overrides both the tuned size and ``max_size``.
        :param num_push_blocks: explicit override the number of push blocks;
        :param num_pull_blocks: explicit override the number of pull blocks;
        :param uncap_pull: run ``2shot_pull`` up to the pull workspace
                           capacity instead of stopping at the tuned NCCL
                           crossover. The only way to lift that ceiling -- an
                           explicit ``max_pull_size`` sizes the buffer without
                           widening the bands. For sweeps that must keep every
                           size on the custom-AR path.

        ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB`` /
        ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB`` supply the default
        for the two size parameters above
        """
        self.disabled = True
        if not can_use_custom_all_reduce_v2(group, device):
            return

        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.has_multicast = _probe_multicast(group=self.group, device=self.device)
        init_tuning = get_tuning(self.world_size, self.has_multicast)
        max_push_size, max_pull_size = init_tuning.get_workspace_sizes(
            max_size=max_size,
            max_push_size=max_push_size,
            max_pull_size=max_pull_size,
        )
        max_push_size = _align_up(max_push_size)
        max_pull_size = _align_up(max_pull_size)
        self._tuning: Final = init_tuning.recompile(
            max_push_size=max_push_size,
            max_pull_size=max_pull_size,
            uncap_pull=uncap_pull,
        )
        del init_tuning
        self.max_push_size: Final = max_push_size
        self.max_pull_size: Final = max_pull_size
        self.max_size = max(max_push_size, max_pull_size)
        if num_push_blocks is None:
            num_push_blocks = self._tuning.num_push_blocks
        if num_pull_blocks is None:
            num_pull_blocks = self._tuning.num_pull_blocks
        self.num_push_blocks: Final = num_push_blocks
        self.num_pull_blocks: Final = num_pull_blocks
        # On a multi-node (MNNVL) group the symm-mem workspace plane works
        # across nodes, but graph zero-copy input registration (cudaIpc /
        # node-local VMM remap) does not -- force eager pull inside graphs.
        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self._is_graph_mode_supported: Final = not tms_cudagraph and not is_multinode
        # device-side pointer table: one row of world_size pointers per captured input
        graph_entries = _MAX_GRAPH_INPUTS if self._is_graph_mode_supported else 0
        self._graph_params = torch.zeros(
            (graph_entries, self.world_size),
            dtype=torch.uint64,
            device=self.device,
        )
        # NOTE: storage is just to keep alive counter & symm-mem
        self.obj, total_bytes, local_bytes, self._storage = _allocate_workspace(
            group=self.group,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            max_push_size=max_push_size,
            max_pull_size=max_pull_size,
            num_push_blocks=num_push_blocks,
            num_pull_blocks=num_pull_blocks,
            num_multicast_blocks=self._tuning.num_multicast_blocks,
        )
        if self.rank == 0:
            MB = 1024 * 1024
            logger.info(
                "All Reduce config: symmetric_memory = %.2f MiB, "
                "local_buffer = %.2f MiB, use_multicast = %s",
                total_bytes / MB,
                (self._graph_params.nbytes + local_bytes) / MB,
                self._tuning.num_multicast_blocks is not None,
            )
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
        self._override_algo = None
        self.disabled = False

    def _select_dispatch(self, nbytes: int) -> Optional[Dispatch]:
        use_graph = (
            self._graph_mode_allowed  # only set when in graph + supported
            and not is_in_tc_piecewise_cuda_graph()
            and torch.cuda.is_current_stream_capturing()
        )
        if self._override_algo is not None:
            algo, use_multicast = self._override_algo
            return Dispatch(
                algo=algo,
                use_graph=use_graph and not use_multicast and algo.is_pull(),
                use_multicast=use_multicast,
            )
        return self._tuning.select(nbytes, in_graph=use_graph)

    def force_algo(self, algo: AllReduceAlgo, use_multicast: bool = False) -> None:
        self._override_algo = (algo, use_multicast)

    def should_custom_ar(self, input: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled:
            return False
        nbytes = input.nbytes
        if nbytes % 16 != 0 or not is_weak_contiguous(input):
            return False
        return self._select_dispatch(nbytes) is not None

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.nbytes
        dispatch = self._select_dispatch(nbytes)
        assert dispatch is not None, f"No dispatch band for {nbytes = }"
        graph_params = (
            self._allocate_graph_row(input, nbytes) if dispatch.use_graph else None
        )
        return custom_all_reduce(
            self.obj,
            input,
            algo=dispatch.algo,
            graph_params=graph_params,
            use_multicast=dispatch.use_multicast,
        )

    def _allocate_graph_row(self, input: torch.Tensor, nbytes: int) -> torch.Tensor:
        index = self._graph_counter + len(self._graph_inputs)
        assert index < _MAX_GRAPH_INPUTS, "Graph input table overflow"
        self._graph_inputs.append((input.data_ptr(), nbytes))
        return self._graph_params[index]

    # ------------------------------------------------------------------
    # CUDA-graph input registration
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        try:
            self._graph_mode_allowed = self._is_graph_mode_supported
            yield
        finally:
            self._graph_mode_allowed = False
        assert not torch.cuda.is_current_stream_capturing(), (
            "Cannot register graph inputs while capturing CUDA graph"
        )
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
        row = torch.tensor(peer_ptrs, dtype=torch.uint64, device=self.device)
        self._graph_params[self._graph_counter : self._graph_counter + count].copy_(row)
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
            del self._storage, self._graph_params
        if hasattr(self, "_vmm_graph_input_manager"):
            self._vmm_graph_input_manager.close()

    def __del__(self):
        self.close()


def _is_vmm_backed_allocator(device: torch.device) -> bool:
    """Check whether expandable-segments VMM backs the caching allocator."""
    probe = torch.empty(1, dtype=torch.uint8, device=device)
    return is_vmm_pointer(probe.data_ptr())


def can_use_custom_all_reduce_v2(group: ProcessGroup, device: torch.device) -> bool:
    supported = get_supported_world_sizes()
    if dist.get_world_size(group=group) not in supported:
        return False
    if not all(in_the_same_node_as(group, source_rank=0)):
        return is_one_nvlink_clique(group, device) and _is_vmm_backed_allocator(device)
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(supported),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True

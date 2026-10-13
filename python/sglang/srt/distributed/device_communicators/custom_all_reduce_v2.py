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
  graph_params, use_multicast, peer_bases, peer_offset)`` with three
  algorithms (1shot_push / 1shot_pull / 2shot_pull) and four pull data
  sources (eager pull buffer / CUDA-graph pointer table / symmetric-pool
  input / multicast address).

All storage is allocated and owned here, in Python.

CUDA-graph inputs are exchanged from Python after capture (cudaIpc handles
for cudaMalloc-backed pointers, fabric/posix-fd VMM mapping for expandable
segments) and written into a device-side pointer table (``graph_params``);
the kernel captured in the graph dereferences its row at replay time.

Eager inputs allocated from the NCCL symmetric pool (``--enable-symm-mem``)
are zero-copy too: each pool segment is peer-mapped once, and a call passes
the per-rank segment bases plus the input's offset, which is the same on
every rank.
"""

import bisect
import logging
import math
from contextlib import contextmanager
from typing import List, Optional, Tuple

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
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    get_symmetric_segments,
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

from .configs.custom_all_reduce_v2 import (
    MC,
    PULL2,
    PUSH1,
    get_all_reduce_config,
    get_supported_world_sizes,
)
from .custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_one_nvlink_clique,
    is_weak_contiguous,
)

logger = logging.getLogger(__name__)

MB = 1024 * 1024

_ALIGN_BYTES = 1024
_SEMAPHORE_BYTES = 128
_MAX_GRAPH_INPUTS = 131072
# resolved once at import time; explicit constructor sizes take precedence
_DEFAULT_MAX_SIZE = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024
# forced per-direction workspace sizes; highest priority, override both the
# tuned config and explicit constructor sizes (None = not forced)
_FORCE_PULL_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB.get()
_FORCE_PUSH_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB.get()


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
        :param max_pull_size: explicit pull buffer size; overrides both
                              the tuned size and ``max_size``. ``0`` builds a
                              push-only instance.
        :param max_push_size: explicit per-slot push workspace size;
                              overrides both the tuned size and ``max_size``.
        :param max_pull_blocks: cap on the barrier plane's block count; ``0``
                                builds a push-only instance.

        ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB`` /
        ``SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB`` take the highest
        priority and override all of the size parameters above.
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
            max_pull_size = min(base_config.max_bytes(push=False), max_size)
        if max_push_size is None:
            max_push_size = min(base_config.max_bytes(push=True), max_size)
        if _FORCE_PULL_SIZE_KB is not None:
            max_pull_size = int(_FORCE_PULL_SIZE_KB) * 1024
        if _FORCE_PUSH_SIZE_KB is not None:
            max_push_size = int(_FORCE_PUSH_SIZE_KB) * 1024
        # Zero on either pull knob opts out of the pull half entirely: no
        # pull plane at all, and every pull algo disabled by its zero
        # capacity in _pick_config. Push-only callers (the fused qk-norm instances)
        # take this path rather than allocating a placeholder buffer just to
        # satisfy a constructor.
        self.pull_enabled = max_pull_blocks != 0 and max_pull_size > 0
        self.max_pull_size = (
            _ceil_align(max(max_pull_size, _ALIGN_BYTES), _ALIGN_BYTES)
            if self.pull_enabled
            else 0
        )
        self.max_push_size = _ceil_align(max(max_push_size, _ALIGN_BYTES), _ALIGN_BYTES)
        self.max_size = max(self.max_pull_size, self.max_push_size)
        num_pull_blocks = base_config.num_pull_blocks
        num_push_blocks = base_config.num_push_blocks
        if max_pull_blocks:
            num_pull_blocks = max(min(num_pull_blocks, max_pull_blocks), 1)
        if max_push_blocks is not None:
            num_push_blocks = max(max_push_blocks, 1)
        self.config = base_config._replace(
            num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks
        )
        # forced sizes bypass the tuned NCCL crossover: fill each forced
        # workspace up to its capacity
        if _FORCE_PUSH_SIZE_KB is not None:
            self._add_rows(head=((math.inf, PUSH1),))
        if _FORCE_PULL_SIZE_KB is not None:
            self.uncap_pull_thresholds()
        self.override_algo: Optional[AllReduceAlgo] = None
        # On a multi-node (MNNVL) group the symm-mem workspace plane works
        # across nodes, but graph zero-copy input registration (cudaIpc /
        # node-local VMM remap) does not — force eager pull inside graphs.
        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self._is_graph_mode_supported = not tms_cudagraph and not is_multinode
        self._symm_supported = not is_multinode
        # device-side pointer table: one row of world_size pointers per
        # graph-captured all-reduce input
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
        self._in_capture = False
        # peer-mapped symmetric-pool segments, sorted by base:
        # (base, end, CPU int64 tensor of every rank's base)
        self._symm_bases: List[int] = []
        self._symm_segments: List[Tuple[int, int, torch.Tensor]] = []
        self._symm_seen = 0
        self._symm_pending: List[Tuple[int, int]] = []
        self._symm_sink = _SegmentSink()
        self._symm_vmm = VmmGraphInputManager(
            obj=self._symm_sink,
            group=self.group,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.disabled = False

    def _init_workspace(self) -> None:
        """Slice one symmetric-memory allocation into every shared buffer.

        Layout per rank: ``[2 * world_size push slots | pull buffer | pull
        semaphores]``. One allocation rather than three keeps it to a single
        rendezvous and a single multicast base to offset from. The push
        counter is rank-local, so it lives in a plain CUDA tensor.
        """
        cfg = self.config
        push_num_slots = 2 * self.world_size  # 2 phases x world_size peers
        push_bytes = push_num_slots * self.max_push_size
        pull_bytes = self.max_pull_size  # 0 when the pull half is disabled
        sem_bytes = _SEMAPHORE_BYTES * cfg.num_pull_blocks if self.pull_enabled else 0
        total_bytes = push_bytes + pull_bytes + sem_bytes
        pull_offset = push_bytes
        sem_offset = push_bytes + pull_bytes

        self._symm_tensor, symm_mem = _allocate_symmetric_memory(
            total_bytes, device=self.device, group=self.group
        )
        slabs = [
            symm_mem.get_buffer(i, [total_bytes], torch.uint8)
            for i in range(self.world_size)
        ]
        # The push slots (lamport pos-zero markers) and the semaphores must
        # start zeroed; the pull buffer need not, but it rides along in the
        # one-shot memset.
        slabs[self.rank].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

        def slice_all(shape: List[int], offset: int) -> List[torch.Tensor]:
            """The same sub-range of every rank's slab, one view per rank."""
            nbytes = 1
            for dim in shape:
                nbytes *= dim
            assert offset + nbytes <= total_bytes
            return [slab[offset : offset + nbytes].view(shape) for slab in slabs]

        multicast_ptr = int(symm_mem.multicast_ptr)
        self.has_multicast = multicast_ptr != 0

        def mc_at(offset: int) -> Optional[int]:
            return multicast_ptr + offset if self.has_multicast else None

        self._push_counter = torch.zeros(
            (cfg.num_push_blocks,), dtype=torch.uint32, device=self.device
        )
        push_plane = PushPlane(
            self.rank,
            self.world_size,
            workspaces=slice_all([push_num_slots, self.max_push_size], 0),
            counter=self._push_counter.view(-1, 1).view(torch.uint8),
            mc_workspace=mc_at(0),
        )
        pull_plane = None
        if self.pull_enabled:
            pull_plane = PullPlane(
                self.rank,
                self.world_size,
                workspaces=slice_all([pull_bytes], pull_offset),
                semaphores=slice_all(
                    [cfg.num_pull_blocks, _SEMAPHORE_BYTES], sem_offset
                ),
                mc_workspace=mc_at(pull_offset),
                mc_semaphore=mc_at(sem_offset),
            )
        if not self.has_multicast or not self.pull_enabled:
            self.config = self.config._replace(
                graph=tuple(r for r in self.config.graph if r[1] != MC),
                eager=tuple(r for r in self.config.eager if r[1] != MC),
                num_mc_blocks=None,
            )

        self.obj = Communicator(push=push_plane, pull=pull_plane)
        if self.config.num_mc_blocks is not None:
            self.obj.set_pull_multicast_blocks(self.config.num_mc_blocks)
        if self.rank == 0:
            logger.info(
                "All Reduce config: symmetric_memory = %.2f MB, "
                "local_buffer = %.2f MB, multicast = %s, pull = %s",
                total_bytes / MB,
                (self.graph_params.nbytes + self._push_counter.nbytes) / MB,
                self.config.num_mc_blocks is not None,
                self.pull_enabled,
            )
        dist.barrier(group=self.group)

    # ------------------------------------------------------------------
    # Algo selection
    # ------------------------------------------------------------------

    def uncap_pull_thresholds(self) -> None:
        """Raise the 2-shot ceiling to the workspace capacity.

        The tuned tables end at the size where NCCL takes over; benchmarks
        and tests that must keep every sweep size on the custom-AR path can
        extend them with ``2shot_pull`` up to ``max_pull_size``.
        """
        self._add_rows(tail=((math.inf, PULL2),))

    def _add_rows(self, head: tuple = (), tail: tuple = ()) -> None:
        self.config = self.config._replace(
            graph=head + self.config.graph + tail,
            eager=head + self.config.eager + tail,
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

    def _pick_config(
        self, nbytes: int, can_use_graph: bool
    ) -> Optional[Tuple[AllReduceAlgo, bool]]:
        table = self.config.graph if can_use_graph else self.config.eager
        for max_bytes, (algo, use_multicast) in table:
            cap = self.max_push_size if algo.is_push() else self.max_pull_size
            if nbytes <= min(max_bytes, cap):
                return algo, use_multicast
        return None

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.override_algo is not None:
            return inp_size <= self.max_size
        zero_copy = self._can_use_graph() or self._lookup_symmetric(inp) is not None
        return self._pick_config(inp_size, zero_copy) is not None

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        can_use_graph = self._can_use_graph()
        symm = None if can_use_graph else self._lookup_symmetric(input)
        if self.override_algo is not None:
            # TODO: enhance this override pattern
            algo, use_multicast = self.override_algo, False
        else:
            # a symmetric input is as zero-copy as a graph-registered one
            config = self._pick_config(nbytes, can_use_graph or symm is not None)
            assert config is not None, f"No config for {nbytes = }"
            algo, use_multicast = config
        staged = algo.is_push() or use_multicast
        use_graph = can_use_graph and not staged
        graph_params = self._allocate_graph_row(input, nbytes) if use_graph else None
        peer_bases, peer_offset = (None, 0) if staged or symm is None else symm
        return custom_all_reduce(
            self.obj,
            input,
            algo=algo,
            graph_params=graph_params,
            use_multicast=use_multicast,
            peer_bases=peer_bases,
            peer_offset=peer_offset,
        )

    # ------------------------------------------------------------------
    # Symmetric-pool inputs
    # ------------------------------------------------------------------

    def _lookup_symmetric(
        self, input: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, int]]:
        """Every rank's segment base and the offset of ``input``, when it lies
        in a peer-mapped NCCL symmetric-pool segment."""
        if not self._symm_supported or self._in_capture:
            return None
        if is_in_tc_piecewise_cuda_graph():
            return None
        ptr = input.data_ptr()
        end = ptr + input.numel() * input.element_size()
        i = bisect.bisect_right(self._symm_bases, ptr) - 1
        if i < 0 or end > self._symm_segments[i][1]:
            if not self._map_symmetric_segments(ptr, end):
                return None
            i = bisect.bisect_right(self._symm_bases, ptr) - 1
            if i < 0 or end > self._symm_segments[i][1]:
                return None
        base, _, peer_bases = self._symm_segments[i]
        return peer_bases, ptr - base

    def _map_symmetric_segments(self, ptr: int, end: int) -> bool:
        """Peer-map the pending pool segments once an input lands in one.

        Collective over the group. Like NCCL's own window registration, it
        relies on every rank allocating the pool identically, so all ranks
        see the same segments and take this branch on the same call.
        """
        new = get_symmetric_segments(self._symm_seen)
        self._symm_seen += len(new)
        self._symm_pending += [seg for seg in new if is_vmm_pointer(seg[0])]
        if not any(b <= ptr and end <= b + n for b, n in self._symm_pending):
            return False
        segments, self._symm_pending = self._symm_pending, []
        self._symm_sink.segments = segments
        self._symm_vmm.register_graph_inputs()
        for (base, size), row in zip(segments, self._symm_sink.peer_ptrs):
            k = bisect.bisect_left(self._symm_bases, base)
            self._symm_bases.insert(k, base)
            self._symm_segments.insert(
                k, (base, base + size, torch.tensor(row, dtype=torch.int64))
            )
        return True

    def _allocate_graph_row(self, input: torch.Tensor, nbytes: int) -> torch.Tensor:
        index = self._graph_counter + len(self._graph_inputs)
        assert index < _MAX_GRAPH_INPUTS, "Graph input table overflow"
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
            self._graph_mode_allowed = self._is_graph_mode_supported
            self._in_capture = True
            yield
        finally:
            self._graph_mode_allowed = False
            self._in_capture = False
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
        if hasattr(self, "_symm_vmm"):
            self._symm_vmm.close()

    def __del__(self):
        self.close()


class _SegmentSink:
    """Feeds pool segments through ``VmmGraphInputManager`` (which maps
    graph inputs) and keeps each segment's per-rank bases."""

    def __init__(self) -> None:
        self.segments: List[Tuple[int, int]] = []
        self.peer_ptrs: List[List[int]] = []

    def get_graph_capture_bases(self):
        return compute_graph_capture_bases(self.segments)

    def register_peer_mapped_inputs(self, peer_ptrs: List[List[int]]) -> None:
        self.peer_ptrs = peer_ptrs


def _is_vmm_backed_allocator(device: torch.device) -> bool:
    """Check whether expandable-segments VMM backs the caching allocator."""
    probe = torch.empty(1, dtype=torch.uint8, device=device)
    return is_vmm_pointer(probe.data_ptr())


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
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

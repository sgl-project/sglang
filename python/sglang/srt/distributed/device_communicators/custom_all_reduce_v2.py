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
from contextlib import contextmanager
from typing import List, NamedTuple, Optional, Tuple

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

from .configs.custom_all_reduce_v2 import (
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


class AllReduceConfig(NamedTuple):
    algo: AllReduceAlgo
    use_graph: bool = False
    use_multicast: bool = False


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
            max_pull_size = min(base_config.max_pull_bytes, max_size)
        if max_push_size is None:
            max_push_size = min(base_config.max_push_bytes, max_size)
        if _FORCE_PULL_SIZE_KB is not None:
            max_pull_size = int(_FORCE_PULL_SIZE_KB) * 1024
        if _FORCE_PUSH_SIZE_KB is not None:
            max_push_size = int(_FORCE_PUSH_SIZE_KB) * 1024

        def force_thresholds(heuristic):
            # forced sizes bypass the tuned NCCL-crossover heuristics: lift
            # each direction's ceiling to the forced workspace capacity (the
            # clip() below only ever lowers thresholds)
            if _FORCE_PULL_SIZE_KB is not None:
                heuristic = heuristic._replace(two_shot_pull_threshold=max_pull_size)
            if _FORCE_PUSH_SIZE_KB is not None:
                heuristic = heuristic._replace(one_shot_push_threshold=max_push_size)
            return heuristic

        base_config = base_config._replace(
            graph=force_thresholds(base_config.graph),
            eager=force_thresholds(base_config.eager),
        )
        # Zero on either pull knob opts out of the pull half entirely: no
        # pull plane at all, and every pull algo disabled below by clipping
        # its threshold to 0. Push-only callers (the fused qk-norm instances)
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
        self.config = base_config.clip(
            max_push_bytes=self.max_push_size, max_pull_bytes=self.max_pull_size
        )._replace(num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks)
        self.override_algo: Optional[AllReduceAlgo] = None
        # On a multi-node (MNNVL) group the symm-mem workspace plane works
        # across nodes, but graph zero-copy input registration (cudaIpc /
        # node-local VMM remap) does not — force eager pull inside graphs.
        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self._is_graph_mode_supported = not tms_cudagraph and not is_multinode
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
            self.config = self.config._replace(num_mc_blocks=None)

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

        The tuned config caps ``2shot_pull`` at the size where NCCL takes
        over; benchmarks and tests that must keep every sweep size on the
        custom-AR path can lift that cap up to ``max_pull_size``.
        """

        if not self.pull_enabled:
            return

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

    def _pick_config(self, nbytes: int, can_use_graph: bool) -> AllReduceConfig | None:
        # TODO: refactor this along with the config file
        heuristic = self.config.graph if can_use_graph else self.config.eager
        can_use_multicast = self.config.num_mc_blocks is not None
        if nbytes <= heuristic.one_shot_push_threshold:
            return AllReduceConfig(AllReduceAlgo.ONE_SHOT_PUSH)
        if nbytes <= heuristic.one_shot_pull_threshold:
            return AllReduceConfig(AllReduceAlgo.ONE_SHOT_PULL, use_graph=can_use_graph)
        if can_use_multicast and heuristic.mc.contains(nbytes):
            return AllReduceConfig(AllReduceAlgo.TWO_SHOT_PULL, use_multicast=True)
        if nbytes <= heuristic.two_shot_pull_threshold:
            return AllReduceConfig(AllReduceAlgo.TWO_SHOT_PULL, use_graph=can_use_graph)
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
        return self._pick_config(inp_size, self._can_use_graph()) is not None

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        if self.override_algo is not None:
            # TODO: enhance this override pattern
            algo = self.override_algo
            use_graph = self._can_use_graph() and not algo.is_push()
            use_multicast = False
        else:
            config = self._pick_config(nbytes, self._can_use_graph())
            assert config is not None, f"No config for {nbytes = }"
            algo, use_graph, use_multicast = config
        graph_params = self._allocate_graph_row(input, nbytes) if use_graph else None
        return custom_all_reduce(
            self.obj,
            input,
            algo=algo,
            graph_params=graph_params,
            use_multicast=use_multicast,
        )

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

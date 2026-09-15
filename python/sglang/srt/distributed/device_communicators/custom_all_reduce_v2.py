"""JIT custom all-reduce (v2) over a decoupled storage plane.

The CUDA side is split into independent pieces:

- ``PushPlane``: the zero-filled symmetric push buffers plus a rank-local
  phase counter. ``1shot_push`` and ``2shot_push`` share this plane: the
  former treats it as one full-input slot per source, while the latter lays
  the reduced shards contiguously across those same slots in its all-gather
  half. When ``2shot_push`` is enabled the plane doubles its slot rows and
  the second half holds the per-source shard inbox, so every push-family
  algorithm advances one double buffer together.
- ``PullPlane``: the symmetric pull buffers plus the per-block semaphores
  guarding them. The buffers exist only because this class hands the
  all-reduce tensors that are not symmetric memory, so it stages them
  through; the K3 fused collectives bring their own symmetric input and
  borrow the semaphores alone.
- ``Communicator``: the planes above (either may be absent), the handle every
  kernel takes, plus the pull launch widths.
- the all-reduce kernel: a pure function of ``(Communicator, input, algo,
  graph_params, use_multicast)`` with four algorithms (1shot_push /
  1shot_pull / 2shot_pull / 2shot_push) and three pull data sources (eager
  pull buffer / CUDA-graph pointer table / multicast address).

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

from .configs.custom_all_reduce_v2 import AllReduceConfig as TunedConfig
from .configs.custom_all_reduce_v2 import (
    Heuristic,
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


def _forced_size(env_kb: Optional[int]) -> Optional[int]:
    """Bytes behind a ``SGLANG_FORCE_*_SIZE_KB`` env var; ``None`` when unset."""
    return None if env_kb is None else int(env_kb) * 1024


# forced per-direction workspace sizes; highest priority, they override both
# the tuned config and explicit constructor sizes
_FORCE_PULL_SIZE = _forced_size(
    envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB.get()
)
_FORCE_PUSH_SIZE = _forced_size(
    envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB.get()
)


def _div_ceil(nbytes: int, div: int) -> int:
    return (nbytes + div - 1) // div


def _align(nbytes: int) -> int:
    """Round a workspace size up to ``_ALIGN_BYTES``, never down to zero."""
    nbytes = max(nbytes, _ALIGN_BYTES)
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


class AllReduceConfig(NamedTuple):
    algo: AllReduceAlgo
    use_graph: bool = False
    use_multicast: bool = False


class _WorkspaceSizes(NamedTuple):
    """Aligned byte capacities of one instance's symmetric-memory workspace.

    ``pull`` and ``push`` are the per-direction message ceilings; either may
    be ``0``. ``push_slot`` is the push plane's one slot size: big enough
    for a whole ``1shot_push`` input, and for one ``2shot_push`` shard (the
    scatter inbox and the gather half both hold one shard per slot), with
    ``two_shot_push`` the message ceiling that buys.
    """

    pull: int
    push: int
    push_slot: int
    two_shot_push: int

    @property
    def pull_enabled(self) -> bool:
        return self.pull > 0

    @property
    def two_shot_push_enabled(self) -> bool:
        return self.two_shot_push > 0

    @property
    def max_message(self) -> int:
        """Largest message any algorithm on this workspace can take."""
        return max(self.pull, self.push)

    @classmethod
    def resolve(
        cls,
        tuned: TunedConfig,
        world_size: int,
        *,
        max_size: int,
        max_pull_size: Optional[int],
        max_push_size: Optional[int],
        max_two_shot_push_size: Optional[int],
        pull_blocks_disabled: bool,
    ) -> "_WorkspaceSizes":
        """Resolve the constructor's optional caps into concrete capacities.

        Precedence, highest first: the ``SGLANG_FORCE_*`` env vars, the
        explicit constructor argument, then what the tuned config wants
        clipped to ``max_size``.
        """
        if max_pull_size is None:
            max_pull_size = min(tuned.max_pull_bytes, max_size)
        if max_push_size is None:
            max_push_size = min(tuned.max_push_bytes, max_size)
        if max_two_shot_push_size is None:
            max_two_shot_push_size = min(tuned.max_two_shot_push_bytes, max_size)
        if _FORCE_PULL_SIZE is not None:
            max_pull_size = _FORCE_PULL_SIZE
        if _FORCE_PUSH_SIZE is not None:
            max_push_size = _FORCE_PUSH_SIZE

        # Zero on either pull knob opts out of the pull half entirely: no
        # pull plane at all, and every pull algo disabled by clipping its
        # threshold to 0. Push-only callers (the fused qk-norm instances)
        # take this path rather than allocating a placeholder buffer just to
        # satisfy a constructor.
        pull_off = pull_blocks_disabled or max_pull_size <= 0
        # Every 2shot_push slot -- scatter inbox or gather half -- holds one
        # shard, so the shard size and the 1shot input ceiling together fix
        # the plane's single slot size.
        shard = (
            0
            if max_two_shot_push_size <= 0
            else _align(_div_ceil(max_two_shot_push_size, world_size))
        )
        return cls(
            pull=0 if pull_off else _align(max_pull_size),
            push=_align(max_push_size),
            push_slot=max(_align(max_push_size), shard),
            two_shot_push=shard * world_size,
        )


def _narrow_config(
    tuned: TunedConfig,
    sizes: _WorkspaceSizes,
    *,
    max_pull_blocks: Optional[int],
    max_push_blocks: Optional[int],
) -> TunedConfig:
    """Narrow a tuned config to what this instance actually allocated.

    Every algo's size threshold is clipped to its workspace capacity, and
    the kernel grids to the caller's block limits.
    """

    def force_thresholds(heuristic: Heuristic) -> Heuristic:
        # forced sizes bypass the tuned NCCL-crossover heuristics: lift
        # each direction's ceiling to the forced workspace capacity (the
        # clip() below only ever lowers thresholds)
        if _FORCE_PULL_SIZE is not None:
            heuristic = heuristic._replace(two_shot_pull_threshold=_FORCE_PULL_SIZE)
        if _FORCE_PUSH_SIZE is not None:
            heuristic = heuristic._replace(one_shot_push_threshold=_FORCE_PUSH_SIZE)
        return heuristic

    # a cap, and `0` is not one: it means "push-only instance", which leaves
    # no pull plane to size in the first place
    num_pull_blocks = tuned.num_pull_blocks
    if max_pull_blocks:
        num_pull_blocks = max(min(num_pull_blocks, max_pull_blocks), 1)
    # not a cap but an override: callers size the push grid by occupancy,
    # which may land either side of the tuned default
    num_push_blocks = tuned.num_push_blocks
    if max_push_blocks is not None:
        num_push_blocks = max(max_push_blocks, 1)
    return (
        tuned._replace(
            graph=force_thresholds(tuned.graph),
            eager=force_thresholds(tuned.eager),
        )
        .clip(
            max_push_bytes=sizes.push,
            max_pull_bytes=sizes.pull,
            max_two_shot_push_bytes=sizes.two_shot_push,
        )
        ._replace(num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks)
    )


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_size: int = _DEFAULT_MAX_SIZE,
        *,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
        max_two_shot_push_size: Optional[int] = None,
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
        :param max_two_shot_push_size: explicit ``2shot_push`` message
                                 ceiling; sizes the scatter slots and may
                                 enlarge the shared push slots. Overrides both
                                 the tuned size and ``max_size``; ``0``
                                 disables the algo and its plane.
        :param max_pull_blocks: cap on the barrier plane's block count; ``0``
                                builds a push-only instance.
        :param max_push_blocks: exact push grid width, overriding the tuned
                                one in either direction.

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
        # A multi-node (MNNVL) group keeps the symm-mem workspace plane but
        # loses graph zero-copy input registration (cudaIpc / node-local VMM
        # remap), and its crossovers differ from a single node at the same
        # world size — so it picks both a different config and eager-only
        # dispatch below.
        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tuned = get_all_reduce_config(self.world_size, multinode=is_multinode)
        sizes = _WorkspaceSizes.resolve(
            tuned,
            self.world_size,
            max_size=max_size,
            max_pull_size=max_pull_size,
            max_push_size=max_push_size,
            max_two_shot_push_size=max_two_shot_push_size,
            pull_blocks_disabled=max_pull_blocks == 0,
        )
        self.pull_enabled = sizes.pull_enabled
        self.two_shot_push_enabled = sizes.two_shot_push_enabled
        self.max_pull_size = sizes.pull
        self.max_push_size = sizes.push
        self.max_two_shot_push_size = sizes.two_shot_push
        self.push_slot_size = sizes.push_slot
        self.max_size = sizes.max_message
        self.config = _narrow_config(
            tuned,
            sizes,
            max_pull_blocks=max_pull_blocks,
            max_push_blocks=max_push_blocks,
        )
        self.override_algo: Optional[AllReduceAlgo] = None
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

        Layout per rank: ``[2 * world_size push slots (4 * world_size when
        ``2shot_push`` is enabled) | pull buffer | pull semaphores]``. The
        first ``2 * world_size`` slots serve ``1shot_push`` and the two-shot
        gather half; the doubled tail is ``2shot_push``'s per-source shard
        inbox, recognised by the plane from the row count, so one rank-local
        counter (a plain CUDA tensor) drives the epoch of every push-family
        algorithm.
        """
        cfg = self.config
        # 2 phases x world_size peers, doubled again for the scatter region
        push_num_slots = (4 if self.two_shot_push_enabled else 2) * self.world_size
        push_bytes = push_num_slots * self.push_slot_size
        pull_bytes = self.max_pull_size  # 0 when the pull half is disabled
        num_sem_blocks = cfg.num_pull_blocks if self.pull_enabled else 0
        sem_bytes = _SEMAPHORE_BYTES * num_sem_blocks
        total_bytes = push_bytes + pull_bytes + sem_bytes
        pull_offset = push_bytes
        sem_offset = pull_offset + pull_bytes

        self._symm_tensor, symm_mem = _allocate_symmetric_memory(
            total_bytes, device=self.device, group=self.group
        )
        slabs = [
            symm_mem.get_buffer(i, [total_bytes], torch.uint8)
            for i in range(self.world_size)
        ]
        # The shared push slots (Lamport pos-zero markers) and semaphores must
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
        counter = self._push_counter.view(-1, 1).view(torch.uint8)
        push_plane = PushPlane(
            self.rank,
            self.world_size,
            workspaces=slice_all([push_num_slots, self.push_slot_size], 0),
            counter=counter,
            mc_workspace=mc_at(0),
        )
        pull_plane = None
        if self.pull_enabled:
            pull_plane = PullPlane(
                self.rank,
                self.world_size,
                workspaces=slice_all([pull_bytes], pull_offset),
                semaphores=slice_all([num_sem_blocks, _SEMAPHORE_BYTES], sem_offset),
                mc_workspace=mc_at(pull_offset),
                mc_semaphore=mc_at(sem_offset),
            )
        if not self.has_multicast or not self.pull_enabled:
            self.config = self.config._replace(num_mc_blocks=None)

        self.obj = Communicator(push=push_plane, pull=pull_plane)
        if self.pull_enabled:
            self.obj.set_pull_blocks(self.config.num_pull_blocks)
        if self.config.num_mc_blocks is not None:
            self.obj.set_pull_multicast_blocks(self.config.num_mc_blocks)
        if self.rank == 0:
            logger.info(
                "All Reduce config: symmetric_memory = %.2f MB, "
                "local_buffer = %.2f MB, multicast = %s, pull = %s, "
                "2shot_push = %s",
                total_bytes / MB,
                (self.graph_params.nbytes + self._push_counter.nbytes) / MB,
                self.config.num_mc_blocks is not None,
                self.pull_enabled,
                self.two_shot_push_enabled,
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
        # the bands are tried first so they may start below the thresholds
        if heuristic.two_shot_push.contains(nbytes):
            return AllReduceConfig(
                AllReduceAlgo.TWO_SHOT_PUSH,
                use_multicast=self.has_multicast
                and heuristic.two_shot_push_mc.contains(nbytes),
            )
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
            use_graph = self._can_use_graph() and algo.supports_zero_copy()
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

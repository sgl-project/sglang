"""JIT custom all-reduce (v2) over a decoupled storage plane.

The CUDA side is split into independent pieces:

- ``PushPlane``: the zero-filled symmetric push buffers plus a rank-local
  phase word shared by every push kernel.
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

The storage lives in the per-group symmetric workspace (``symm_workspace.py``),
shared with the other push/pull kernels; this class requests its share.

On single-node groups, CUDA-graph inputs are exchanged from Python after
capture (cudaIpc handles for cudaMalloc-backed pointers, FABRIC/POSIX-fd VMM mapping
for expandable segments) and written into a device-side pointer table
(``_peer_ptr_table``); the kernel captured in the graph dereferences its row at
replay time. Multi-node (MNNVL) groups skip direct graph-input registration
and use the torch symmetric-memory workspace during capture and replay too.
Their input tensors therefore do not require a VMM-backed caching allocator.
"""

import logging
from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.kernels.kernel_api_logging import _KERNEL_API_LOG_LEVEL
from sglang.kernels.ops.communication.all_reduce import (
    SUPPORTED_DTYPES,
    AllReduceAlgo,
    Communicator,
    IPCManager,
    custom_all_reduce,
    get_all_reduce_module,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.cuda_vmm_utils import (
    VmmGraphInputManager,
    compute_graph_capture_bases,
    is_vmm_pointer,
)

from .configs.custom_all_reduce_v2 import (
    AllReduceConfig,
    get_all_reduce_config,
    get_supported_world_sizes,
)
from .custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_one_nvlink_clique,
    is_weak_contiguous,
)
from .dpx_hier_all_reduce import DpxHierAllReduce
from .symm_workspace import (
    WorkspaceRequest,
    register_v2_push_slot_bytes,
    request_symm_workspace,
)

if TYPE_CHECKING:
    from tvm_ffi import Module

    from .symm_workspace import WorkspaceLease

logger = logging.getLogger(__name__)

_is_hip = is_hip()

_ALIGN_BYTES = 1024
_MAX_GRAPH_INPUTS = 131072
# resolved once at import time; explicit constructor sizes take precedence
_DEFAULT_MAX_SIZE = envs.SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB.get() * 1024
# forced per-direction workspace sizes; highest priority, override both the
# tuned config and explicit constructor sizes (None = not forced)
_FORCE_PULL_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PULL_SIZE_KB.get()
_FORCE_PUSH_SIZE_KB = envs.SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB.get()


def _ceil_align(nbytes: int, align: int) -> int:
    return (nbytes + align - 1) // align * align


# (JIT module, algorithm name, multicast) of one eager launch
_EagerPlan = Tuple["Module", str, bool]


class AllReduceChoice(NamedTuple):
    algo: AllReduceAlgo
    use_graph: bool = False
    use_multicast: bool = False
    # hier_2shot_push (ROCm gfx1250 DPX): a 2shot_push split into pair and cross
    # stages; ``algo`` is TWO_SHOT_PUSH
    use_hier: bool = False


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
        :param max_pull_blocks: cap on the pull kernels' default launch width;
                                ``0`` builds a push-only instance.
        :param max_push_blocks: the push kernels' default launch width.

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
        dpx_pairs, hier_max_bytes = None, 0
        if _is_hip:
            base_config, dpx_pairs, hier_max_bytes = DpxHierAllReduce.plan(
                group=group,
                device=device,
                config=base_config,
                max_size=max_size,
                sizes_own_push=max_push_size is not None
                or max_pull_blocks == 0
                or _FORCE_PUSH_SIZE_KB is not None,
            )
        if max_pull_size is None:
            max_pull_size = min(base_config.max_pull_bytes, max_size)
        if max_push_size is None:
            max_push_size = min(base_config.max_push_bytes(self.world_size), max_size)
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
        self._has_pull = max_pull_blocks != 0 and max_pull_size > 0
        self._max_pull_size = (
            _ceil_align(max(max_pull_size, _ALIGN_BYTES), _ALIGN_BYTES)
            if self._has_pull
            else 0
        )
        self._max_push_size = _ceil_align(
            max(max_push_size, _ALIGN_BYTES), _ALIGN_BYTES
        )
        self.max_size = max(self._max_pull_size, self._max_push_size)
        num_pull_blocks = base_config.num_pull_blocks
        num_push_blocks = base_config.num_push_blocks
        if max_pull_blocks:
            num_pull_blocks = max(min(num_pull_blocks, max_pull_blocks), 1)
        if max_push_blocks is not None:
            num_push_blocks = max(max_push_blocks, 1)
        self._eager_plan_cache: Dict[Tuple[int, torch.dtype], Optional[_EagerPlan]] = {}
        self.config = base_config.clip(
            max_push_bytes=self._max_push_size,
            max_pull_bytes=self._max_pull_size,
            world_size=self.world_size,
        )._replace(num_pull_blocks=num_pull_blocks, num_push_blocks=num_push_blocks)
        self.forced_algo: Optional[AllReduceAlgo] = None
        # On a multi-node (MNNVL) group the symm-mem workspace plane works
        # across nodes, but graph zero-copy input registration (cudaIpc /
        # node-local VMM remap) does not — force eager pull inside graphs.
        # On ROCm the 2-shot graph path would store into peers' cached input
        # buffers, which XGMI does not keep coherent; stage through the
        # uncached workspace inside graphs instead.
        is_multinode = not all(in_the_same_node_as(group, source_rank=0))
        tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        self._supports_graph_inputs = not (tms_cudagraph or is_multinode or _is_hip)
        # device-side pointer table: one row of world_size pointers per
        # graph-captured all-reduce input
        self._peer_ptr_table: torch.Tensor = torch.zeros(
            (_MAX_GRAPH_INPUTS, self.world_size),
            dtype=torch.uint64,
            device=self.device,
        )
        register_v2_push_slot_bytes(group, self._max_push_size)
        _lease = request_symm_workspace(
            group,
            device,
            WorkspaceRequest(
                owner="custom_all_reduce_v2",
                push_slot_bytes=self._max_push_size,
                pull_bytes=self._max_pull_size,
                semaphores=num_pull_blocks if self._has_pull else 0,
            ),
        )
        if _lease is None:
            return
        self._lease: WorkspaceLease = _lease
        if not _lease.has_multicast or not self._has_pull:
            self.config = self.config._replace(num_mc_blocks=None)
        self._hier: Optional[DpxHierAllReduce] = None
        if hier_max_bytes:
            self._hier = DpxHierAllReduce(
                group=group,
                device=device,
                rank=self.rank,
                world_size=self.world_size,
                pairs=dpx_pairs,
                max_bytes=hier_max_bytes,
                push_blocks=self.config.num_push_blocks,
            )
        # only graph-input registration opens IPC handles
        self._ipc_manager: Optional[IPCManager] = (
            IPCManager() if self._supports_graph_inputs else None
        )
        self._vmm_graph_input_manager = VmmGraphInputManager(
            obj=self,
            group=self.group,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._pending_graph_inputs: List[Tuple[int, int]] = []  # (data_ptr, nbytes)
        self._num_registered_rows = 0
        self._in_capture = False  # inside capture()
        self.disabled = False

    @cached_property
    def obj(self) -> Communicator:
        obj = self._lease.get_communicator(
            push_blocks=self.config.num_push_blocks,
            pull_blocks=self.config.num_pull_blocks if self._has_pull else None,
        )
        if self.config.num_mc_blocks is not None:
            obj.set_pull_multicast_blocks(self.config.num_mc_blocks)
        return obj

    @property
    def config(self) -> AllReduceConfig:
        return self._config

    @config.setter
    def config(self, config: AllReduceConfig) -> None:
        # the cached eager launches were chosen under the old table
        self._config = config
        self._eager_plan_cache.clear()

    @cached_property
    def has_multicast(self) -> bool:
        return self._lease.has_multicast

    # ------------------------------------------------------------------
    # Algo selection
    # ------------------------------------------------------------------

    def uncap_pull_thresholds(self) -> None:
        """Raise the 2-shot ceiling to the workspace capacity.

        The tuned config caps ``2shot_pull`` at the size where NCCL takes
        over; benchmarks and tests that must keep every sweep size on the
        custom-AR path can lift that cap up to ``max_pull_size``.

        NOTE: This is a test-only API
        """

        if not self._has_pull:
            return

        def uncap(heuristic):
            return heuristic._replace(two_shot_pull_threshold=self._max_pull_size)

        self.config = self.config._replace(
            graph=uncap(self.config.graph),
            eager=uncap(self.config.eager),
        )

    def _zero_copy_inputs(self) -> bool:
        """Inside capture(), on a group whose graph inputs are registered by pointer."""
        return self._in_capture and self._supports_graph_inputs

    def _can_use_graph(self) -> bool:
        return (
            self._zero_copy_inputs()
            and not is_in_tc_piecewise_cuda_graph()
            and torch.cuda.is_current_stream_capturing()
        )

    def _choose_algo(self, nbytes: int, can_use_graph: bool) -> AllReduceChoice | None:
        # TODO: refactor this along with the config file
        heuristic = self.config.graph if can_use_graph else self.config.eager
        can_use_multicast = self.config.num_mc_blocks is not None
        if nbytes <= heuristic.one_shot_push_threshold:
            return AllReduceChoice(AllReduceAlgo.ONE_SHOT_PUSH)
        # the pair halves must stay whole 16 B vectors
        if nbytes % 32 == 0 and heuristic.hier_2shot_push.contains(nbytes):
            return AllReduceChoice(AllReduceAlgo.TWO_SHOT_PUSH, use_hier=True)
        if nbytes <= heuristic.two_shot_push_threshold:
            return AllReduceChoice(AllReduceAlgo.TWO_SHOT_PUSH)
        if nbytes <= heuristic.one_shot_pull_threshold:
            return AllReduceChoice(AllReduceAlgo.ONE_SHOT_PULL, use_graph=can_use_graph)
        if can_use_multicast and heuristic.mc.contains(nbytes):
            return AllReduceChoice(AllReduceAlgo.TWO_SHOT_PULL, use_multicast=True)
        if nbytes <= heuristic.two_shot_pull_threshold:
            return AllReduceChoice(AllReduceAlgo.TWO_SHOT_PULL, use_graph=can_use_graph)
        return None

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        if self.disabled:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        if self.forced_algo is not None:
            return inp_size <= self.max_size
        if not self._zero_copy_inputs():
            return self._eager_plan(inp_size, inp.dtype) is not None
        return self._choose_algo(inp_size, self._can_use_graph()) is not None

    def _eager_plan(self, nbytes: int, dtype: torch.dtype):
        """The launch an eager all-reduce of this size takes, resolved once."""
        key = (nbytes, dtype)
        if key not in self._eager_plan_cache:
            choice = self._choose_algo(nbytes, False)
            if choice is None:
                self._eager_plan_cache[key] = None
            elif choice.use_hier:
                # runs over the pair and cross planes, not one module
                self._eager_plan_cache[key] = (None, "hier_2shot_push", False)
            else:
                self._eager_plan_cache[key] = (
                    get_all_reduce_module(dtype, self.world_size),
                    choice.algo.algo_name,
                    choice.use_multicast,
                )
        return self._eager_plan_cache[key]

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def uses_one_shot_push(self, nbytes: int) -> bool:
        """Use the normal algorithm policy, including explicit overrides."""
        if self.disabled or nbytes <= 0 or nbytes % 16 or nbytes > self._max_push_size:
            return False
        if self.forced_algo is not None:
            return self.forced_algo == AllReduceAlgo.ONE_SHOT_PUSH
        choice = self._choose_algo(nbytes, self._can_use_graph())
        return choice is not None and choice.algo == AllReduceAlgo.ONE_SHOT_PUSH

    def uses_two_shot_push(self, nbytes: int) -> bool:
        if self.disabled or nbytes <= 0 or nbytes % 16:
            return False
        if self.forced_algo is not None:
            return self.forced_algo == AllReduceAlgo.TWO_SHOT_PUSH
        choice = self._choose_algo(nbytes, self._can_use_graph())
        return choice is not None and choice.algo == AllReduceAlgo.TWO_SHOT_PUSH

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        nbytes = input.numel() * input.element_size()
        if (
            self.forced_algo is None
            and not self._zero_copy_inputs()
            and not _KERNEL_API_LOG_LEVEL
        ):
            plan = self._eager_plan(nbytes, input.dtype)
            assert plan is not None, f"No config for {nbytes = }"
            module, algo_name, use_multicast = plan
            if module is None:
                return self._hier.all_reduce(input)
            out = module.all_reduce(self.obj, input, algo_name, None, use_multicast)
            return out if isinstance(out, torch.Tensor) else torch.from_dlpack(out)
        if self.forced_algo is not None:
            # TODO: enhance this override pattern
            algo = self.forced_algo
            use_graph = self._can_use_graph() and not algo.is_push()
            use_multicast = use_hier = False
        else:
            choice = self._choose_algo(nbytes, self._can_use_graph())
            assert choice is not None, f"No algorithm for {nbytes = }"
            algo, use_graph, use_multicast, use_hier = choice
        if use_hier:
            return self._hier.all_reduce(input)
        graph_row = self._allocate_graph_row(input, nbytes) if use_graph else None
        return custom_all_reduce(
            self.obj,
            input,
            algo=algo,
            graph_params=graph_row,
            use_multicast=use_multicast,
        )

    def _allocate_graph_row(self, input: torch.Tensor, nbytes: int) -> torch.Tensor:
        index = self._num_registered_rows + len(self._pending_graph_inputs)
        assert index < _MAX_GRAPH_INPUTS, "Graph input table overflow"
        self._pending_graph_inputs.append((input.data_ptr(), nbytes))
        return self._peer_ptr_table[index]

    # ------------------------------------------------------------------
    # CUDA-graph input registration
    # ------------------------------------------------------------------

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        self._in_capture = True
        try:
            yield
        finally:
            self._in_capture = False
        assert not torch.cuda.is_current_stream_capturing(), (
            "Cannot register graph inputs while capturing CUDA graph"
        )
        self._register_graph_inputs()

    def _register_graph_inputs(self) -> None:
        if not self._pending_graph_inputs:
            return
        first_ptr = self._pending_graph_inputs[0][0]
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
        assert self._ipc_manager is not None
        ptrs = [ptr for ptr, _ in self._pending_graph_inputs]
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
        return compute_graph_capture_bases(self._pending_graph_inputs)

    # NOTE: also used in cuda_vmm_utils.py
    def register_peer_mapped_inputs(self, peer_ptrs: List[List[int]]) -> None:
        """Write per-input peer pointers into the device-side pointer table."""
        assert len(peer_ptrs) == len(self._pending_graph_inputs)
        count = len(peer_ptrs)
        rows = torch.tensor(peer_ptrs, dtype=torch.uint64, device=self.device)
        self._peer_ptr_table[
            self._num_registered_rows : self._num_registered_rows + count
        ].copy_(rows)
        # the rows must be visible before any (PDL-chained) graph replay
        torch.cuda.synchronize()
        self._num_registered_rows += count
        self._pending_graph_inputs.clear()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        # the storage belongs to the symmetric workspace, which frees it collectively
        if not self.disabled and "obj" in self.__dict__:
            if self._ipc_manager is not None:
                self._ipc_manager.destroy()
            del self.obj  # drop the pointer holder before the workspace tensors
        if not self.disabled and self._hier is not None:
            self._hier.close()
            self._hier = None
        if hasattr(self, "_vmm_graph_input_manager"):
            self._vmm_graph_input_manager.close()

    def __del__(self):
        self.close()


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
    if _is_hip:
        from sglang.srt.utils.hip_vmm_utils import (
            MIN_HIP_VERSION,
            hip_runtime_version,
        )

        version = hip_runtime_version()
        if version[:2] < MIN_HIP_VERSION:
            logger.warning(
                "CustomAllReduceV2 is disabled: HIP runtime %s is older than %s, "
                "the oldest it is validated on.",
                ".".join(map(str, version)),
                ".".join(map(str, MIN_HIP_VERSION)),
            )
            return False
    supported = get_supported_world_sizes()
    if dist.get_world_size(group=group) not in supported:
        return False
    if not all(in_the_same_node_as(group, source_rank=0)):
        if _is_hip:
            from .gfx1250_fabric import probe_fabric_clique

            # NVML cliques do not exist on ROCm; only a gfx1250 UALink fabric
            # clique can host a multi-node workspace
            return probe_fabric_clique(group, device)
        return is_one_nvlink_clique(group, device)
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(supported),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True

import contextlib
import logging
import platform
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_attn_tensor_model_parallel_rank,
    get_attn_tensor_model_parallel_world_size,
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_moe_tp_group,
    get_tp_group,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    ceil_align,
    get_cuda_driver_bindings,
    is_flashinfer_available,
    is_sm100_supported,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

# FlashInfer allreduce fusion: set when flashinfer is available (see block below)
_flashinfer_comm = None
_TorchDistBackend = None
_mnnvl_comm_backend = None
_create_allreduce_fusion_workspace = None
_flashinfer_allreduce_unavailable = False
_posix_transport_override_logged = False
_mnnvl_non_blackwell_fallback_logged = False


def _resolve_backend(backend: str) -> str:
    """Force any auto/mnnvl selection back to trtllm when not on SM100.

    MNNVL is only enabled on Blackwell GPUs (SM10x) where it has been validated;
    elsewhere we use TRT-LLM unconditionally.
    """
    global _mnnvl_non_blackwell_fallback_logged
    if backend in ("auto", "mnnvl") and not is_sm100_supported():
        if not _mnnvl_non_blackwell_fallback_logged:
            logger.info(
                "FlashInfer allreduce fusion: forcing trtllm backend "
                "(mnnvl is only enabled on Blackwell systems)."
            )
            _mnnvl_non_blackwell_fallback_logged = True
        return "trtllm"
    return backend


def flashinfer_mnnvl_allreduce_fusion_enabled(server_args) -> bool:
    """True when FlashInfer is configured to (potentially) run MNNVL allreduce fusion.

    MNNVL has a known piecewise-CUDA-graph replay hang (Lamport spin in the
    FlashInfer MNNVL path), so callers use this to skip PCG capture entirely.
    Returns True if the user selected ``mnnvl`` explicitly, or if ``auto`` is
    used on a Blackwell system where the auto-resolver may pick mnnvl.
    """
    backend = getattr(server_args, "flashinfer_allreduce_fusion_backend", None)
    if backend is None:
        return False
    return _resolve_backend(backend) == "mnnvl" or (
        backend == "auto" and is_sm100_supported()
    )


def _should_force_posix_fd_transport() -> bool:
    force_posix_env = envs.SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT.get()
    if force_posix_env is not None:
        return force_posix_env

    machine = platform.machine().lower()
    if machine not in ("aarch64", "arm64"):
        return False

    if not torch.cuda.is_available():
        return False

    try:
        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception as e:
        logger.debug("Failed to get CUDA device capability: %s", e)
        return False

    return major == 10


@contextlib.contextmanager
def _flashinfer_posix_fd_transport_override_if_needed():
    # TODO(mmangkad): Remove this temporary override once the
    # FlashInfer unified allreduce-fusion transport issue on
    # GB200/GB300 platforms is fixed and verified resolved.
    global _posix_transport_override_logged

    if not _should_force_posix_fd_transport():
        yield
        return

    try:
        import flashinfer.comm.mnnvl as flashinfer_mnnvl
    except Exception as e:
        logger.debug(
            "Failed to import flashinfer.comm.mnnvl for transport override: %s", e
        )
        yield
        return

    original_checker = getattr(flashinfer_mnnvl, "is_mnnvl_fabric_supported", None)
    if original_checker is None:
        yield
        return

    if not _posix_transport_override_logged:
        logger.warning(
            "Applying FlashInfer transport workaround: forcing PosixFD "
            "symmetric-memory handle exchange on aarch64 + sm10x to avoid "
            "known data corruption with Fabric handle exchange on GB systems. "
            "Set SGLANG_FLASHINFER_FORCE_POSIX_FD_TRANSPORT=0 to disable."
        )
        _posix_transport_override_logged = True

    def _always_disable_fabric(_device_idx: int) -> bool:
        return False

    flashinfer_mnnvl.is_mnnvl_fabric_supported = _always_disable_fabric
    try:
        yield
    finally:
        flashinfer_mnnvl.is_mnnvl_fabric_supported = original_checker


if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        _flashinfer_comm = comm
        _create_allreduce_fusion_workspace = comm.create_allreduce_fusion_workspace
    except (ImportError, AttributeError) as e:
        logger.warning(
            "flashinfer.comm allreduce_fusion API is not available (%s), "
            "falling back to standard implementation",
            e,
        )

    try:
        from flashinfer.comm.mnnvl import TorchDistBackend

        class _FixedTorchDistBackend(TorchDistBackend):
            """Workaround for FlashInfer TorchDistBackend issues.

            1. bcast fix: TorchDistBackend.bcast passes the in-group rank
               directly as `src` to broadcast_object_list, which expects a
               global rank.
            2. Graph-capture fix: initialize with NCCL device_group (so
               the backend derives correct device_idx / GPU mapping), but
               broadcast via GLOO cpu_group (to avoid NCCL collectives
               that interfere with CUDA graph capture).
            """

            def __init__(self, device_group, cpu_group):
                super().__init__(group=device_group)
                self._cpu_group = cpu_group

            def bcast(self, data, root):
                import torch.distributed as dist

                group_ranks = dist.get_process_group_ranks(self._cpu_group)
                global_root = group_ranks[root]
                object_list = [data]
                dist.broadcast_object_list(
                    object_list, src=global_root, group=self._cpu_group
                )
                return object_list[0]

        _TorchDistBackend = _FixedTorchDistBackend
    except ImportError:
        logger.debug(
            "flashinfer.comm.mnnvl.TorchDistBackend is not available, "
            "allreduce fusion will use the default process group"
        )

    try:
        from flashinfer.comm.mnnvl import CommBackend

        class TorchDistributedCommBackend(CommBackend):
            """
            Use torch distributed instead of MPI to set up flashinfer MNNVL
            workspaces during initialization.
            """

            def __init__(self, group: ProcessGroup):
                self._group = group

            def Get_rank(self) -> int:
                return self._group.rank()

            def Get_size(self) -> int:
                return self._group.size()

            def allgather(self, data: int):
                gathered = [None] * self.Get_size()
                dist.all_gather_object(gathered, data, group=self._group)
                return gathered

            def bcast(self, data, root: int = 0):
                """Broadcast a picklable Python object from root to all ranks."""
                obj_list = [data]
                dist.broadcast_object_list(obj_list, src=root, group=self._group)
                return obj_list[0]

            def barrier(self):
                dist.barrier(group=self._group)

            def Split(self, color: int, key: int):
                # No need to split; we already use the proper group.
                return self._group

        _mnnvl_comm_backend = TorchDistributedCommBackend
    except ImportError:
        _mnnvl_comm_backend = None


# FlashInfer allreduce fusion backend support matrix for
# --flashinfer-allreduce-fusion-backend:
#
#   Backend   | SM100 | SM90 | Single-Node | Multi-Node |
#   --------- | ----- | ---- | ----------- | ---------- |
#   trtllm    | Yes   | Yes  | Yes         | No         |
#   mnnvl     | Yes   | No   | Yes         | Yes        |
#
# auto/mnnvl: only resolves to mnnvl on Blackwell GPUs (SM10x) where it has
# been validated. On every other platform the selection is forced to trtllm
# (see _resolve_backend).


def is_flashinfer_allreduce_unavailable() -> bool:
    return _flashinfer_allreduce_unavailable


def _make_flashinfer_workspace_allocation_prop(cuda_driver):
    if _should_force_posix_fd_transport():
        handle_type = (
            cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    else:
        from flashinfer.comm.mnnvl import is_mnnvl_fabric_supported

        if is_mnnvl_fabric_supported(torch.cuda.current_device()):
            handle_type = (
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            handle_type = (
                cuda_driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )

    prop = cuda_driver.CUmemAllocationProp()
    prop.requestedHandleTypes = handle_type
    prop.type = cuda_driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = cuda_driver.CUmemLocation()
    prop.location.type = cuda_driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = torch.cuda.current_device()
    prop.allocFlags.gpuDirectRDMACapable = 1
    return prop


def _flashinfer_trtllm_workspace_allocation_sizes(
    cuda_driver,
    prop,
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[int]:
    """Mirror FlashInfer TRTLLM SymmDeviceMemory local allocation sizes."""
    elem_size = 4 if dtype == torch.float32 else 2
    buffer_size = world_size * max_token_num * hidden_dim * 2
    flag_size = world_size * 256 * 4

    max_comm_size = 2147483647 & ~((1 << 21) - 1)
    lamport_comm_size = min(
        world_size * max_token_num * hidden_dim * elem_size,
        max_comm_size,
    )
    lamport_buffer_size = lamport_comm_size * 3

    # trtllm_create_ipc_workspace_for_all_reduce_fusion rounds each logical
    # buffer to 2 MiB before passing it to SymmDeviceMemory.
    buffer_sizes = (
        ceil_align(size, 1 << 21)
        for size in (buffer_size, flag_size, lamport_buffer_size)
    )

    signal_pad_size = 2048
    allocation_sizes = []
    for buffer_size in buffer_sizes:
        err, alloc_granularity = cuda_driver.cuMemGetAllocationGranularity(
            prop,
            cuda_driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMemGetAllocationGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(buffer_size + signal_pad_size, alloc_granularity)

        mc_prop = cuda_driver.CUmulticastObjectProp()
        mc_prop.numDevices = world_size
        mc_prop.size = allocation_size
        mc_prop.handleTypes = prop.requestedHandleTypes

        err, mc_granularity = cuda_driver.cuMulticastGetGranularity(
            mc_prop,
            cuda_driver.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
        )
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(
                "cuMulticastGetGranularity failed for FlashInfer "
                f"workspace preflight: {err}"
            )

        allocation_size = ceil_align(allocation_size, mc_granularity)
        allocation_sizes.append(allocation_size)
    return allocation_sizes


def _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop) -> bool:
    handles = []
    try:
        for allocation_size in allocation_sizes:
            err, handle = cuda_driver.cuMemCreate(allocation_size, prop, 0)
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                return False
            handles.append(handle)
        return True
    finally:
        for handle in reversed(handles):
            cuda_driver.cuMemRelease(handle)


def _preflight_check_workspace_memory(
    world_size: int,
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> bool:
    """Collectively decide whether to enter FlashInfer workspace creation.

    FlashInfer TRTLLM workspaces allocate several SymmDeviceMemory buffers and
    then exchange handles across ranks. If one rank fails local cuMemCreate and
    exits while peers enter handle exchange, peers can hang until the watchdog
    aborts. Probe the same handle type and allocation sequence first, then vote
    on a CPU group so all ranks proceed or skip together.
    """
    import torch.distributed as dist

    group = cpu_group
    if group is None:
        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return True
        group = tp_group.cpu_group

    allocation_sizes = []
    try:
        cuda_driver = get_cuda_driver_bindings()
        prop = _make_flashinfer_workspace_allocation_prop(cuda_driver)
        allocation_sizes = _flashinfer_trtllm_workspace_allocation_sizes(
            cuda_driver,
            prop,
            world_size,
            max_token_num,
            hidden_dim,
            dtype,
        )
        local_ok = _probe_cumem_create_sequence(cuda_driver, allocation_sizes, prop)
    except Exception as e:
        logger.warning(
            "FlashInfer workspace preflight probe failed (%s). "
            "Skipping allreduce fusion.",
            e,
        )
        local_ok = False

    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)

    logger.debug(
        "FlashInfer workspace preflight [rank %s]: probe=%.2f GB, "
        "local_probe=%s, vote=%s",
        dist.get_rank(group=group),
        sum(allocation_sizes) / 1e9,
        "OK" if local_ok else "FAIL",
        "PROCEED" if flag.item() == 1 else "SKIP",
    )
    if flag.item() == 0:
        logger.warning(
            "FlashInfer workspace preflight: cuMemCreate probe failed on at "
            "least one rank. Skipping allreduce fusion to avoid cross-rank "
            "desync inside the flashinfer collective."
        )
        return False
    return True


class FlashInferWorkspaceManager:
    """
    Manages FlashInfer's unified allreduce workspace.
    Supports trtllm and mnnvl backends via create_allreduce_fusion_workspace().
    """

    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.group = None
        self.max_token_num = None
        self.hidden_dim = None
        self.dtype = None
        self.initialized = False
        # Track max sizes ever requested so the workspace only grows (fewer recreates)
        self._max_token_num_seen: Optional[int] = None
        self._max_hidden_dim_seen: Optional[int] = None
        self._logged_init = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        backend: str = "auto",
        group: Optional[ProcessGroup] = None,
        use_fp32_lamport: bool = False,
        dtype: Optional[torch.dtype] = None,
        use_oneshot: Optional[bool] = None,
        device_group: Optional["torch.distributed.ProcessGroup"] = None,
        cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
    ):
        """Initialize workspace using FlashInfer's unified API."""
        global _flashinfer_allreduce_unavailable

        # Track the high-water mark so allocations only grow
        self._max_token_num_seen = max(max_token_num, self._max_token_num_seen or 0)
        self._max_hidden_dim_seen = max(hidden_dim, self._max_hidden_dim_seen or 0)

        # Reuse existing workspace if it already covers this problem size
        if (
            self.initialized
            and self.world_size == world_size
            and self.is_buffer_size_sufficient(
                token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype or torch.bfloat16,
                use_oneshot=use_oneshot,
            )
        ):
            return

        # Same world_size but buffer too small: free old workspace before creating new
        if self.initialized and self.world_size == world_size:
            self.cleanup()

        if _flashinfer_comm is None or _create_allreduce_fusion_workspace is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )
            return

        self.cleanup()

        if not _preflight_check_workspace_memory(
            world_size=world_size,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            cpu_group=cpu_group,
        ):
            _flashinfer_allreduce_unavailable = True
            self.workspace = None
            self.initialized = False
            return

        # Determine GPUs per node for MNNVL topology detection
        gpus_per_node = None
        node_pg = cpu_group if cpu_group is not None else group
        if node_pg is not None:
            gpus_per_node = sum(in_the_same_node_as(node_pg, source_rank=0))

        comm_backend = None
        if (
            _TorchDistBackend is not None
            and device_group is not None
            and cpu_group is not None
        ):
            comm_backend = _TorchDistBackend(
                device_group=device_group, cpu_group=cpu_group
            )
        elif _mnnvl_comm_backend is not None and group is not None:
            comm_backend = _mnnvl_comm_backend(group)

        try:
            alloc_token_num = max(max_token_num, self._max_token_num_seen or 0)
            alloc_hidden_dim = max(hidden_dim, self._max_hidden_dim_seen or 0)
            create_kw = dict(
                backend=backend,
                world_size=world_size,
                rank=rank,
                max_token_num=alloc_token_num,
                hidden_dim=alloc_hidden_dim,
                dtype=dtype or torch.bfloat16,
                gpus_per_node=gpus_per_node,
                comm_backend=comm_backend,
                # Pin the symmetric-memory rendezvous to the actual
                # subgroup. Without this, flashinfer >=0.6.10 falls back
                # to WORLD and TP/EP/CP subgroup peers get addressed
                # incorrectly (kernel hangs in cuda-graph warmup).
                group=device_group,
            )
            if use_oneshot is not None:
                create_kw["force_oneshot_support"] = bool(use_oneshot)
            if use_fp32_lamport:
                create_kw["use_fp32_lamport"] = True
            with _flashinfer_posix_fd_transport_override_if_needed():
                self.workspace = _create_allreduce_fusion_workspace(**create_kw)
            self.world_size = world_size
            self.rank = rank
            self.group = (device_group, cpu_group)
            self.max_token_num = alloc_token_num
            self.hidden_dim = alloc_hidden_dim
            self.dtype = dtype or torch.bfloat16
            self.initialized = True

            backend_name = getattr(self.workspace, "backend", "unknown")
            if not self._logged_init:
                logger.info(
                    f"FlashInfer AllReduce Fusion enabled: backend={backend_name}, "
                    f"rank={rank}, world_size={world_size}"
                )
                self._logged_init = True
            else:
                logger.debug(
                    f"FlashInfer workspace re-initialized: backend={backend_name}, "
                    f"rank={rank}, world_size={world_size}"
                )
        except Exception as e:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                f"Failed to initialize FlashInfer workspace (backend={backend}): {e}. "
                "Disabling flashinfer allreduce fusion permanently."
            )
            self.workspace = None
            self.initialized = False

    def is_buffer_size_sufficient(
        self,
        token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[bool] = None,
    ) -> bool:
        if not self.initialized or self.workspace is None:
            return False
        try:
            return self.workspace.is_buffer_size_sufficient(
                tp_size=self.world_size,
                num_tokens=token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                use_oneshot=use_oneshot,
            )
        except Exception as e:
            logger.debug(f"FlashInfer workspace size check failed: {e}")
            # Fallback: some backends may not implement is_buffer_size_sufficient;
            # reuse if within our allocated dimensions.
            if (
                self.max_token_num is not None
                and self.hidden_dim is not None
                and token_num <= self.max_token_num
                and hidden_dim <= self.hidden_dim
            ):
                return True
            return False

    def cleanup(self):
        """Clean up workspace."""
        if self.workspace is not None:
            try:
                if hasattr(self.workspace, "destroy"):
                    self.workspace.destroy()
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace = None
                self.initialized = False
                self.world_size = None
                self.rank = None
                self.group = None
                self.max_token_num = None
                self.hidden_dim = None
                self.dtype = None
                self._logged_init = False


_attn_tp_workspace_manager = FlashInferWorkspaceManager()
_moe_tp_workspace_manager = FlashInferWorkspaceManager()


def _get_workspace_manager(use_attn_tp_group: bool) -> FlashInferWorkspaceManager:
    return (
        _attn_tp_workspace_manager if use_attn_tp_group else _moe_tp_workspace_manager
    )


def _sync_allreduce_unavailable_across_tp():
    """Synchronize _flashinfer_allreduce_unavailable across all TP ranks.

    If workspace initialization fails on any rank, all ranks must agree to
    disable fusion. Otherwise ranks diverge during CUDA graph capture: some
    use FlashInfer fusion (skipping custom allreduce), others fall back to
    standard allreduce (calling register_buffer collectives), causing a hang
    in register_graph_buffers.
    """
    global _flashinfer_allreduce_unavailable
    try:
        import torch.distributed as dist

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            return
        flag = torch.tensor(
            [1 if _flashinfer_allreduce_unavailable else 0],
            dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=tp_group.cpu_group)
        if flag.item() > 0 and not _flashinfer_allreduce_unavailable:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "FlashInfer allreduce fusion disabled globally because "
                "workspace initialization failed on at least one rank."
            )
    except Exception as e:
        logger.debug(f"Failed to sync flashinfer unavailable flag: {e}")


def ensure_workspace_initialized(
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
    dtype: Optional[torch.dtype] = None,
    token_num: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    use_attn_tp_group: bool = True,
):
    """Ensure workspace is initialized."""
    if _flashinfer_allreduce_unavailable:
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    tp_coordinator = get_tp_group()

    if use_attn_tp_group:
        world_size = get_attn_tensor_model_parallel_world_size()
        rank = get_attn_tensor_model_parallel_rank()
        coordinator = get_attn_tp_group()
    else:
        if get_moe_expert_parallel_world_size() > 1:
            world_size = get_moe_expert_parallel_world_size()
            rank = get_moe_expert_parallel_rank()
            coordinator = get_moe_ep_group()
        else:
            world_size = get_moe_tensor_parallel_world_size()
            rank = get_moe_tensor_parallel_rank()
            coordinator = get_moe_tp_group()

    # When the sub-group IS the full TP group, pass None so the workspace
    # uses the default process group directly (no TorchDistBackend needed).
    # For true sub-groups, use NCCL device_group for GPU/device mapping and
    # GLOO cpu_group for metadata broadcasts (avoids NCCL collectives that
    # interfere with CUDA graph capture).
    if coordinator.device_group is tp_coordinator.device_group:
        device_group = None
        cpu_group = None
    else:
        device_group = coordinator.device_group
        cpu_group = coordinator.cpu_group

    if world_size <= 1:
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    token_num = token_num or max_token_num
    group_key = (device_group, cpu_group)
    effective_dtype = dtype or torch.bfloat16
    backend = _resolve_backend(
        get_global_server_args().flashinfer_allreduce_fusion_backend or "auto"
    )

    if (
        not workspace_manager.initialized
        or workspace_manager.world_size != world_size
        or workspace_manager.rank != rank
        or workspace_manager.group != group_key
        or not workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=effective_dtype,
            use_oneshot=use_oneshot,
        )
    ):
        workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            backend=backend,
            group=cpu_group,
            use_fp32_lamport=use_fp32_lamport,
            dtype=dtype,
            use_oneshot=use_oneshot,
            device_group=device_group,
            cpu_group=cpu_group,
        )

        _sync_allreduce_unavailable_across_tp()

    return workspace_manager.initialized


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 16384,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)
    return norm_out, residual_out


@register_custom_op(
    mutates_args=["input_tensor", "residual", "weight"],
    fake_impl=fake_flashinfer_allreduce_residual_rmsnorm,
)
def flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    use_attn_tp_group: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's unified fused allreduce + residual + RMS norm operation.
    Automatically selects between trtllm and mnnvl backends based on topology
    and hardware (controlled by --flashinfer-allreduce-fusion-backend).

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision
        use_attn_tp_group: If True, use attention TP group; otherwise use MoE TP group

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard implementation"
        )
        return None, None

    if use_attn_tp_group:
        world_size = get_attn_tensor_model_parallel_world_size()
    else:
        if get_moe_expert_parallel_world_size() > 1:
            world_size = get_moe_expert_parallel_world_size()
        else:
            world_size = get_moe_tensor_parallel_world_size()

    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num
    if (
        not input_tensor.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        logger.debug("Non-contiguous tensors, skipping FlashInfer allreduce fusion")
        return None, None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
        dtype=input_tensor.dtype,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
        use_attn_tp_group=use_attn_tp_group,
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    if workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None, None

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    _flashinfer_comm.allreduce_fusion(
        input=input_tensor,
        workspace=workspace_manager.workspace,
        pattern=_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        trigger_completion_at_end=trigger_completion_at_end,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=weight,
        rms_eps=eps,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
    )

    return norm_out, residual_out


def pre_initialize_workspaces(
    max_token_num: int,
    hidden_dim: int,
    dtype: torch.dtype,
    use_oneshot: Optional[bool] = None,
):
    """Pre-initialize flashinfer workspaces before CUDA graph capture.

    This must be called before graph capture to avoid collective operations
    (broadcasts, barriers) inside the graph capture context, which can
    deadlock with custom_all_reduce.register_graph_buffers.
    """
    if _flashinfer_allreduce_unavailable or _flashinfer_comm is None:
        return

    # Initialize MoE workspace
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=False,
    )

    # Initialize attention workspace
    ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        dtype=dtype,
        use_oneshot=use_oneshot,
        use_attn_tp_group=True,
    )


def cleanup_flashinfer_workspace():
    global _attn_tp_workspace_manager, _moe_tp_workspace_manager
    if _attn_tp_workspace_manager is not None:
        _attn_tp_workspace_manager.cleanup()
    if (
        _moe_tp_workspace_manager is not None
        and _moe_tp_workspace_manager is not _attn_tp_workspace_manager
    ):
        _moe_tp_workspace_manager.cleanup()

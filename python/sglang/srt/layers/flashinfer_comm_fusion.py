import contextlib
import logging
import platform
from typing import Optional, Tuple

import torch

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
from sglang.srt.environ import envs
from sglang.srt.utils import (
    ceil_align,
    get_cuda_driver_bindings,
    is_flashinfer_available,
)
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_TorchDistBackend = None
_flashinfer_allreduce_unavailable = False
_posix_transport_override_logged = False


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

        if hasattr(comm, "allreduce_fusion") and hasattr(
            comm, "create_allreduce_fusion_workspace"
        ):
            _flashinfer_comm = comm
        else:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                "flashinfer.comm unified allreduce_fusion API is not available, "
                "falling back to standard implementation"
            )
    except ImportError:
        _flashinfer_allreduce_unavailable = True
        logger.warning(
            "flashinfer.comm is not available, falling back to standard "
            "implementation"
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
    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.group = None
        self.max_token_num = None
        self.hidden_dim = None
        self.dtype = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[bool] = None,
        device_group: Optional["torch.distributed.ProcessGroup"] = None,
        cpu_group: Optional["torch.distributed.ProcessGroup"] = None,
    ):
        """Initialize workspace"""
        if _flashinfer_comm is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )
            return

        self.cleanup()

        global _flashinfer_allreduce_unavailable
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

        try:
            kwargs = dict(
                backend="trtllm",
                world_size=world_size,
                rank=rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype,
                force_oneshot_support=bool(use_oneshot),
                # Pin the symmetric-memory rendezvous to the actual
                # subgroup. Without this, flashinfer >=0.6.10 falls back
                # to WORLD and TP/EP/CP subgroup peers get addressed
                # incorrectly (kernel hangs in cuda-graph warmup).
                group=device_group,
            )
            if (
                _TorchDistBackend is not None
                and device_group is not None
                and cpu_group is not None
            ):
                kwargs["comm_backend"] = _TorchDistBackend(
                    device_group=device_group, cpu_group=cpu_group
                )
            with _flashinfer_posix_fd_transport_override_if_needed():
                self.workspace = _flashinfer_comm.create_allreduce_fusion_workspace(
                    **kwargs
                )
        except Exception as e:
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                f"Failed to initialize FlashInfer workspace: {e}. "
                "Disabling flashinfer allreduce fusion permanently."
            )
            self.workspace = None
            self.initialized = False
            return

        self.world_size = world_size
        self.rank = rank
        self.group = (device_group, cpu_group)
        self.max_token_num = max_token_num
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.initialized = True

        backend = getattr(self.workspace, "backend", "unknown")
        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, "
            f"world_size {world_size}, backend {backend}"
        )

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
            return False

    def cleanup(self):
        """Clean up workspace"""
        if self.workspace is not None:
            try:
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
    dtype: torch.dtype = torch.float16,
    token_num: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
    use_attn_tp_group: bool = True,
):
    """Ensure workspace is initialized"""
    if _flashinfer_allreduce_unavailable:
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

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

    # Always pass the coordinator's groups: flashinfer >=0.6.10 reads the
    # rendezvous group from `group=...` (falling back to WORLD when None),
    # so leaving it None silently rendezvouses on WORLD and the kernel ends
    # up addressing the wrong peers in TP/EP/CP subgroup setups.
    device_group = coordinator.device_group
    cpu_group = coordinator.cpu_group

    if world_size <= 1:
        return False

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
    token_num = token_num or max_token_num
    group_key = (device_group, cpu_group)

    if (
        not workspace_manager.initialized
        or workspace_manager.world_size != world_size
        or workspace_manager.rank != rank
        or workspace_manager.group != group_key
        or not workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            use_oneshot=use_oneshot,
        )
    ):
        workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
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
    Use FlashInfer's fused allreduce + residual + RMS norm operation

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
        # If MoE expert parallel world size > 1, use expert parallel group
        # Otherwise, use tensor parallel group
        # The two values cannot be larger than 1 at the same time
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
        dtype=input_tensor.dtype,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
        use_attn_tp_group=use_attn_tp_group,
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    workspace_manager = _get_workspace_manager(use_attn_tp_group)
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

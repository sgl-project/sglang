import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

# FlashInfer allreduce fusion: set when flashinfer is available (see block below)
_flashinfer_comm = None
_workspace_manager = None
_mnnvl_comm_backend = None
_AllReduceFusionPattern = None
_create_allreduce_fusion_workspace = None
_allreduce_fusion = None
_flashinfer_allreduce_unavailable = False

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
        # Try to import the unified allreduce API
        from flashinfer.comm.allreduce import (
            allreduce_fusion,
            create_allreduce_fusion_workspace,
        )

        # AllReduceFusionPattern might be in trtllm_ar or allreduce module
        try:
            from flashinfer.comm.allreduce import AllReduceFusionPattern
        except ImportError:
            from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        _AllReduceFusionPattern = AllReduceFusionPattern
        _create_allreduce_fusion_workspace = create_allreduce_fusion_workspace
        _allreduce_fusion = allreduce_fusion
    except ImportError:
        # Fall back to legacy API if unified API is not available
        _AllReduceFusionPattern = None
        _create_allreduce_fusion_workspace = None
        _allreduce_fusion = None
        logger.warning(
            "FlashInfer unified allreduce API not available, using legacy API"
        )

    try:
        from flashinfer.comm.mnnvl import CommBackend

        class TorchDistributedCommBackend(CommBackend):
            """
            Use torch distributed instead of MPI to set up flashinfer MNNVL workspaces during initialization
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
                """
                Broadcast a picklable Python object from `root` to all ranks.
                Uses torch.distributed.broadcast_object_list under the hood.

                Returns the broadcasted object on every rank.
                """
                obj_list = [data]
                # broadcast_object_list mutates obj_list in-place
                dist.broadcast_object_list(obj_list, src=root, group=self._group)
                return obj_list[0]

            def barrier(self):
                """
                Synchronize all ranks in this communicator.
                """
                dist.barrier(group=self._group)

            def Split(self, color: int, key: int):
                # No need to split, we already use the proper group
                return self._group

        _mnnvl_comm_backend = TorchDistributedCommBackend
    except ImportError:
        _mnnvl_comm_backend = None


# FlashInfer allreduce fusion (fused allreduce + Residual + RMSNorm) backend support
# for --flashinfer-allreduce-fusion-backend:
#
#   Feature / Framework   | SM100 | SM90 | Single Node | Multi-Node |
#   --------------------- | ----- | ---- | ----------- | ---------- |
#   TRT-LLM AllReduce     | Yes   | Yes  | Yes         | No         |
#   MNNVL AllReduce       | Yes   | No   | Yes         | Yes        |
#
# With backend "auto": trtllm is used on single-node, mnnvl on single or multi-node (SM100 only).
# Multi-node + Hopper is unsupported (trtllm would be chosen but does not support multi-node).


def is_flashinfer_allreduce_unavailable() -> bool:
    return _flashinfer_allreduce_unavailable


class FlashInferWorkspaceManager:
    """
    Workspace manager using FlashInfer's unified allreduce API.
    Wraps FlashInfer's create_allreduce_fusion_workspace() for automatic backend selection.
    """

    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.max_token_num = None
        self.hidden_dim = None
        self.dtype = None
        self.initialized = False
        # Max size ever requested (not cleared on cleanup) so we only grow and minimize recreates
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
    ):
        """Initialize workspace using FlashInfer's unified API"""
        # Track max size ever requested so we can create with at least that (only grow, minimize recreates)
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
        # Same world_size but buffer too small: free old workspace before creating new one
        if self.initialized and self.world_size == world_size:
            self.cleanup()

        if _flashinfer_comm is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace initialization"
            )
            return

        # Determine GPUs per node for MNNVL backend
        # FlashInfer will use this to determine topology internally
        gpus_per_node = None
        if group is not None:
            gpus_per_node = sum(in_the_same_node_as(group, source_rank=0))

        # Create comm backend for MNNVL if needed
        comm_backend = None
        if _mnnvl_comm_backend is not None and group is not None:
            comm_backend = _mnnvl_comm_backend(group)

        try:
            # Create with at least the max size we've ever been asked for (only grow, fewer recreates)
            alloc_token_num = max(max_token_num, self._max_token_num_seen or 0)
            alloc_hidden_dim = max(hidden_dim, self._max_hidden_dim_seen or 0)
            # Use FlashInfer's unified API to create workspace
            create_kw = dict(
                backend=backend,
                world_size=world_size,
                rank=rank,
                max_token_num=alloc_token_num,
                hidden_dim=alloc_hidden_dim,
                dtype=dtype or torch.bfloat16,
                gpus_per_node=gpus_per_node,
                comm_backend=comm_backend,
            )
            if use_oneshot is not None:
                create_kw["force_oneshot_support"] = bool(use_oneshot)
            self.workspace = _create_allreduce_fusion_workspace(**create_kw)
            self.world_size = world_size
            self.rank = rank
            self.max_token_num = alloc_token_num
            self.hidden_dim = alloc_hidden_dim
            self.dtype = dtype or torch.bfloat16
            self.initialized = True

            backend_name = getattr(self.workspace, "backend", "unknown")
            if not self._logged_init:
                logger.info(
                    f"FlashInfer workspace initialized for rank {rank}, "
                    f"world_size {world_size}, backend: {backend_name}"
                )
                self._logged_init = True
            else:
                logger.debug(
                    f"FlashInfer workspace re-initialized for rank {rank}, "
                    f"world_size {world_size}, backend: {backend_name}"
                )
        except Exception as e:
            global _flashinfer_allreduce_unavailable
            _flashinfer_allreduce_unavailable = True
            logger.warning(
                f"Failed to initialize FlashInfer workspace: {e}. "
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
            # Fallback: some backends (e.g. MNNVL) may use a different API; reuse if within our allocated size
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
                self.max_token_num = None
                self.hidden_dim = None
                self.dtype = None
                self._logged_init = False


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
    dtype: Optional[torch.dtype] = None,
    group: Optional[ProcessGroup] = None,
    token_num: Optional[int] = None,
    use_oneshot: Optional[bool] = None,
):
    """Ensure workspace is initialized"""
    if _flashinfer_allreduce_unavailable:
        return False

    if not is_flashinfer_available() or _flashinfer_comm is None:
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = get_tensor_model_parallel_rank()
    token_num = token_num or max_token_num
    effective_dtype = dtype or torch.bfloat16

    if (
        not _workspace_manager.initialized
        or _workspace_manager.world_size != world_size
        or _workspace_manager.rank != rank
        or not _workspace_manager.is_buffer_size_sufficient(
            token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=effective_dtype,
            use_oneshot=use_oneshot,
        )
    ):
        backend = get_global_server_args().flashinfer_allreduce_fusion_backend or "auto"
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            backend=backend,
            use_fp32_lamport=use_fp32_lamport,
            dtype=dtype,
            group=group,
            use_oneshot=use_oneshot,
        )

    return _workspace_manager.initialized


def fake_flashinfer_allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 16384,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use FlashInfer's unified fused allreduce + residual + RMS norm operation.
    Automatically selects between IPC and MNNVL backends based on topology and hardware.

    Args:
        input_tensor: Input tensor that needs allreduce
        residual: Residual tensor
        weight: RMS norm weight
        eps: RMS norm epsilon
        max_token_num: Maximum token number
        use_oneshot: Whether to use oneshot mode
        trigger_completion_at_end: Whether to trigger completion at end
        fp32_acc: Whether to use fp32 precision

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available():
        logger.debug(
            "FlashInfer not available, falling back to standard implementation"
        )
        return None, None

    world_size = get_tensor_model_parallel_world_size()
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

    # Get TP group for workspace initialization
    try:
        group = get_tp_group().cpu_group
    except Exception:
        group = None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
        dtype=input_tensor.dtype,
        group=group,
        token_num=input_tensor.shape[0],
        use_oneshot=use_oneshot,
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    if _workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None, None

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    try:
        if _AllReduceFusionPattern is None or _allreduce_fusion is None:
            return None, None

        _allreduce_fusion(
            input=input_tensor,
            workspace=_workspace_manager.workspace,
            pattern=_AllReduceFusionPattern.kARResidualRMSNorm,
            launch_with_pdl=trigger_completion_at_end,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=weight,
            rms_eps=eps,
        )
    except Exception as e:
        logger.warning(f"FlashInfer allreduce fusion failed: {e}")
        return None, None

    return norm_out, residual_out


def cleanup_flashinfer_workspace():
    """Clean up FlashInfer workspace"""
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()

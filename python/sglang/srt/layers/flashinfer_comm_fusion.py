import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tp_group
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_unified_allreduce = None
_workspace_manager = None
_mnnvl_comm_backend = None

if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        _flashinfer_comm = comm
    except ImportError:
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

        _unified_allreduce = True
        _AllReduceFusionPattern = AllReduceFusionPattern
        _create_allreduce_fusion_workspace = create_allreduce_fusion_workspace
        _allreduce_fusion = allreduce_fusion
    except ImportError:
        # Fall back to legacy API if unified API is not available
        _unified_allreduce = False
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


class FlashInferWorkspaceManager:
    """
    Workspace manager using FlashInfer's unified allreduce API.
    Wraps FlashInfer's create_allreduce_fusion_workspace() for automatic backend selection.
    """

    def __init__(self):
        self.workspace = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group: Optional[ProcessGroup] = None,
        use_fp32_lamport: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize workspace using FlashInfer's unified API"""
        if self.initialized and self.world_size == world_size:
            # Check if workspace is sufficient for current problem size
            if self.workspace is not None:
                try:
                    if hasattr(self.workspace, "is_buffer_size_sufficient"):
                        if self.workspace.is_buffer_size_sufficient(
                            world_size,
                            max_token_num,
                            hidden_dim,
                            dtype or torch.bfloat16,
                        ):
                            return
                except Exception:
                    pass
            # Workspace needs to be recreated
            self.cleanup()

        if not _unified_allreduce:
            logger.warning(
                "FlashInfer unified API not available, skipping workspace initialization"
            )
            return

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

        # Determine backend from environment variable or use auto
        backend_choice = envs.SGLANG_FLASHINFER_ALLREDUCE_FUSION_BACKEND.get()
        if not backend_choice or backend_choice == "":
            backend_choice = "auto"
        elif backend_choice not in ["auto", "trtllm", "mnnvl"]:
            logger.warning(
                f"Invalid backend choice '{backend_choice}' for SGLANG_FLASHINFER_ALLREDUCE_FUSION_BACKEND. "
                f"Valid options are: 'auto', 'trtllm', 'mnnvl'. Using 'auto'."
            )
            backend_choice = "auto"

        try:
            # Use FlashInfer's unified API to create workspace
            # Backend can be forced via SGLANG_FLASHINFER_ALLREDUCE_FUSION_BACKEND env var
            self.workspace = _create_allreduce_fusion_workspace(
                backend=backend_choice,
                world_size=world_size,
                rank=rank,
                max_token_num=max_token_num,
                hidden_dim=hidden_dim,
                dtype=dtype or torch.bfloat16,
                gpus_per_node=gpus_per_node,
                comm_backend=comm_backend,
            )

            self.world_size = world_size
            self.rank = rank
            self.initialized = True

            logger.info(
                f"FlashInfer workspace initialized for rank {rank}, "
                f"world_size {world_size}, backend: {self.workspace.backend}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize FlashInfer workspace: {e}")
            self.workspace = None
            self.initialized = False

    def cleanup(self):
        """Clean up workspace"""
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


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    max_token_num: int = 2048,
    hidden_dim: int = 4096,
    use_fp32_lamport: bool = False,
    dtype: Optional[torch.dtype] = None,
    group: Optional[ProcessGroup] = None,
):
    """Ensure workspace is initialized using FlashInfer's unified API"""
    if not is_flashinfer_available() or not _unified_allreduce:
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _workspace_manager.initialized
        or _workspace_manager.world_size != world_size
    ):
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
            dtype=dtype,
            group=group,
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
    if not is_flashinfer_available() or not _unified_allreduce:
        logger.debug(
            "FlashInfer unified API not available, falling back to standard "
            "implementation"
        )
        return None, None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num

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
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    if _workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None, None

    # Pre-allocate output tensors
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    # Use FlashInfer's unified allreduce_fusion API
    # It automatically dispatches to the correct backend based on workspace type
    try:
        if _AllReduceFusionPattern is None or _allreduce_fusion is None:
            # Fallback to legacy API
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


def flashinfer_allreduce(
    input: torch.Tensor,
    group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """
    Standalone allreduce operation using FlashInfer's unified API.
    Automatically selects between IPC and MNNVL backends.

    Args:
        input: Input tensor to allreduce
        group: Process group (defaults to TP group)

    Returns:
        Output tensor after allreduce
    """
    if not is_flashinfer_available() or not _unified_allreduce:
        logger.debug("FlashInfer unified API not available")
        return None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return input

    if group is None:
        try:
            group = get_tp_group().cpu_group
        except Exception:
            return None

    # Ensure workspace is initialized
    if not ensure_workspace_initialized(
        max_token_num=input.shape[0] if len(input.shape) > 1 else 1,
        hidden_dim=input.shape[-1],
        dtype=input.dtype,
        group=group,
    ):
        logger.debug("FlashInfer workspace not available")
        return None

    if _workspace_manager.workspace is None:
        logger.debug("FlashInfer workspace is None")
        return None

    # Use FlashInfer's unified allreduce_fusion API for standalone allreduce
    output = torch.empty_like(input)
    try:
        if _AllReduceFusionPattern is None or _allreduce_fusion is None:
            return None

        _allreduce_fusion(
            input=input,
            workspace=_workspace_manager.workspace,
            pattern=_AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            output=output,
        )
    except Exception as e:
        logger.warning(f"FlashInfer allreduce failed: {e}")
        return None

    return output


def cleanup_flashinfer_workspace():
    """Clean up FlashInfer workspace"""
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()

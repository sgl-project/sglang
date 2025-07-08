from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tensor_model_parallel_world_size, parallel_state
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_flashinfer_mnnvlmoe = None
_workspace_manager = None
_alltoall_workspace_manager = None

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
        from flashinfer.comm.trtllm_alltoall import MnnvlMoe as MnnvlMoe

        _flashinfer_mnnvlmoe = MnnvlMoe
    except ImportError:
        logger.warning("flashinfer.comm.trtllm_alltoall.MnnvlMoe is not available")


class FlashInferWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.ipc_handles = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group=None,
        use_fp32_lamport: bool = False,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        if _flashinfer_comm is None:
            logger.warning(
                "FlashInfer comm not available, skipping workspace " "initialization"
            )
            return

        self.cleanup()

        self.ipc_handles, self.workspace_tensor = (
            comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                rank,
                world_size,
                max_token_num,
                hidden_dim,
                group=group,
                use_fp32_lamport=use_fp32_lamport,
            )
        )

        self.world_size = world_size
        self.rank = rank
        self.initialized = True

        logger.info(
            f"FlashInfer workspace initialized for rank {rank}, "
            f"world_size {world_size}"
        )

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.ipc_handles is not None:
            try:
                _flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                    self.ipc_handles, group=dist.group.WORLD
                )
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.ipc_handles = None
                self.initialized = False


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    max_token_num: int = 1024, hidden_dim: int = 4096, use_fp32_lamport: bool = False
):
    """Ensure workspace is initialized"""
    if not is_flashinfer_available() or _flashinfer_comm is None:
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
        )

    return _workspace_manager.initialized


def flashinfer_allreduce_add_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 1024,
    use_oneshot: bool = True,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
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

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (norm_output, residual_output)
    """
    if not is_flashinfer_available() or _flashinfer_comm is None:
        logger.debug(
            "FlashInfer not available, falling back to standard " "implementation"
        )
        return None, None

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    if not ensure_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == torch.float32),
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    token_num, hidden_dim = input_tensor.shape

    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(input_tensor)

    _flashinfer_comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=dist.get_rank(),
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=True,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(_flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=None,
        scale_out=None,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
    )

    return norm_out, residual_out


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()


class FlashInferAllToAllWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        gpus_per_node: int = 8,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        if _flashinfer_comm is None or _flashinfer_mnnvlmoe is None:
            logger.warning(
                "FlashInfer AlltoAll not available, skipping AllToAll workspace "
                "initialization"
            )
            return

        self.cleanup()

        self.mapping = _flashinfer_comm.mapping.Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=gpus_per_node,
            tp_size=world_size,
        )
        self.workspace_tensor = _flashinfer_mnnvlmoe.get_moe_workspaces(self.mapping)

        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.initialized = True

        logger.info(
            f"FlashInfer AllToAll workspace initialized for rank {rank}, "
            f"world_size {world_size}"
        )

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.workspace_tensor is not None:
            try:
                del self.workspace_tensor
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.mapping = None
                self.initialized = False


_alltoall_workspace_manager = FlashInferAllToAllWorkspaceManager()


def ensure_alltoall_workspace_initialized():
    """Ensure workspace is initialized"""
    if (
        not is_flashinfer_available()
        or _flashinfer_comm is None
        or _flashinfer_mnnvlmoe is None
    ):
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _alltoall_workspace_manager.initialized
        or _alltoall_workspace_manager.world_size != world_size
    ):
        _alltoall_workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            # TODO(tmorris): Get num gpus per node
        )

    return _alltoall_workspace_manager.initialized


def flashinfer_alltoall_dispatch(
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoEAlltoallInfo]:
    assert (
        ensure_alltoall_workspace_initialized()
    ), "FlashInfer AllToAll workspace not available"

    # gather router info
    # Assume same number of tokens across all devices if global_num_tokens_cpu is None
    max_num_token = max(global_num_tokens_cpu) if global_num_tokens_cpu is not None else x.shape[0]
    topk_ids = torch.nn.functional.pad(
        topk_ids, (0, 0, 0, max_num_token - topk_ids.shape[0]), "constant", num_experts
    )
    topk_weights = torch.nn.functional.pad(
        topk_weights, (0, 0, 0, max_num_token - topk_weights.shape[0])
    )
    gathered_topk_ids, gathered_topk_weights = (
        parallel_state.get_tp_group().all_gatherv([topk_ids, topk_weights])
    )
    gathered_topk_ids = torch.flatten(
        gathered_topk_ids.contiguous(), start_dim=0, end_dim=-2
    )
    gathered_topk_weights = torch.flatten(
        gathered_topk_weights.contiguous(), start_dim=0, end_dim=-2
    )
    gathered_target_rank_ids = _flashinfer_mnnvlmoe.compute_target_rank_id(
        gathered_topk_ids, num_experts, ep_size
    )
    alltoall_info, topk_ids, topk_weights = (
        _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv_prepare(
            gathered_target_rank_ids,
            None,
            gathered_topk_ids,
            gathered_topk_weights,
            max_num_token,
            num_experts,
            top_k,
            ep_rank,
            ep_size,
        )
    )

    x = _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv(
        x, alltoall_info, _alltoall_workspace_manager.workspace_tensor, ep_rank, ep_size
    )

    return x, topk_ids, topk_weights, alltoall_info


def flashinfer_alltoall_combine(
    output: torch.Tensor,
    alltoall_info: MoEAlltoallInfo,
    top_k: int,
    ep_rank: int,
    ep_size: int,
    token_count: int,
):
    assert (
        ensure_alltoall_workspace_initialized()
    ), "FlashInfer AllToAll workspace not available"
    return _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv_combine(
        output,
        alltoall_info,
        _alltoall_workspace_manager.workspace_tensor,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        token_count=token_count,
    )

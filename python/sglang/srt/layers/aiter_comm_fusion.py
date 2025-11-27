import logging
from typing import Tuple

import aiter
import torch
import torch.distributed as dist
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.utils import direct_register_custom_op

logger = logging.getLogger(__name__)


class AiterCommManager:
    def __init__(self):
        self.world_size = None
        self.rank = None
        self.dtype = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        dtype: torch.dtype,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        self.cleanup()

        self.world_size = world_size
        self.rank = rank
        self.dtype = dtype
        self.dist_env = aiter.TRTLLMDistEnv(rank, world_size, dtype=self.dtype)

        set_custom_all_reduce(True)
        init_distributed_environment(
            world_size=self.world_size,
            rank=rank,
            distributed_init_method=None,
        )
        ensure_model_parallel_initialized(self.world_size, 1)

        self.initialized = True

    def cleanup(self):
        self.dist_env = None
        self.initialized = False


_aiter_comm_manager = AiterCommManager()


def ensure_workspace_initialized(dtype):
    """Ensure workspace is initialized"""
    if _aiter_comm_manager is None:
        return False

    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if (
        not _aiter_comm_manager.initialized
        or _aiter_comm_manager.world_size != world_size
        or _aiter_comm_manager.dtype != dtype
    ):
        _aiter_comm_manager.initialize(
            world_size=world_size,
            rank=rank,
            dtype=dtype,
        )

    return _aiter_comm_manager.initialized


def aiter_allreduce_residual_rmsnorm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1e-6,
    fp8_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Use Aiter's fused AllReduce + Residual Add + RMSNorm + optional quantization operation
    """
    world_size = get_tensor_model_parallel_world_size()
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None, None
    if not ensure_workspace_initialized(allreduce_in.dtype):
        logger.debug("Aiter workspace is not initialized")
        return None, None, None
    num_tokens = allreduce_in.shape[0]
    if num_tokens <= 1024:
        residual_out, norm_out = tensor_model_parallel_fused_allreduce_rmsnorm(allreduce_in, residual_in, rms_weight, eps)
    else:
        return _aiter_comm_manager.dist_env.allreduce_add_rms_fused(
            allreduce_in,
            residual_in,
            rms_weight,
            eps,
            fp8_out,
        )
    return residual_out, norm_out, None


def fake_aiter_allreduce_residual_rmsnorm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1e-6,
    fp8_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(allreduce_in)
    scale_out = torch.empty(
        allreduce_in.shape[0],
        1,
        dtype=torch.float32,
        device=allreduce_in.device,
    )
    return residual_out, norm_out, scale_out


direct_register_custom_op(
    "aiter_allreduce_residual_rmsnorm",
    aiter_allreduce_residual_rmsnorm,
    mutates_args=["allreduce_in"],
    fake_impl=fake_aiter_allreduce_residual_rmsnorm,
)

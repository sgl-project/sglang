import logging
from typing import Tuple

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.utils import is_flashinfer_available

logger = logging.getLogger(__name__)

_flashinfer_comm = None
_workspace_manager = None

if is_flashinfer_available():
    try:
        import flashinfer.comm as comm

        _flashinfer_comm = comm
    except ImportError:
        logger.warning(
            "flashinfer.comm is not available, falling back to standard "
            "implementation"
        )


class FlashInferAllReduceWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
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

        self.workspace_tensor = comm.trtllm_create_ipc_workspace_for_all_reduce(
            rank,
            world_size,
            max_token_num,
            hidden_dim,
            group=group,
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
        if self.initialized and self.workspace_tensor is not None:
            try:
                _flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                    self.workspace_tensor, group=dist.group.WORLD
                )
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.initialized = False


_workspace_manager = FlashInferAllReduceWorkspaceManager()


def ensure_all_reduce_workspace_initialized(
    max_token_num: int = 128, hidden_dim: int = 4096
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
        )

    return _workspace_manager.initialized


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()


def flashinfer_allreduce(
    input_tensor: torch.Tensor,
    max_token_num: int = 128,
) -> torch.Tensor:
    """
    Use FlashInfer's custom all reduce operation

    Args:
        input_tensor: Input tensor that needs allreduce
        max_token_num: Maximum token number

    Returns:
        -> torch.Tensor:: (out)
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

    if not ensure_all_reduce_workspace_initialized(
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    token_num, hidden_dim = input_tensor.shape
    message_size = token_num * hidden_dim
    out = torch.empty_like(input_tensor)
    device = input_tensor.device
    _flashinfer_comm.trtllm_custom_all_reduce(
        inp=input_tensor,
        out=out,
        tp_size=world_size,
        tp_rank=dist.get_rank(),
        token_num=token_num,
        fusion_op_code=(_flashinfer_comm.AllReduceFusionOp.NONE),
        strategy_code=(_flashinfer_comm.AllReduceStrategyType.ONESHOT),
        config_code=(_flashinfer_comm.AllReduceStrategyConfig.USE_MEMCPY),
        launch_with_pdl=True,
        flag_value=1,
        peer_comm_buffer_ptrs=torch.tensor(
            _workspace_manager.workspace_tensor[0], dtype=torch.int64
        ),
        peer_barrier_ptrs_in=torch.tensor(
            _workspace_manager.workspace_tensor[2], dtype=torch.int64
        ),
        peer_barrier_ptrs_out=torch.tensor(
            _workspace_manager.workspace_tensor[3], dtype=torch.int64
        ),
        bias=torch.zeros(hidden_dim, dtype=input_tensor.dtype, device=device),
        residual=torch.zeros(hidden_dim, dtype=input_tensor.dtype, device=device),
        weight=torch.zeros(hidden_dim, dtype=input_tensor.dtype, device=device),
        weight_pre_residual_norm=torch.zeros(
            hidden_dim, dtype=input_tensor.dtype, device=device
        ),
        eps=1e-6,
        intermediate_buffer=torch.zeros(
            message_size, dtype=input_tensor.dtype, device=device
        ),
        lamport_peer_comm_buffer_ptrs_0=torch.tensor(
            _workspace_manager.workspace_tensor[4], dtype=torch.int64
        ),
        lamport_peer_comm_buffer_ptrs_1=torch.tensor(
            _workspace_manager.workspace_tensor[5], dtype=torch.int64
        ),
        lamport_peer_comm_buffer_ptrs_2=torch.tensor(
            _workspace_manager.workspace_tensor[6], dtype=torch.int64
        ),
    )
    return out

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from .parallel_state import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
    get_attn_cp_group,
    get_attn_tp_group,
    get_moe_ep_group,
    get_moe_tp_group,
    get_tp_group,
)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    return get_moe_ep_group().all_reduce(input_)


class AttnContextParallelCommunicate:
    def __init__(self):
        self._pending_operations: list[torch.distributed.P2POp] = []
        self._active_requests = None
        self.cp_group = get_attn_cp_group()
        self.cp_rank = get_attn_context_model_parallel_rank()
        self.cp_world_size = get_attn_context_model_parallel_world_size()
        self.send_rank = (self.cp_rank + 1) % self.cp_world_size
        self.recv_rank = (self.cp_rank - 1 + self.cp_world_size) % self.cp_world_size

    def send_recv_kvcache(self, tensor_to_send):
        result_tensor = torch.zeros_like(tensor_to_send)

        send_operation = torch.distributed.P2POp(
            torch.distributed.isend, tensor_to_send, self.send_rank, group=self.cp_group
        )
        recv_operation = torch.distributed.P2POp(
            torch.distributed.irecv, result_tensor, self.recv_rank, group=self.cp_group
        )

        self._pending_operations.extend([send_operation, recv_operation])

        print(
            f"RingComm | send_recv | RANK:{self.cp_rank} | "
            f"ACTION:sending | TO:{self.send_rank} | TENSOR:{tensor_to_send}",
            flush=True,
        )
        print(
            f"RingComm | send_recv | RANK:{self.cp_rank} | "
            f"ACTION:receiving | FROM:{self.recv_rank} | TENSOR:{result_tensor}",
            flush=True,
        )

        return result_tensor

    def commit(self):
        if self._active_requests is not None:
            raise RuntimeError("Commit called twice")
        self._active_requests = torch.distributed.batch_isend_irecv(
            self._pending_operations
        )
        print(
            f"RingComm | commit | RANK:{self.cp_rank} | "
            f"ACTION:committed | NUM_OPS:{len(self._pending_operations) // 2}",
            flush=True,
        )

    def wait(self):
        if self._active_requests is None:
            raise RuntimeError("Wait called before commit")
        for i, request in enumerate(self._active_requests):
            request.wait()
            operation_type = "send" if i % 2 == 0 else "receive"
            peer_rank = self.send_rank if operation_type == "send" else self.recv_rank
            print(
                f"RingComm | wait | RANK:{self.cp_rank} | "
                f"ACTION:completed_{operation_type} | "
                f"{'FROM' if operation_type == 'receive' else 'TO'}:{peer_rank}",
                flush=True,
            )

        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []
        print(
            f"RingComm | wait | RANK:{self.cp_rank} | "
            f"ACTION:all_operations_completed",
            flush=True,
        )

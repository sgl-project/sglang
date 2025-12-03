import logging
import torch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
import mori
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def init_mori_op(group, world_size, rank, hidden_dim, scale_dim, input_dtype, max_num_token, num_local_experts, topk):
    cpu_group = group.cpu_group
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")
    logger.info(f'[MORI init] {world_size=} {rank=} {hidden_dim=} {scale_dim=} {input_dtype=} {max_num_token=} {num_local_experts=} {topk=}')

    if world_size <= 8:
        # single node
        kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        warp_num_per_block = 16
        block_num = 80
        rdma_block_num = 0
    else:
        # multi node
        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
        warp_num_per_block = 8
        block_num = 64
        rdma_block_num = 32

    mori_config = mori.ops.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        data_type=fp8_dtype,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=input_dtype.itemsize,
        max_num_inp_token_per_rank=max_num_token,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        kernel_type=kernel_type,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=2
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    return mori_op


class MoriDispatcher(BaseDispatcher):

    def __init__(self, *args, **kwargs):
        self.mori_op = init_mori_op(*args, **kwargs)

    def dispatch(self, hidden_states, topk_output, scale=None):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        return self.mori_op.dispatch(hidden_states, topk_weights, scale, topk_ids)
    
    def combine(self, combine_input, topk_ids):
        return self.mori_op.combine(combine_input, None, topk_ids)

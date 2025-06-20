# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/model_executor/layers/fused_moe/pplx_prepare_finalize.py
import logging
import threading
from typing import Optional
from weakref import WeakValueDictionary

import pplx_kernels as pplx
import torch
from pplx_kernels.nvshmem import (
    nvshmem_alloc_empty_unique_id,
    nvshmem_finalize,
    nvshmem_get_unique_id,
    nvshmem_init,
)

from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.layers.moe.ep_moe.utils import moe_kernel_quantize_input
from sglang.srt.utils import run_once

logger = logging.getLogger(__name__)


# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/distributed/parallel_state.py#L940
PPLX_DID_INIT: bool = False


# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/model_executor/layers/fused_moe/layer.py#L277
class AllToAllCache:

    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    def destroy(self):
        with self._lock:
            # TODO: can we do del self._cache?
            for _, a2a in self._cache.items():
                a2a.destroy()
                pplx_finalize()

    def get_or_create(self, **kwargs):

        # Create a hashable key from the kwargs
        key = tuple(sorted((k, v) for k, v in kwargs.items()))

        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                # TODO check if it's right
                pplx_init(kwargs["rank"], kwargs["world_size"])

                # TODO (varun): Add support to switch to intranode
                # when all communications are within the same
                # node.
                logger.debug("Create AllToAll %s", kwargs)
                instance = pplx.AllToAll.internode(**kwargs)
                self._cache[key] = instance
            return instance


# Global singleton
_all_to_all_cache = AllToAllCache()


# Factory function as a cleaner interface
def get_all_to_all(**kwargs):
    return _all_to_all_cache.get_or_create(**kwargs)


# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/distributed/parallel_state.py#L944
@run_once
def pplx_init(rank, world_size):
    if world_size > 1:
        try:
            global PPLX_DID_INIT
            logger.debug(
                "Initialize NVSHMEM for PPLX kernels: rank=%d, " "world size=%d",
                rank,
                world_size,
            )
            uid = (
                nvshmem_get_unique_id()
                if rank == 0
                else nvshmem_alloc_empty_unique_id()
            )
            uid_gpu = uid.cuda()
            get_world_group().broadcast(uid_gpu, src=0)
            uid = uid_gpu.to(device="cpu")
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, rank, world_size)
            PPLX_DID_INIT = True
        except Exception as ex:
            logger.error("Failed to initialize NVSHMEM for PPLX: %s", ex)


# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/distributed/parallel_state.py#L968
@run_once
def pplx_finalize():
    global PPLX_DID_INIT
    if PPLX_DID_INIT:
        logger.debug("PPLX NVSHMEM finalize")
        _all_to_all_cache.destroy()
        nvshmem_finalize()


# Note use: layer.get_all_to_all() to get an AllToAll instance
# The max_num_tokens, world_size and dp_size must be the same
# as the ones used to create the AllToAll.
class PplxPrepareAndFinalize:

    def __init__(
        self,
        a2a: pplx.AllToAll,
        max_num_tokens: int,
        world_size: int,
        rank: int,
        dp_size: int,
        num_experts: int,
        quant_dtype: Optional[torch.dtype] = None,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__()
        assert max_num_tokens > 0
        self.a2a = a2a
        self.block_shape = block_shape
        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.rank = rank
        self.num_experts = num_experts
        self.quant_dtype = quant_dtype
        self.dp_size = dp_size

        # rem_experts need to be 0 for pplx to work properly.
        rem_experts = self.num_experts % self.world_size
        assert rem_experts == 0
        self.num_local_experts = (self.num_experts // self.world_size) + (
            1 if self.rank < rem_experts else 0
        )

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        rank_topk_weights: torch.Tensor,
        rank_topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        num_tokens = a1.size(0)  # M
        hidden_dim = a1.size(-1)  # K

        assert rank_topk_ids.size(0) == num_tokens
        # assert expert_map is None, "NYI"

        # Is this always going to be a1.device?
        device = a1.device

        if apply_router_weight_on_input:
            topk = rank_topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert (
                topk == 1
            ), "apply_router_weight_on_input is only implemented for topk=1"
            a1 = a1 * rank_topk_weights.to(a1.dtype)

        per_act_token = (
            a1_scale.numel() != 1
            if a1_scale is not None
            else (a2_scale.numel() != 1 if a2_scale is not None else False)
        )

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1, a1_scale, self.quant_dtype, per_act_token, self.block_shape
        )

        expert_num_tokens = torch.empty(
            self.num_local_experts,
            dtype=torch.int32,
            device=device,
        )

        expert_x = torch.empty(
            (self.num_local_experts, self.max_num_tokens * self.dp_size, hidden_dim),
            dtype=a1q.dtype,
            device=device,
        )

        expert_x_scale: Optional[torch.Tensor] = None
        if a1q.dtype.itemsize == 1:
            float32_size = torch.float32.itemsize
            block_size = (
                self.block_shape[0] if self.block_shape is not None else 1
            ) * float32_size
            expert_x_scale = torch.empty(
                (
                    self.num_experts,
                    expert_x.size(1),
                    (expert_x.size(2) + block_size - 1) // block_size,
                ),
                dtype=torch.float32,
                device=device,
            )

        # This argument is optional, defaults to indices.size(0)
        # There's not much point setting this unless it is != indices.size(0)
        bound_m: Optional[torch.Tensor] = None

        self.a2a.dispatch(
            out_expert_num_tokens=expert_num_tokens,
            out_expert_x=expert_x,
            out_expert_x_scale=expert_x_scale,
            dp_x=a1q,
            dp_x_scale=a1q_scale,
            indices=rank_topk_ids,
            bound_m=bound_m,
        )

        return expert_x, expert_x_scale, expert_num_tokens

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        num_tokens = output.size(0)  # M
        # This argument is optional
        # There's not much point setting this unless it is != topk_ids.size(0)
        bound_m: Optional[torch.Tensor] = None

        assert topk_ids.size(0) == num_tokens, f"{topk_ids.size(0)} == {num_tokens}"
        assert (
            output.size(0) <= self.max_num_tokens
        ), f"{output.size(0)} <= {self.max_num_tokens}"
        assert output.size(1) == fused_expert_output.size(-1)

        # Set weights to 1 if we did them in dispatch. This is hacky.
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        self.a2a.combine(
            out_tokens=output,
            indices=topk_ids,
            weights=topk_weights,
            expert_y=fused_expert_output,
            bound_m=bound_m,
        )


# Adapted from https://github.com/vllm-project/vllm/blob/f9c069c85e029830094ff9abb926ffbf37b7c7e7/vllm/model_executor/layers/fused_moe/layer.py#L683
def construct_prepare_finalize(
    max_num_tokens: int,
    num_experts: int,
    experts_per_token: int,
    rank: int,
    world_size: int,
    dp_size: int,
    hidden_dim: int,
    param_dtype: torch.dtype,
    block_size: int,
) -> Optional[PplxPrepareAndFinalize]:
    logger.debug("using PplxPrepareAndFinalize")
    assert world_size % dp_size == 0

    all_to_all = get_all_to_all(
        max_num_tokens=max_num_tokens,
        num_experts=num_experts,
        experts_per_token=experts_per_token,  # topk
        rank=rank,
        world_size=world_size,
        dp_size=world_size // dp_size,  # dp_size means attn_tp_size in pplx
        hidden_dim=hidden_dim,
        hidden_dim_bytes=hidden_dim * param_dtype.itemsize,
        # For blocked per token: set to
        #   ceil_div(hidden_dim, block_size) * sizeof(float32)
        # For per-token: set to sizeof(float32)
        hidden_dim_scale_bytes=(
            0
            if param_dtype.itemsize != 1
            else ((hidden_dim + block_size - 1) // block_size * torch.float32.itemsize)
        ),
    )

    return PplxPrepareAndFinalize(
        all_to_all,
        max_num_tokens=max_num_tokens,
        world_size=world_size,
        rank=rank,
        dp_size=dp_size,
        num_experts=num_experts,
        quant_dtype=param_dtype,
    )

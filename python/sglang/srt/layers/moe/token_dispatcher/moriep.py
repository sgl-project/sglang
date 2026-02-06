from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hip

if TYPE_CHECKING:
    from sglang.srt.single_batch_overlap import CombineOverlapArgs
    import mori

from enum import Enum, auto
from functools import lru_cache

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import QuantType, get_hip_quant

logger = logging.getLogger(__name__)


class MoriEPNormalDispatchOutput(NamedTuple):
    """Mori EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


assert isinstance(MoriEPNormalDispatchOutput, DispatchOutput)


class MoriEPNormalCombineInput(NamedTuple):
    """Mori EP combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


assert isinstance(MoriEPNormalCombineInput, CombineInput)


class EpMode(Enum):
    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True)
class EpDispatchConfig:
    kernel_type: mori.ops.EpDispatchCombineKernelType
    warp_num_per_block: int
    block_num: int
    rdma_block_num: int


def get_ep_dispatch_configs():
    import mori

    return {
        EpMode.INTRA_NODE: EpDispatchConfig(
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=80,
            rdma_block_num=0,
        ),
        EpMode.INTER_NODE: EpDispatchConfig(
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
    }


# init_mori_op only needs do once in model initial stage
# use lru_cache to reuse the same mori_op instance to avoid the init overhead for mori
@lru_cache(maxsize=1)
def init_mori_op(
    group,
    router_topk,
    num_experts,
    num_local_experts,
    hidden_size,
    params_dtype,
    num_max_dispatch_tokens_per_rank,
):

    import mori

    world_size = get_moe_expert_parallel_world_size()
    rank = get_moe_expert_parallel_rank()

    cpu_group = group.cpu_group
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")
    logger.info(
        f"[MORI init] {world_size=} {rank=} {hidden_size=} {params_dtype=} {num_max_dispatch_tokens_per_rank=} {num_local_experts=} {router_topk=}"
    )

    mode = EpMode.INTRA_NODE if world_size <= 8 else EpMode.INTER_NODE
    cfg = get_ep_dispatch_configs()[mode]

    kernel_type = cfg.kernel_type
    warp_num_per_block = cfg.warp_num_per_block
    block_num = cfg.block_num
    rdma_block_num = cfg.rdma_block_num

    mori_config = mori.ops.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        data_type=fp8_dtype,
        hidden_dim=hidden_size,
        scale_dim=(
            hidden_size // 128
            if get_bool_env_var("SGLANG_MORI_FP8_DISP", "False")
            else 1
        ),
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=params_dtype.itemsize,
        max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        kernel_type=kernel_type,
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=2,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    return mori_op


class _MoriEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        return_recv_hook: bool,
        deepep_mode: DeepEPMode,
    ):
        try:
            import mori  # noqa: F401
        except ImportError:
            raise ImportError("Mori EP is not installed. Please install.")
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.return_recv_hook = return_recv_hook
        self.deepep_mode = deepep_mode

        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )

        self.mori_op = init_mori_op(
            self.group,
            self.router_topk,
            self.num_experts,
            self.num_local_experts,
            self.hidden_size,
            self.params_dtype,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
        )

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError


class _MoriEPDispatcherImplNormal(_MoriEPDispatcherImplBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quant_config = {}
        # [kk TODO] need to support mxfp4 type
        self.quant_func = get_hip_quant(QuantType.per_1x128)

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        return (
            hidden_states,
            topk_weights,
            topk_ids,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
    ):
        num_token = hidden_states.shape[0]
        scale = None

        fp8_dispatch = get_bool_env_var("SGLANG_MORI_FP8_DISP", "False")

        if fp8_dispatch:
            # FP8 quant
            if num_token > 0:
                # NOTE: aiter is able to handle token=0 case in UT. But for some reason it failed at e2e case. Root cause TBD.
                hidden_states, scale = self.quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )
            else:
                hidden_states = torch.empty(
                    hidden_states.shape, dtype=fp8_dtype, device=hidden_states.device
                )
                scale = torch.empty(
                    (0, self.hidden_size // 128),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )

        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self._dispatch_core(hidden_states, topk_weights, topk_ids, scale)

        return MoriEPNormalDispatchOutput(
            packed_recv_hidden,
            recv_scales,
            recv_topk_ids,
            recv_topk_weights,
            packed_recv_count,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ):
        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self.mori_op.dispatch(hidden_states, topk_weights, scale, topk_ids)

        # TODO(billishyahao): EPLB
        # get_global_expert_distribution_recorder().on_deepep_dispatch_normal(

        return (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        previous_event = None
        return hidden_states, topk_ids, topk_weights, previous_event

    def combine_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        hidden_states = self._combine_core(hidden_states, topk_ids, topk_weights)
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        combined_hidden_states = self.mori_op.combine(hidden_states, None, topk_ids)
        return combined_hidden_states[0]

    def set_quant_config(self, quant_config: dict):
        self.quant_config = quant_config


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class MoriEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.deepep_mode = deepep_mode

        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _MoriEPDispatcherImplNormal(
                group=group,
                router_topk=router_topk,
                permute_fusion=permute_fusion,
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                hidden_size=hidden_size,
                params_dtype=params_dtype,
                return_recv_hook=return_recv_hook,
                deepep_mode=deepep_mode,
            )
        if self.deepep_mode.enable_low_latency():
            raise NotImplementedError

        self._stage = _Stage.INITIAL

    def dispatch(self, *args, **kwargs) -> DispatchOutput:
        self.dispatch_a(*args, **kwargs)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ) -> Tuple:
        self.combine_a(combine_input, overlap_args)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
        overlap_args: Optional[CombineOverlapArgs] = None,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            overlap_args=overlap_args,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _MoriEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        if self.deepep_mode.enable_low_latency():
            raise NotImplementedError
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

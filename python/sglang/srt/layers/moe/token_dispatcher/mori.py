from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

import aiter
import torch

# from sglang.srt.layers.moe.token_dispatcher.base_dispatcher import BaseDispatcher
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.utils import MoRIEPMode, get_deepep_config, is_tbo_enabled
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.utils import (
    get_bool_env_var,
    get_int_env_var,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()

# TODO:
# 1. Find substitution of deep_ep Buffer and Config
# 2. Change the deep_ep conditions into mori_ep conditions
try:
    from mori.ops.dispatch_combine import (
        EpDispatchCombineConfig,
        EpDispatchCombineKernelType,
        EpDispatchCombineOp,
    )

    # NOTE: Could we need to support 'npu' devices?
    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_mori = True
except ImportError:
    use_mori = False


from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_use_fp8_dispatch = get_bool_env_var("SGLANG_MORI_FP8_DISPATCH") and _use_aiter

logger = logging.getLogger(__name__)


# --------------------------- MoRI Dispatch Output ---------------------------------
# TODO: Change the output format to meet MoRI Dispatch output
# * [ ] MoRINormalOutput
# * [V] MoRILLOutput


class MoRINormalOutput(NamedTuple):
    """MoRI normal dispatch output."""

    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    # hidden_states_scale
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_NORMAL


class MoRILLOutput(NamedTuple):
    """MoRI low latency dispatch output."""

    hidden_states: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    scales: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.MORI_LL


assert isinstance(MoRINormalOutput, DispatchOutput)
assert isinstance(MoRILLOutput, DispatchOutput)


# ----------------------------- MoRI Combine Input ----------------------------------
# TODO: Change the input format to meet MoRI Combine input


class MoRINormalCombineInput(NamedTuple):
    """MoRI normal combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORI_NORMAL


class MoRILLCombineInput(NamedTuple):
    """MoRI low latency combine input."""

    pass

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.MORI_LL


assert isinstance(MoRINormalCombineInput, CombineInput)
assert isinstance(MoRILLCombineInput, CombineInput)


class MoRIDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


_GLOBAL_MORI_OPS_HANDLE: EpDispatchCombineOp = None
_GLOBAL_MORI_CONFIG: EpDispatchCombineConfig = None


class _MoRIDispatcherImplBase:
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
        use_fp8_w8a8: bool = False,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        if not use_mori:
            raise ImportError(
                "MoRI is not installed. Please install MoRI package from "
                "https://github.com/ROCm/mori."
            )
        self.config = config
        global _GLOBAL_MORI_OPS_HANDLE
        if _GLOBAL_MORI_OPS_HANDLE is None:
            self._ops_handle = EpDispatchCombineOp(config)
            _GLOBAL_MORI_OPS_HANDLE = self._ops_handle
        else:
            self._ops_handle = _GLOBAL_MORI_OPS_HANDLE
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.quant_dtype = quant_dtype
        self.moriep_mode = moriep_mode

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError


class _MoRIDispatcherImplNormal(_MoRIDispatcherImplBase):
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
        use_fp8_w8a8: bool = False,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        super().__init__(config, moriep_mode, use_fp8_w8a8, quant_dtype)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError("mori normal mode is currently not supported.")

    def combine(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError("mori normal mode is currently not supported.")


class _MoRIDispatcherImplLowLatency(_MoRIDispatcherImplBase):
    def __init__(
        self,
        config: EpDispatchCombineConfig,
        moriep_mode: MoRIEPMode,
        use_fp8_w8a8: bool = False,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        assert (
            moriep_mode == MoRIEPMode.LOW_LATENCY
        ), f"Invalid moriep_mode: {moriep_mode}."
        super().__init__(config, moriep_mode, use_fp8_w8a8, quant_dtype)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        scales = None
        if self.use_fp8_w8a8 and _use_fp8_dispatch:
            from aiter import QuantType, get_hip_quant

            # NOTE: get_hip_quant not supports QuantType.per_128x128.
            quant_type = QuantType.per_1x128
            quant_func = get_hip_quant(quant_type)
            hidden_states, scales = quant_func(
                hidden_states,
                quant_dtype=aiter.dtypes.fp8,
            )

        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = self._ops_handle.dispatch(
            input=hidden_states,
            weights=topk_weights,
            scales=scales,
            indices=topk_idx,
        )

        return MoRILLOutput(
            hidden_states=dispatch_output,
            topk_idx=dispatch_indices,
            topk_weights=dispatch_weights,
            scales=dispatch_scales,
            num_recv_tokens_per_expert=dispatch_recv_num_token,
        )

    def combine(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        num_original_tokens = output.shape[0]  # Original number of tokens

        try:
            combined_outputs, combined_weights = self._ops_handle.combine(
                input=hidden_states,
                weights=topk_weights,
                indices=topk_idx,
            )
            output.copy_(combined_outputs[:num_original_tokens], non_blocking=True)
        except Exception as e:
            logger.error(f"mori combine failed: {e}")
            raise RuntimeError(f"mori combine failed: {e}") from e


from sglang.srt.distributed.parallel_state import GroupCoordinator


# TODO: should Implement MORI dispatcher, below is nonsense code for removal
# NOTE: to implement TBO with MoRI, first use same function name and logic flow
#       of DeepEPDispatcher (check _use_aiter or _use_hip in deepep.py)
class MoRIDispatcher(BaseDispatcher):
    def __init__(
        self,
        # group: torch.distributed.ProcessGroup, # NOTE: Changed to get GroupCoordinator
        group: GroupCoordinator,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        use_fp8_w8a8: bool = False,
        quant_dtype: torch.dtype = torch.float8_e4m3fnuz,
        moriep_mode: MoRIEPMode = MoRIEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        if not use_mori:
            raise ImportError(
                "MoRI is not installed. Please install MoRI package from "
                "https://github.com/ROCm/mori."
            )

        # Deepep initializes params_dtype from deepseek config but we have to manually
        # if it not works
        if params_dtype is None:
            if _use_fp8_dispatch and use_fp8_w8a8:
                params_dtype = aiter.dtypes.fp8
            else:
                params_dtype = torch.bfloat16

        # TODO: Clean the unused APIs
        from sglang.srt.distributed.parallel_state import (
            get_tp_group,
            in_the_same_node_as,
        )

        self.group = group.device_group
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.quant_dtype = quant_dtype
        self._internode = False
        assert dist.get_backend(group.cpu_group) != dist.Backend.NCCL, (
            f"NCCL backend not support inter-node communication. "
            f"backend: {dist.get_backend(group.cpu_group)}"
        )
        if not all(in_the_same_node_as(group.cpu_group, source_rank=0)):
            self._internode = True

        # We use rank of _MOE_EP
        self.rank = group.rank_in_group
        self.world_size = get_tp_group().world_size

        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 256
        )

        self.config = self._make_mori_config(
            data_type=params_dtype,
            hidden_dim=hidden_size,
            rank=self.rank,
            world_size=self.world_size,
            max_num_tokens=self.num_max_dispatch_tokens_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=router_topk,
        )

        # NOTE:
        # 1. mori does not use recv hook like deepep. So there is no distinguish
        #    of each stages
        # 2. Currently, mori has low-latency mode only but high-throughput
        #    mode will be added later. So, the 'normal' mode should be change
        #    to 'ht' (high-throughput) mode in the future.
        self.moriep_mode = moriep_mode
        if self.moriep_mode.enable_normal():
            self._normal_dispatcher = _MoRIDispatcherImplNormal(
                self.config, self.moriep_mode, self.use_fp8_w8a8, self.quant_dtype
            )
        if self.moriep_mode.enable_low_latency():
            self._low_latency_dispatcher = _MoRIDispatcherImplLowLatency(
                self.config, self.moriep_mode, self.use_fp8_w8a8, self.quant_dtype
            )

    # NOTE: Moved to MoRIEPMoE class
    # def _init_mori_shmem():
    #     pass

    def _make_mori_config(
        self,
        hidden_dim: int,
        rank: int,
        world_size: int,
        max_num_tokens: int,
        num_experts_per_rank: int,
        num_experts_per_token: int,
        data_type: torch.dtype = torch.bfloat16,
    ):
        global _GLOBAL_MORI_CONFIG
        if _GLOBAL_MORI_CONFIG is not None:
            return _GLOBAL_MORI_CONFIG

        # Determine data type size
        dtype_to_size = {
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float8_e4m3fnuz: 1,
            torch.float8_e4m3fn: 1,
        }
        max_token_type_size = dtype_to_size.get(data_type, 2)

        scale_shape = self.scale_shape(max_num_tokens, hidden_dim)
        _scale_dim = scale_shape[-1] if scale_shape is not None else 0
        _scale_type_size = torch.float32.itemsize
        _GLOBAL_MORI_CONFIG = EpDispatchCombineConfig(
            data_type=data_type,
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_tokens,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,
            # Performance tuning parameters (can be optimized later)
            # warp_num_per_block=8,  # Good default for MI300X
            # block_num=80,          # Good default for MI300X
            max_token_type_size=max_token_type_size,
            # Quantization support
            # NOTE: scale_dim calc function from vLLM's `scale_shape` function.
            scale_dim=_scale_dim,
            scale_type_size=_scale_type_size,  # scale from aiter uses fp32
            # Use internal buffer management
            # use_external_inp_buf=False,
            # Determine kernel type based on topology
            kernel_type=(
                EpDispatchCombineKernelType.InterNode
                if self._internode
                else EpDispatchCombineKernelType.IntraNode
            ),
        )

        logger.debug(
            f"[rank:{rank}] mori dispatcher created with configs: "
            f"{_GLOBAL_MORI_CONFIG=}"
        )

        return _GLOBAL_MORI_CONFIG

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> DispatchOutput:
        ret = self._get_impl(forward_batch).dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        return ret

    def combine(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple:
        ret = self._get_impl(forward_batch).combine(
            output,
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        return ret

    def _get_impl(self, forward_batch: ForwardBatch) -> _MoRIDispatcherImplBase:
        resolved_moriep_mode = self.moriep_mode.resolve(
            forward_batch.is_extend_in_batch
        )
        if resolved_moriep_mode == MoRIEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_moriep_mode == MoRIEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid moriep_mode: {self.moriep_mode}")

    def scale_shape(
        self,
        max_tokens: int,
        hidden_dim: int,
    ) -> Optional[tuple[int, int]]:
        from sglang.srt.layers.moe.ep_moe.layer import get_mori_quant_config

        global _use_fp8_dispatch
        quant_config = get_mori_quant_config()

        if _use_fp8_dispatch and self.use_fp8_w8a8:
            if quant_config["use_block_quant"]:
                assert quant_config["block_shape"] is not None
                _, block_k = quant_config["block_shape"]
                k_tiles = cdiv(hidden_dim, block_k)
                return (max_tokens, k_tiles)
            else:
                return (1, 1)
        else:
            return None


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

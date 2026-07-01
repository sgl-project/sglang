from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalDispatchOutput,
    DeepEPPDispatchHooks,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPOutputDtype,
    get_deepep_output_dtype,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, is_hip, is_npu

_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    if _is_npu:
        raise ImportError("DeepEP v2 ElasticBuffer is GPU-only")

    from deep_ep import ElasticBuffer

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    use_deepep_v2 = True
except ImportError:
    use_deepep_v2 = False

logger = logging.getLogger(__name__)


def _should_record_stat_approx() -> bool:
    try:
        return (
            get_global_server_args().expert_distribution_recorder_mode == "stat_approx"
        )
    except ValueError:
        return False


class DeepEPV2Buffer:
    _buffer = None
    _buffer_key = None
    _num_comm_sms = 0

    @classmethod
    def allow_hybrid_mode(cls) -> bool:
        return not get_bool_env_var("EP_DISABLE_GIN")

    @classmethod
    def get_buffer(
        cls,
        group: torch.distributed.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        use_fp8_dispatch: bool,
    ):
        if not use_deepep_v2:
            raise ImportError(
                "DeepEP v2 requires a DeepEP package that exports ElasticBuffer. "
                "Please install a recent DeepEP package from "
                "https://github.com/deepseek-ai/DeepEP."
            )

        allow_hybrid_mode = cls.allow_hybrid_mode()
        required_bytes = ElasticBuffer.get_buffer_size_hint(
            group,
            num_max_dispatch_tokens_per_rank,
            hidden_size,
            num_topk=router_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        buffer_key = (
            group,
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
        )

        if (
            cls._buffer is not None
            and cls._buffer_key == buffer_key
            and cls._buffer.num_bytes >= required_bytes
        ):
            return cls._buffer, cls._num_comm_sms

        cls._buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=router_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        cls._buffer_key = buffer_key
        cls._num_comm_sms = cls._buffer.get_theoretical_num_sms(
            num_experts, router_topk
        )
        logger.info(
            "Initialized DeepEP v2 ElasticBuffer with %s bytes and %s comm SMs.",
            cls._buffer.num_bytes,
            cls._num_comm_sms,
        )
        return cls._buffer, cls._num_comm_sms


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPV2Dispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        async_finish: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not use_deepep_v2:
            raise ImportError(
                "DeepEP v2 requires a DeepEP package that exports ElasticBuffer. "
                "Please install a recent DeepEP package from "
                "https://github.com/deepseek-ai/DeepEP."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.async_finish = async_finish
        self.handle = None
        self._current_use_fp8_dispatch = False
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

        self.expert_mask_gpu = None
        if _use_aiter and num_local_experts is not None:
            expert_mask = torch.zeros(
                num_local_experts + 1,
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            expert_mask[:-1] = 1
            self.expert_mask_gpu = expert_mask

        self.set_deepep_dispatcher_dtype()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        topk_ids = topk_output.topk_ids.to(torch.int64)
        topk_weights = topk_output.topk_weights

        if self.use_fp8:
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )

        previous_event = ElasticBuffer.capture() if self.async_finish else None
        self._dispatch_intermediate_state = (
            hidden_states,
            topk_ids,
            topk_weights,
            previous_event,
        )

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        hidden_states, topk_ids, topk_weights, previous_event = (
            self._dispatch_intermediate_state
        )
        del self._dispatch_intermediate_state

        buffer, num_sms = self._get_buffer(isinstance(hidden_states, tuple))
        self._current_use_fp8_dispatch = isinstance(hidden_states, tuple)
        recv_x, recv_topk_ids, recv_topk_weights, self.handle, event = buffer.dispatch(
            hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            num_sms=num_sms,
            previous_event=previous_event,
            async_with_compute_stream=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
            use_tma_aligned_col_major_sf=(
                isinstance(hidden_states, tuple)
                and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
            ),
        )
        if self.async_finish:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            recv_x, recv_x_scale = recv_x
        else:
            recv_x_scale = None

        if _should_record_stat_approx():
            get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
                self.handle.num_recv_tokens_per_expert_list,
                num_tokens_per_rank=None,
                num_tokens_per_rdma_rank=None,
                num_tokens_per_expert=None,
            )

        return DeepEPNormalDispatchOutput(
            recv_x,
            recv_x_scale,
            recv_topk_ids,
            recv_topk_weights,
            self.handle.num_recv_tokens_per_expert_list,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

    def combine_a(self, combine_input: CombineInput):
        hidden_states, _topk_ids, _topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        previous_event = ElasticBuffer.capture() if self.async_finish else None
        self._combine_intermediate_state = (
            hidden_states,
            previous_event,
        )

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        hidden_states, previous_event = self._combine_intermediate_state
        del self._combine_intermediate_state

        buffer, num_sms = self._get_buffer(self._current_use_fp8_dispatch)
        combined_hidden_states, _, event = buffer.combine(
            hidden_states,
            handle=self.handle,
            num_sms=num_sms,
            previous_event=previous_event,
            async_with_compute_stream=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        if self.async_finish:
            event.current_stream_wait()
        self.handle = None
        self._current_use_fp8_dispatch = False
        return combined_hidden_states

    def _get_buffer(self, use_fp8_dispatch: bool):
        return DeepEPV2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8_dispatch,
        )

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        self.set_deepep_dispatcher_dtype()

    def set_deepep_dispatcher_dtype(self) -> None:
        self.deepep_output_dtype = get_deepep_output_dtype(self)
        if self.deepep_output_dtype == DeepEPOutputDtype.NVFP4:
            raise RuntimeError(
                "deepep_v2 currently supports bf16/fp8 dispatch only; "
                "nvfp4 dispatch needs a dedicated ElasticBuffer path."
            )
        if self.deepep_output_dtype == DeepEPOutputDtype.INT8:
            raise RuntimeError(
                "deepep_v2 currently supports bf16/fp8 dispatch only; "
                "int8 dispatch is NPU-specific and unsupported by ElasticBuffer."
            )
        self.use_fp8 = self.deepep_output_dtype == DeepEPOutputDtype.FP8

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        raise ValueError("deepep_v2 does not support single-batch overlap yet.")

    def clear_overlap_args(self):
        super().clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

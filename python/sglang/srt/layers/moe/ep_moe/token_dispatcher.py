import logging
from dataclasses import dataclass

from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.managers.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import DeepEPMode, get_int_env_var, load_json_config

try:
    from deep_ep import Buffer, Config

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    use_deepep = True
except ImportError:
    use_deepep = False

from enum import Enum, IntEnum, auto
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.ep_moe.kernels import (
    deepep_permute_triton_kernel,
    deepep_post_reorder_triton_kernel,
    deepep_run_moe_deep_preprocess,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = None,
        num_experts: int = None,
    ):
        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank is not None
            assert num_experts is not None and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        if deepep_mode == DeepEPMode.normal:
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms // 2
        elif deepep_mode in [DeepEPMode.low_latency, DeepEPMode.auto]:
            num_qps_per_rank = num_experts // group.size()
        else:
            raise NotImplementedError

        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            # TODO can be false when unneeded
            allow_mnnvl=True,
        )
        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        if not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY


class DeepEPConfig:
    _instance = None

    def __init__(self):
        config_str = global_server_args_dict["deepep_config"]
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128
        )

        self.handle = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        topk_idx = topk_idx.to(torch.int64)
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            # TODO hard code 128 block quant,use fp8 communication
            hidden_states = sglang_per_token_group_quant_fp8(hidden_states, 128)
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_idx, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_idx, topk_weights, previous_event):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
                event,
            ) = self._dispatch_core(
                hidden_states, topk_idx, topk_weights, previous_event
            )
            event.current_stream_wait() if self.async_finish else ()
            return (
                hidden_states,
                topk_idx,
                topk_weights,
                None,
                num_recv_tokens_per_expert_list,
                None,
                None,
                None,
            )
        else:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                num_recv_tokens_per_expert_list,
                event,
            ) = self._dispatch_core(
                hidden_states, topk_idx, topk_weights, previous_event
            )
            event.current_stream_wait() if self.async_finish else ()
            if hidden_states.shape[0] > 0:
                reorder_topk_ids, seg_indptr, hidden_states = self._deepep_permute(
                    hidden_states, topk_idx, fp8_dtype=hidden_states.dtype
                )
            else:
                reorder_topk_ids = torch.empty(
                    (0,), device=hidden_states.device, dtype=torch.int64
                )
                seg_indptr = torch.zeros(
                    (self.num_experts + 1,),
                    device=hidden_states.device,
                    dtype=torch.int64,
                )

            masked_m = expected_m = None
            return (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                None,
                seg_indptr,
                masked_m,
                expected_m,
            )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )

        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )

        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert_list,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            event,
        )

    def _deepep_permute(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        fp8_dtype: Optional[torch.dtype] = None,
        use_fp8_w8a8: bool = False,
        use_block_quant: bool = False,
    ):
        """
        Copy from Megatron-Core token_dispatcher MoEFlexTokenDispatcher
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py
        """

        reorder_topk_ids, self.src2dst, seg_indptr = deepep_run_moe_deep_preprocess(
            topk_idx, self.num_experts
        )
        num_total_tokens = reorder_topk_ids.numel()
        gateup_input = torch.empty(
            (int(num_total_tokens), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                fp8_dtype
                if (use_fp8_w8a8 and not use_block_quant)
                else hidden_states.dtype
            ),
        )
        # PreReorder
        deepep_permute_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            self.src2dst,
            topk_idx,
            None,
            self.router_topk,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )
        return reorder_topk_ids, seg_indptr, gateup_input

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            output = hidden_states
        else:
            if hidden_states.shape[0] > 0:
                num_tokens = self.src2dst.shape[0] // self.router_topk
                output = torch.empty(
                    (num_tokens, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                deepep_post_reorder_triton_kernel[(num_tokens,)](
                    hidden_states,
                    output,
                    self.src2dst,
                    topk_idx,
                    topk_weights,
                    self.router_topk,
                    hidden_states.shape[1],
                    BLOCK_SIZE=512,
                )
            else:
                output = torch.zeros(
                    (0, hidden_states.shape[1]),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_normal()

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        topk_idx = topk_idx.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_idx.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_idx,
            use_fp8=True,
        )
        return (
            hidden_states,
            topk_idx,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_idx,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        reorder_topk_ids = seg_indptr = None

        return (
            hidden_states,
            topk_idx,
            topk_weights,
            reorder_topk_ids,
            None,
            seg_indptr,
            masked_m,
            expected_m,
        )

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        use_fp8: bool = False,
    ):
        buffer = self._get_buffer()
        packed_recv_hidden, packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_idx,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
        )
        return packed_recv_hidden, packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_idx,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        hook() if self.return_recv_hook else event.current_stream_wait()
        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        combined_hidden_states, event, hook = buffer.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            self.handle,
            async_finish=not self.return_recv_hook,
            return_recv_hook=self.return_recv_hook,
        )
        self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL

    def dispatch(self, *args, **kwargs) -> Tuple:
        self.dispatch_a(*args, **kwargs)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_mode: ForwardMode = None,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl(forward_mode).dispatch_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        self._dispatch_intermediate_state = forward_mode, inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        forward_mode, inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl(forward_mode).dispatch_b(*inner_state)

    def combine(self, *args, **kwargs) -> Tuple:
        self.combine_a(*args, **kwargs)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl(forward_mode).combine_a(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = forward_mode, inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        forward_mode, inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl(forward_mode).combine_b(*inner_state)

    def _get_impl(self, forward_mode: ForwardMode) -> _DeepEPDispatcherImplBase:
        resolved_deepep_mode = self.deepep_mode.resolve(forward_mode)
        if resolved_deepep_mode == DeepEPMode.normal:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

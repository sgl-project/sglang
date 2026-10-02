"""MSCCL++ expert-parallel dispatcher implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput
from sglang.srt.layers.moe.utils import MSCCLPPEPLayout, MSCCLPPMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from .utils import (
    MSCCLPPCombineInputBase,
    MSCCLPPDispatchOutputBase,
    MSCCLPPExpertMajorLatencyDispatchOutput,
    MSCCLPPRankMajorLatencyDispatchOutput,
)


class _MSCCLPPDispatcherImplBase(ABC):
    @abstractmethod
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPDispatchOutputBase:
        pass

    @abstractmethod
    def combine(self, combine_input: MSCCLPPCombineInputBase) -> torch.Tensor:
        pass


class _MSCCLPPDispatcherImplLowLatency(_MSCCLPPDispatcherImplBase):
    """MSCCL++ EP low-latency all-to-all implementation.

    * :meth:`dispatch` runs ``MoECommunicator.dispatch`` to scatter each token to
      the ranks owning its top-k experts and returns an
      :class:`MSCCLPPLatencyDispatchOutput`. Triton uses padded expert-major output;
      FlashInfer CUTLASS uses the communicator's fixed rank-major buffers.
    * :meth:`combine` runs ``MoECommunicator.combine`` to reduce the per-slot
      expert outputs back to each source token.

    Expert-major combine applies routing weights from the handle, so Triton runs
    with unit weights. Rank-major CUTLASS applies weights while producing one
    rank-local partial per row, which combine reduces on the source rank without
    applying weights again.
    """

    # One heavy resource set per (group, geometry, capacity, layout), reused by
    # every MoE layer. Static buffer addresses are also required by CUDA graphs.
    _shared_resources: ClassVar[
        dict[
            tuple[
                object,
                int,
                int,
                int,
                int,
                int,
                MSCCLPPEPLayout,
                bool,
                Optional[tuple[int, int]],
            ],
            tuple[object, object, Optional[torch.Tensor], int],
        ]
    ] = {}

    @staticmethod
    def _resolve_capacity_per_attention_tp(
        num_tokens: int,
        attn_tp_size: int,
    ) -> int:
        if num_tokens < 0 or attn_tp_size <= 0:
            raise ValueError(
                "num_tokens must be non-negative and attn_tp_size positive"
            )
        return (num_tokens + attn_tp_size - 1) // attn_tp_size

    @classmethod
    def _resolve_prefill_capacity(
        cls,
        chunked_prefill_size: int,
        attn_tp_size: int,
    ) -> int:
        """Resolve the per-rank prefill capacity from the chunk size."""
        return cls._resolve_capacity_per_attention_tp(
            chunked_prefill_size,
            attn_tp_size,
        )

    @classmethod
    def _resolve_runtime_decode_capacity(cls, attn_tp_size: int) -> int:
        from sglang.srt.runtime_context import (
            get_exec,
            get_spec,
            max_speculative_num_draft_tokens,
        )

        cuda_graph_config = get_exec().graph.cuda_graph_config
        if cuda_graph_config is None:
            return 0

        spec = get_spec()
        num_tokens_per_bs = 1
        if SpeculativeAlgorithm.from_string(
            spec.speculative_algorithm
        ).is_speculative():
            num_tokens_per_bs = max_speculative_num_draft_tokens()
            if num_tokens_per_bs is None:
                raise ValueError(
                    "MSCCL++ requires speculative_num_draft_tokens to be resolved "
                    "before allocating the communicator"
                )

        return cls._resolve_capacity_per_attention_tp(
            num_tokens_per_bs * cuda_graph_config.decode.max_bs,
            attn_tp_size,
        )

    @classmethod
    def _get_or_create_shared_resources(
        cls,
        group: torch.distributed.ProcessGroup,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        router_topk: int,
        allocation_capacity: int,
        output_layout: MSCCLPPEPLayout,
        enable_direct_send: bool,
        num_blocks: Optional[tuple[int, int]],
    ) -> tuple[object, object, Optional[torch.Tensor], int]:
        key = (
            group,
            num_experts,
            num_local_experts,
            hidden_size,
            router_topk,
            allocation_capacity,
            output_layout,
            enable_direct_send,
            num_blocks,
        )
        cached = cls._shared_resources.get(key)
        if cached is not None:
            return cached

        try:
            from mscclpp import CommGroup
            from mscclpp.ep import (
                CombineMode,
                DispatchLayout,
                MoECommunicator,
                MoECommunicatorConfig,
                MoEMode,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MSCCL++ EP is not available. Install an MSCCL++ build that "
                "provides `mscclpp.ep`, then select the `mscclpp` MoE a2a backend."
            ) from exc

        ep_group = CommGroup(torch_group=group)
        num_ranks = ep_group.nranks

        if not hasattr(DispatchLayout, output_layout.name):
            raise RuntimeError(
                f"MSCCL++ {output_layout.value} dispatch is unavailable; install "
                "the new mscclpp.ep API build."
            )
        native_output_layout = getattr(DispatchLayout, output_layout.name)
        communicator_config = dict(
            comm=ep_group,
            device=torch.cuda.current_device(),
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            local_expert_start=ep_group.my_rank * num_local_experts,
            hidden_size=hidden_size,
            topk=router_topk,
            max_tokens_per_rank=allocation_capacity,
            mode=MoEMode.LATENCY,
            output_layout=native_output_layout,
            invalid_token_expert_id=num_experts,
            num_blocks=num_blocks,
        )
        if enable_direct_send:
            communicator_config["combine_mode"] = CombineMode.DIRECT_SEND
        moe_comm = MoECommunicator(MoECommunicatorConfig(**communicator_config))
        if not moe_comm.is_available():
            raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

        if output_layout == MSCCLPPEPLayout.RANK_MAJOR:
            dispatch_output_buffer = moe_comm.get_dispatch_output_buffer()
        else:
            dispatch_output_buffer = torch.empty(
                (
                    num_local_experts,
                    num_ranks * allocation_capacity,
                    hidden_size,
                ),
                dtype=torch.bfloat16,
                device=torch.device("cuda", torch.cuda.current_device()),
            )

        resources = (ep_group, moe_comm, dispatch_output_buffer, num_ranks)
        cls._shared_resources[key] = resources
        return resources

    @staticmethod
    def _resolve_enable_direct_send(
        output_layout: MSCCLPPEPLayout,
    ) -> bool:
        enable_direct_send = envs.SGLANG_MSCCLPP_ENABLE_DIRECT_SEND.get()
        if enable_direct_send and output_layout != MSCCLPPEPLayout.RANK_MAJOR:
            raise ValueError("MSCCL++ direct send requires rank-major output")
        return enable_direct_send

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        output_layout: MSCCLPPEPLayout = MSCCLPPEPLayout.EXPERT_MAJOR,
    ):
        if not isinstance(output_layout, MSCCLPPEPLayout):
            raise TypeError("output_layout must be an MSCCLPPEPLayout")
        from sglang.srt.runtime_context import get_parallel, get_schedule

        parallel = get_parallel()
        self._attn_tp_size = int(parallel.attn_tp_size)
        self._prefill_capacity = self._resolve_prefill_capacity(
            int(get_schedule().chunked_prefill_size),
            self._attn_tp_size,
        )
        self._decode_capacity = self._resolve_runtime_decode_capacity(
            self._attn_tp_size
        )
        self._allocation_capacity = max(self._prefill_capacity, self._decode_capacity)

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.output_layout = output_layout
        self.enable_direct_send = self._resolve_enable_direct_send(output_layout)
        self.overlap_enabled = (
            output_layout == MSCCLPPEPLayout.RANK_MAJOR
            and envs.SGLANG_MSCCLPP_ENABLE_SHARED_EXPERTS_OVERLAP.get()
        )
        num_blocks = (130, 32) if self.overlap_enabled else None

        # Allocating these capacity-scaled resources per layer would OOM large
        # MoE models, so Latency implementations with identical geometry share them.
        (
            self._ep_group,
            self._moe_comm,
            self._dispatch_output_buffer,
            self.num_ranks,
        ) = self._get_or_create_shared_resources(
            group,
            num_experts,
            num_local_experts,
            hidden_size,
            router_topk,
            self._allocation_capacity,
            output_layout,
            self.enable_direct_send,
            num_blocks,
        )

        # The DispatchHandle produced by dispatch and consumed by combine
        # (carries topk_ids / topk_weights / scatter metadata). Reset after each
        # combine, analogous to DeepEP's ``self.handle``. Kept per-instance: MoE
        # layers run sequentially, so each layer's dispatch->combine pair owns
        # the shared communicator for the duration of its forward.
        self._combine_handle = None

    def _select_active_capacity(self, num_tokens: int) -> int:
        from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

        global_num_tokens = get_dp_global_num_tokens()
        if global_num_tokens is not None and len(global_num_tokens) > 1:
            active_capacity = self._resolve_capacity_per_attention_tp(
                max((int(value) for value in global_num_tokens), default=0),
                self._attn_tp_size,
            )
        else:
            active_capacity = num_tokens

        if active_capacity > self._allocation_capacity:
            raise RuntimeError(
                "MSCCL++ active capacity exceeds the shared allocation: "
                f"active={active_capacity}, allocated={self._allocation_capacity}"
            )
        return active_capacity

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPDispatchOutputBase:
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()

        active_capacity = self._select_active_capacity(hidden_states.shape[0])
        dispatch_out, handle = self._moe_comm.dispatch(
            hidden_states,
            topk_ids,
            topk_weights,
            output_buffer=self._dispatch_output_buffer,
            runtime_max_tokens_per_rank=(
                active_capacity
                if self.output_layout == MSCCLPPEPLayout.RANK_MAJOR
                else None
            ),
        )
        self._combine_handle = handle

        hidden_states_scale = (
            None if dispatch_out.quant is None else dispatch_out.quant.block_scales
        )

        if self.output_layout == MSCCLPPEPLayout.RANK_MAJOR:
            assert dispatch_out.topk_ids is not None
            assert dispatch_out.weights is not None
            assert dispatch_out.layout.num_tokens_per_rank is not None
            assert dispatch_out.combine_input_buffer is not None
            active_rows = self.num_ranks * active_capacity
            return MSCCLPPRankMajorLatencyDispatchOutput(
                hidden_states=dispatch_out.tokens[:active_rows],
                hidden_states_scale=hidden_states_scale,
                topk_output=StandardTopKOutput(
                    dispatch_out.weights[:active_rows],
                    dispatch_out.topk_ids[:active_rows],
                    topk_output.router_logits,
                ),
                expert_output_buffer=dispatch_out.combine_input_buffer,
                enable_direct_send=self.enable_direct_send,
            )

        masked_m = dispatch_out.layout.num_tokens_per_expert
        assert isinstance(masked_m, torch.Tensor)

        return MSCCLPPExpertMajorLatencyDispatchOutput(
            hidden_states=dispatch_out.tokens,
            hidden_states_scale=hidden_states_scale,
            masked_m=masked_m,
        )

    def combine(self, combine_input: MSCCLPPCombineInputBase) -> torch.Tensor:
        assert self._combine_handle is not None, (
            "MSCCL++ low-latency combine called before dispatch"
        )

        combined_x = self._moe_comm.combine(
            combine_input.hidden_states,
            self._combine_handle,
            apply_router_weights=combine_input.apply_router_weights,
        )

        self._combine_handle = None
        return combined_x


class MSCCLPPDispatcher(BaseDispatcher):
    """MSCCL++ expert-parallel dispatcher."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        mode: MSCCLPPMode = MSCCLPPMode.LATENCY,
        output_layout: MSCCLPPEPLayout = MSCCLPPEPLayout.EXPERT_MAJOR,
    ):
        super().__init__()

        if not isinstance(mode, MSCCLPPMode):
            raise TypeError("mode must be an MSCCLPPMode")
        if not mode.is_latency():
            raise NotImplementedError("MSCCL++ throughput mode is not implemented")
        if not isinstance(output_layout, MSCCLPPEPLayout):
            raise TypeError("output_layout must be an MSCCLPPEPLayout")
        if output_layout is MSCCLPPEPLayout.TOKEN_MAJOR:
            raise ValueError("MSCCL++ does not support token-major layout yet")

        self.mode = mode
        self.output_layout = output_layout
        self._dispatcher = _MSCCLPPDispatcherImplLowLatency(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            output_layout=output_layout,
        )

        self._active_dispatcher: Optional[_MSCCLPPDispatcherImplBase] = None

    def _resolve_dispatcher(self) -> _MSCCLPPDispatcherImplBase:
        return self._dispatcher

    @staticmethod
    def clear_shared_resources() -> None:
        """Release cached communicators after all dispatchers are no longer used."""
        _MSCCLPPDispatcherImplLowLatency._shared_resources.clear()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> MSCCLPPDispatchOutputBase:
        if self._active_dispatcher is not None:
            raise RuntimeError("MSCCLPPDispatcher.dispatch called before combine")

        dispatcher = self._resolve_dispatcher()
        dispatch_output = dispatcher.dispatch(hidden_states, topk_output)
        self._active_dispatcher = dispatcher
        return dispatch_output

    def combine(self, combine_input: MSCCLPPCombineInputBase) -> torch.Tensor:
        dispatcher = self._active_dispatcher
        if dispatcher is None:
            raise RuntimeError("MSCCLPPDispatcher.combine called before dispatch")

        combined_x = dispatcher.combine(combine_input)
        self._active_dispatcher = None
        return combined_x

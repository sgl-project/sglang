"""MSCCL++ expert-parallel dispatcher implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from .utils import (
    MSCCLPPCombineInputBase,
    MSCCLPPDispatchOutputBase,
    MSCCLPPExpertMajorLLDispatchOutput,
    MSCCLPPOutputLayout,
    MSCCLPPRankMajorLLDispatchOutput,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


def _resolve_ll_num_blocks(
    output_layout: MSCCLPPOutputLayout, overlap_enabled: bool
) -> Optional[tuple[int, int]]:
    if output_layout == MSCCLPPOutputLayout.RANK_MAJOR and overlap_enabled:
        # The shared-expert MLP runs concurrently on another stream. Keep the
        # full dispatch grid, but leave SM headroom during combine for that work.
        return 130, 32
    return None


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
      :class:`MSCCLPPLLDispatchOutput`. Triton uses padded expert-major output;
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
                MSCCLPPOutputLayout,
                Optional[tuple[int, int]],
            ],
            tuple[object, object, Optional[torch.Tensor], int],
        ]
    ] = {}

    @staticmethod
    def _resolve_cuda_graph_caps(
        cuda_graph_bs: Sequence[int] | None,
        cuda_graph_max_bs: int | None,
        *,
        num_tokens_per_bs: int = 1,
        attn_tp_size: int = 1,
    ) -> tuple[int, ...]:
        """Convert request buckets to token rows per rank after attention TP."""
        if num_tokens_per_bs <= 0 or attn_tp_size <= 0:
            raise ValueError("num_tokens_per_bs and attn_tp_size must be positive")
        batch_sizes = {int(value) for value in cuda_graph_bs or () if int(value) > 0}
        if cuda_graph_max_bs is not None and cuda_graph_max_bs > 0:
            batch_sizes.add(int(cuda_graph_max_bs))
        return tuple(
            sorted(
                {
                    (batch_size * num_tokens_per_bs + attn_tp_size - 1) // attn_tp_size
                    for batch_size in batch_sizes
                }
            )
        )

    @staticmethod
    def _resolve_prefill_capacity(
        chunked_prefill_size: int,
        ep_size: int,
        configured_capacity: int,
        *,
        dp_size: int = 1,
        dp_attention_enabled: bool = False,
    ) -> int:
        """Resolve the per-rank prefill capacity from the global chunk bound."""
        if ep_size <= 0 or configured_capacity <= 0 or dp_size <= 0:
            raise ValueError(
                "ep_size, dp_size, and configured_capacity must be positive"
            )
        global_chunked_prefill_size = (
            chunked_prefill_size * dp_size
            if dp_attention_enabled and chunked_prefill_size > 0
            else chunked_prefill_size
        )
        chunked_capacity = (
            (global_chunked_prefill_size + ep_size - 1) // ep_size
            if global_chunked_prefill_size > 0
            else 0
        )
        return max(configured_capacity, chunked_capacity)

    @staticmethod
    def _resolve_shared_runtime_capacity(
        prefill_capacity: int,
        decode_capacities: Sequence[int],
    ) -> int:
        """Return the allocation capacity shared by prefill and graph buckets."""
        if prefill_capacity <= 0:
            raise ValueError("prefill_capacity must be positive")
        return max((prefill_capacity, *(int(value) for value in decode_capacities)))

    @classmethod
    def _resolve_runtime_cuda_graph_caps(cls, attn_tp_size: int) -> tuple[int, ...]:
        from sglang.srt.runtime_context import (
            get_exec,
            get_spec,
            max_speculative_num_draft_tokens,
        )

        cuda_graph_config = get_exec().graph.cuda_graph_config
        if cuda_graph_config is None:
            return ()

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

        return cls._resolve_cuda_graph_caps(
            cuda_graph_config.decode.bs,
            cuda_graph_config.decode.max_bs,
            num_tokens_per_bs=num_tokens_per_bs,
            attn_tp_size=attn_tp_size,
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
        output_layout: MSCCLPPOutputLayout,
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
        moe_comm = MoECommunicator(
            MoECommunicatorConfig(
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
                combine_mode=CombineMode.RANK_LOCAL_REDUCE,
                num_blocks=num_blocks,
            )
        )
        if not moe_comm.is_available():
            raise RuntimeError("MSCCL++ EP low-latency runtime is unavailable")

        if output_layout == MSCCLPPOutputLayout.RANK_MAJOR:
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

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        output_layout: MSCCLPPOutputLayout = MSCCLPPOutputLayout.EXPERT_MAJOR,
    ):
        if not isinstance(output_layout, MSCCLPPOutputLayout):
            raise TypeError("output_layout must be an MSCCLPPOutputLayout")
        from sglang.srt.runtime_context import get_parallel, get_schedule

        num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_MSCCLPP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

        parallel = get_parallel()
        self._attn_tp_size = int(parallel.attn_tp_size)
        self._prefill_capacity = self._resolve_prefill_capacity(
            int(get_schedule().chunked_prefill_size or 0),
            group.size(),
            num_max_dispatch_tokens_per_rank,
            dp_size=int(parallel.dp_size),
            dp_attention_enabled=bool(parallel.enable_dp_attention),
        )
        self._decode_caps = (
            self._resolve_runtime_cuda_graph_caps(self._attn_tp_size)
            if output_layout == MSCCLPPOutputLayout.RANK_MAJOR
            else ()
        )
        self._allocation_capacity = self._resolve_shared_runtime_capacity(
            self._prefill_capacity,
            self._decode_caps,
        )

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.output_layout = output_layout
        self.overlap_enabled = (
            output_layout == MSCCLPPOutputLayout.RANK_MAJOR
            and envs.SGLANG_MSCCLPP_LL_OVERLAP.get()
        )
        num_blocks = _resolve_ll_num_blocks(output_layout, self.overlap_enabled)

        # Allocating these capacity-scaled resources per layer would OOM large
        # MoE models, so LL implementations with identical geometry share them.
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
            num_blocks,
        )

        # The DispatchHandle produced by dispatch and consumed by combine
        # (carries topk_ids / topk_weights / scatter metadata). Reset after each
        # combine, analogous to DeepEP's ``self.handle``. Kept per-instance: MoE
        # layers run sequentially, so each layer's dispatch->combine pair owns
        # the shared communicator for the duration of its forward.
        self._combine_handle = None

    def _select_active_capacity(self, num_tokens: int) -> int:
        if num_tokens > self._allocation_capacity:
            raise RuntimeError(
                "MSCCL++ local token count exceeds the shared allocation: "
                f"local={num_tokens}, allocated={self._allocation_capacity}"
            )
        if self.output_layout != MSCCLPPOutputLayout.RANK_MAJOR:
            return self._allocation_capacity

        if torch.cuda.is_current_stream_capturing():
            for capacity in self._decode_caps:
                if num_tokens <= capacity:
                    return capacity
            return self._allocation_capacity

        from sglang.srt.layers.dp_attention import get_dp_global_num_tokens

        global_num_tokens = get_dp_global_num_tokens()
        if global_num_tokens is None:
            return self._allocation_capacity

        global_max = max((int(value) for value in global_num_tokens), default=0)
        active_capacity = max(
            1, (global_max + self._attn_tp_size - 1) // self._attn_tp_size
        )
        if num_tokens > active_capacity:
            raise RuntimeError(
                "MSCCL++ rank-major local token count exceeds the scheduler-global "
                "maximum after attention-TP scatter: "
                f"local={num_tokens}, per_rank_max={active_capacity}"
            )
        if active_capacity > self._allocation_capacity:
            raise RuntimeError(
                "MSCCL++ rank-major eager capacity exceeds the shared allocation: "
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
                if self.output_layout == MSCCLPPOutputLayout.RANK_MAJOR
                else None
            ),
        )
        self._combine_handle = handle

        hidden_states_scale = (
            None if dispatch_out.quant is None else dispatch_out.quant.block_scales
        )

        if self.output_layout == MSCCLPPOutputLayout.RANK_MAJOR:
            assert dispatch_out.topk_ids is not None
            assert dispatch_out.weights is not None
            assert dispatch_out.layout.num_tokens_per_rank is not None
            assert dispatch_out.combine_input_buffer is not None
            active_rows = self.num_ranks * active_capacity
            return MSCCLPPRankMajorLLDispatchOutput(
                hidden_states=dispatch_out.tokens[:active_rows],
                hidden_states_scale=hidden_states_scale,
                topk_output=StandardTopKOutput(
                    dispatch_out.weights[:active_rows],
                    dispatch_out.topk_ids[:active_rows],
                    topk_output.router_logits,
                ),
                expert_output_buffer=dispatch_out.combine_input_buffer,
            )

        masked_m = dispatch_out.layout.num_tokens_per_expert
        assert isinstance(masked_m, torch.Tensor)

        # Average tokens per expert (same hint DeepEP-LL passes to the masked
        # GEMM); ``world_size`` copies of each token are scattered across the
        # ``num_experts`` experts. Unused by the Triton runner but kept for
        # parity with the DeepEP-LL contract.
        expected_m = (
            hidden_states.shape[0] * self.num_ranks * self.router_topk
            + self.num_experts
            - 1
        ) // self.num_experts

        return MSCCLPPExpertMajorLLDispatchOutput(
            hidden_states=dispatch_out.tokens,
            hidden_states_scale=hidden_states_scale,
            masked_m=masked_m,
            expected_m=expected_m,
        )

    def combine(self, combine_input: MSCCLPPCombineInputBase) -> torch.Tensor:
        assert self._combine_handle is not None, (
            "MSCCL++ low-latency combine called before dispatch"
        )

        # The handle carries the layout-specific routing and scatter metadata.
        combined_x = self._moe_comm.combine(
            combine_input.hidden_states, self._combine_handle
        )

        self._combine_handle = None
        return combined_x


class MSCCLPPDispatcher(BaseDispatcher):
    """MSCCL++ low-latency expert-parallel dispatcher."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        output_layout: MSCCLPPOutputLayout = MSCCLPPOutputLayout.EXPERT_MAJOR,
    ):
        super().__init__()

        if not isinstance(output_layout, MSCCLPPOutputLayout):
            raise TypeError("output_layout must be an MSCCLPPOutputLayout")

        self.output_layout = output_layout
        self._low_latency_dispatcher = _MSCCLPPDispatcherImplLowLatency(
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
        return self._low_latency_dispatcher

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

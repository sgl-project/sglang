from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.token_dispatcher import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
    TorchDistributedCommBackend,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatcher,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.runtime_context import get_flags, get_parallel, get_schedule, get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

try:
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.comm import MoeAlltoAll, moe_a2a_get_workspace_size_per_rank
    from flashinfer.comm.mapping import Mapping
    from flashinfer.comm.mnnvl import MnnvlConfig

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    use_flashinfer = True
except ImportError:
    use_flashinfer = False

logger = logging.getLogger(__name__)

MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()

# FlashInfer keys MNNVL allocations by workspace size; aligned tail padding gives
# concurrently live paths distinct persistent workspaces without extra token work.
_WORKSPACE_NAMESPACE_ALIGNMENT = 128


def _max_tokens_per_scattered_source(
    dp_global_num_tokens: list[int], attn_tp_size: int
) -> int:
    assert attn_tp_size > 0
    max_dp_tokens = max(dp_global_num_tokens)
    return (max_dp_tokens + attn_tp_size - 1) // attn_tp_size


def _scattered_source_token_counts(
    dp_global_num_tokens: list[int], attn_tp_size: int
) -> list[int]:
    assert attn_tp_size > 0
    counts = []
    for num_tokens in dp_global_num_tokens:
        base, remainder = divmod(num_tokens, attn_tp_size)
        counts.extend(
            base + int(attn_tp_rank < remainder) for attn_tp_rank in range(attn_tp_size)
        )
    return counts


def _workspace_size_for_namespace(workspace_size: int, *, speculative: bool) -> int:
    slot = int(speculative)
    return workspace_size + slot * _WORKSPACE_NAMESPACE_ALIGNMENT


class FlashinferDispatchOutput(NamedTuple):
    """Flashinfer EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: StandardTopKOutput
    # Provide an output tensor to fused_moe so it writes directly to our buffer
    moe_output: Optional[torch.Tensor] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.FLASHINFER


assert isinstance(FlashinferDispatchOutput, DispatchOutput)


class FlashinferCombineInput(NamedTuple):
    """Flashinfer combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.FLASHINFER


assert isinstance(FlashinferCombineInput, CombineInput)


class FlashinferDispatcher(BaseDispatcher):
    """Main dispatcher class for Flashinfer A2A backend."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        num_experts: int = None,
        num_local_experts: int = None,  # Unused
        hidden_size: int = None,
        params_dtype: torch.dtype = None,  # Unused
        moe_runner_config=None,
    ):
        super().__init__()
        if not use_flashinfer:
            raise ImportError(
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )

        self.ep_size = group.size()
        self.ep_rank = group.rank()
        self.router_topk = router_topk
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        runner_backend = get_moe_runner_backend()
        self.invalid_token_expert_id = (
            -1
            if (
                runner_backend.is_deep_gemm()
                or runner_backend.is_flashinfer_trtllm()
                or runner_backend.is_flashinfer_trtllm_routed()
            )
            else self.num_experts
        )
        # TODO: Can other moe runners use payload_in_workspace too?
        self.payload_in_workspace = get_moe_runner_backend().is_flashinfer_cutlass()
        if moe_runner_config is None:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

            moe_runner_config = MoeRunnerConfig(
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                hidden_size=hidden_size,
                top_k=router_topk,
            )
        self.prefill_dispatcher = StandardDispatcher(moe_runner_config)

        # FlashInfer sizes the workspace from the maximum dispatched tokens per
        # EP rank. See FlashInfer's moe_a2a_get_workspace_size_per_rank(),
        # which reserves ep_size * max_num_tokens * payload bytes, and the C++
        # dispatch op's epSize * runtimeMaxTokensPerRank payload buffer.
        #
        # The workspace must fit both:
        #  (a) the fattest prefill batch (bounded by chunked_prefill_size), and
        #  (b) the largest decode batch (bounded by max_running_requests, which
        #      resolve_max_num_reqs caps at 4096 per DP worker).
        # max_running_requests is not yet resolved at model-construction time,
        # so we use 4096 as a floor to cover decode batches and _dummy_run
        # (which warms up at batch_size = req_to_token_pool.size).
        cps = get_schedule().chunked_prefill_size
        default_max_tokens = max(cps if cps and cps > 0 else 4096, 4096)
        configured_max_tokens = (
            envs.SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self.max_num_tokens = (
            configured_max_tokens
            if configured_max_tokens is not None
            else default_max_tokens
        )

        # Calculate workspace size. For eagle mode, use the larger workspace size since nextn layer will be unquantized.
        speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        if MOE_NVFP4_DISPATCH and not speculative_algo.is_eagle():
            total_dispatch_payload_size_per_token = (
                hidden_size // 2  # nvfp4 hidden states
                + hidden_size // 16  # fp8 scaling factors
                + self.router_topk * 4  # int32 topks ids
                + self.router_topk * 4  # float32 topk weights
            )
        else:
            total_dispatch_payload_size_per_token = (
                hidden_size * 2  # bf16 hidden states
                + self.router_topk * 4  # int32 topks ids
                + self.router_topk * 4  # float32 topk weights
            )
        combine_payload_size_per_token = hidden_size * 2  # bf16 hidden states
        self.workspace_size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.ep_size,
            max_num_tokens=self.max_num_tokens,
            total_dispatch_payload_size_per_token=total_dispatch_payload_size_per_token,
            combine_payload_size_per_token=combine_payload_size_per_token,
        )

        self.mapping = Mapping(
            rank=self.ep_rank,
            tp_size=self.ep_size,
            moe_ep_size=self.ep_size,
            world_size=self.ep_size,
            gpus_per_node=torch.cuda.device_count(),
            pp_size=1,
            cp_size=1,
        )
        mnnvl_config = MnnvlConfig(comm_backend=TorchDistributedCommBackend(group))
        is_speculative_model = get_flags().moe.speculative_context

        def make_moe_a2a() -> MoeAlltoAll:
            # Target and draft decode graphs can coexist; prefill/mixed extend use
            # AG+RS and do not lease MNNVL A2A workspaces.
            workspace_size = _workspace_size_for_namespace(
                self.workspace_size,
                speculative=is_speculative_model,
            )
            return MoeAlltoAll(
                mapping=self.mapping,
                max_num_tokens=self.max_num_tokens,
                top_k=self.router_topk,
                num_experts=self.num_experts,
                workspace_size_per_rank=workspace_size,
                mnnvl_config=mnnvl_config,
            )

        self.moe_a2a = make_moe_a2a()

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        self.prefill_dispatcher.set_quant_config(quant_config)

    def _dispatch_prefill_allgather(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:
        # Eager extend can overlap another stream, so use BF16 all-gatherv instead
        # of reusing pure-decode A2A signal state across streams.

        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer WideEP prefill AG requires BF16 hidden states, got "
                f"{hidden_states.dtype}."
            )
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError(
                "FlashInfer WideEP prefill AG requires materialized top-k "
                f"routing, got {type(topk_output).__name__}."
            )

        dp_global = get_dp_global_num_tokens()
        if dp_global is None:
            source_sizes = [hidden_states.shape[0]] * self.ep_size
        else:
            source_sizes = _scattered_source_token_counts(
                dp_global, get_parallel().attn_tp_size
            )
        if len(source_sizes) != self.ep_size:
            raise RuntimeError(
                "FlashInfer WideEP prefill AG source geometry does not match "
                f"EP: len(source_sizes)={len(source_sizes)}, ep_size={self.ep_size}."
            )
        if source_sizes[self.ep_rank] != hidden_states.shape[0]:
            raise RuntimeError(
                "FlashInfer WideEP prefill AG local source geometry mismatch: "
                f"source_sizes[{self.ep_rank}]={source_sizes[self.ep_rank]} != "
                f"hidden_states.shape[0]={hidden_states.shape[0]}."
            )

        topk_ids = topk_output.topk_ids.to(torch.int32)
        hidden_states, topk_ids, topk_weights = get_parallel().tp_group.all_gatherv(
            [hidden_states, topk_ids, topk_output.topk_weights],
            sizes=source_sizes,
        )
        self.prefill_source_sizes = source_sizes
        return self.prefill_dispatcher.dispatch(
            hidden_states,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
        )

    @debug_kernel_api
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> FlashinferDispatchOutput | StandardDispatchOutput:
        if get_is_extend_in_batch():
            return self._dispatch_prefill_allgather(hidden_states, topk_output)
        self.active_moe_a2a = self.moe_a2a
        # Block-wise FP8 runners quantize before GEMM, so keep dispatch/combine BF16;
        # FP4 retains its packed wire path keyed by input_global_scale.
        runner_backend = get_moe_runner_backend()
        weight_dtype = self.quant_config.get("weight_dtype")
        uses_bf16_fp8_payload = weight_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ) and (
            runner_backend.is_deep_gemm()
            or runner_backend.is_flashinfer_trtllm()
            or runner_backend.is_flashinfer_trtllm_routed()
        )
        if uses_bf16_fp8_payload and hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer A2A with an FP8 DeepGEMM/TRT-LLM Gen MoE runner "
                "requires BF16 dispatch and combine payloads, but received "
                f"{hidden_states.dtype}."
            )

        output_dtype = hidden_states.dtype
        x = hidden_states
        x_sf = None
        # FlashInfer dispatch requires materialized top-k IDs and weights.
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        # FlashInfer MoeAlltoAll's expert-ID ABI is int32. This dispatcher is
        # only selected for moe_a2a_backend="flashinfer".
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights

        global_scale = self.quant_config.get("input_global_scale", None)
        if global_scale is not None:
            if x.shape[0] > 0:
                x, x_sf = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
            else:
                x_col = x.shape[1]
                x = torch.zeros(0, x_col // 2, dtype=torch.uint8, device=x.device)
                x_sf = torch.zeros(0, x_col // 16, dtype=torch.uint8, device=x.device)

        payloads = []
        payloads.append(x)
        if x_sf is not None:
            payloads.append(x_sf)
            expert_id_payload_index = 2
        else:
            expert_id_payload_index = 1
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        # runtime_max_tokens_per_rank selection
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # MoeAlltoAll uses fixed-geometry buffers shaped
        # [ep_size, runtime_max_tokens_per_rank, ...], so every EP rank must pass
        # the SAME value. This code (Python) runs during eager forwards and during
        # CUDA-graph *capture*; on *replay* dispatch() is not re-executed and the
        # value baked at capture is reused. Two cases, both rank-invariant:
        #
        # Each EP source owns ceil(max(dp_global) / attn_tp_size) tokens; the
        # shared maximum keeps graph geometry rank-uniform (issue #30242).
        #
        # Case 2 — x.shape[0]: no per-rank DP list (dp_global absent or scalar).
        #   This is SP attention feeding EP (tokens are sequence-parallel scattered
        #   uniformly, so x.shape[0] is already identical on every EP rank), a
        #   single EP rank, or CUDA-graph capture of those. x.shape[0] is
        #   rank-invariant here, so it is both correct and right-sized.
        dp_global = get_dp_global_num_tokens()
        if dp_global is not None and len(dp_global) > 1:
            # Case 1
            attn_tp_size = get_parallel().attn_tp_size
            self.runtime_max_tokens_per_rank = _max_tokens_per_scattered_source(
                dp_global, attn_tp_size
            )
        else:
            # Case 2. Guard against the #30242 failure mode: DP attention must
            # never land here with ep_size > 1, because there x.shape[0] differs
            # across ranks and is NOT a safe fixed geometry. DP attention is
            # routed to Case 1 via require_mlp_tp_gather=True; reaching here with
            # DP attention on and ep_size > 1 means the DP all-gather was skipped
            # (e.g. SGLANG_SCHEDULER_SKIP_ALL_GATHER, unsupported) -> fail fast.
            assert not is_dp_attention_enabled() or self.ep_size == 1, (
                "FlashInfer A2A: DP attention reached the x.shape[0] fallback "
                f"with ep_size={self.ep_size} > 1 (dp_global={dp_global}); "
                "runtime_max_tokens_per_rank would not be rank-invariant."
            )
            self.runtime_max_tokens_per_rank = x.shape[0]

        # MoeAlltoAll does not resize its max_num_tokens workspace; reject larger
        # runtime geometry here before it becomes an illegal memory access.
        assert self.runtime_max_tokens_per_rank <= self.max_num_tokens, (
            "FlashInfer A2A runtime token geometry exceeds its fixed workspace: "
            f"runtime_max_tokens_per_rank={self.runtime_max_tokens_per_rank} > "
            f"max_num_tokens={self.max_num_tokens}. Increase "
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK to cover the "
            "largest mixed prefill and speculative-verify batch."
        )

        # The recv buffer reserves runtime_max_tokens_per_rank slots for THIS
        # rank, so it must cover this rank's own tokens. This holds in both cases
        # (Case 1: ceil(max(dp_global) / attn_tp_size) covers every token-scatter
        # shard; Case 2: exactly x.shape[0]),
        # so a violation signals a sizing/plumbing bug (e.g. an un-adjusted spec
        # count) rather than a benign case.
        assert self.runtime_max_tokens_per_rank >= x.shape[0], (
            f"runtime_max_tokens_per_rank={self.runtime_max_tokens_per_rank} < "
            f"x.shape[0]={x.shape[0]}: MoeAlltoAll recv buffer would overflow."
        )

        # Passing topk_ids + invalid_token_expert_id triggers the sanitize step
        # inside moe_a2a. The recv buffer has shape
        # [ep_size, max_tokens_per_rank, ...], so any rank below max leaves
        # padding slots whose expert_id would otherwise route to a real expert
        # and waste downstream MoE compute. Sanitizing the padding to a
        # sentinel id is structural, not optional.
        recv_tensors = self.active_moe_a2a.dispatch(
            topk_ids,
            payloads,
            self.runtime_max_tokens_per_rank,
            invalid_token_expert_id=self.invalid_token_expert_id,
            expert_id_payload_index=expert_id_payload_index,
        )
        if x_sf is not None:
            x_recv, x_sf_recv, topk_ids_recv, topk_weights_recv = recv_tensors
            x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
            # TODO: fuse interleave into cutlass moe
            if get_moe_runner_backend().is_flashinfer_cutlass():
                x_sf = nvfp4_block_scale_interleave(x_sf)
        else:
            x_recv, topk_ids_recv, topk_weights_recv = recv_tensors
        x = x_recv.view(-1, x_recv.shape[-1])
        topk_ids = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])

        # Provide an output tensor to fused_moe so it writes directly to our buffer
        moe_output = None
        if self.payload_in_workspace:
            moe_output = self.active_moe_a2a.get_combine_payload_tensor_in_workspace(
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype
            ).view(-1, self.hidden_size)
        return FlashinferDispatchOutput(
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
            moe_output,
        )

    @debug_kernel_api
    def combine(
        self, combine_input: FlashinferCombineInput | StandardCombineInput
    ) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        if combine_input.format == CombineInputFormat.STANDARD:
            if hidden_states.dtype != torch.bfloat16:
                raise TypeError(
                    "FlashInfer WideEP prefill RS requires BF16 expert output, "
                    f"got {hidden_states.dtype}."
                )
            source_sizes = self.prefill_source_sizes
            hidden_states = get_parallel().tp_group.reduce_scatterv(
                hidden_states, sizes=source_sizes
            )
            del self.prefill_source_sizes
            return hidden_states

        weight_dtype = self.quant_config.get("weight_dtype")
        runner_backend = get_moe_runner_backend()
        if (
            weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and (
                runner_backend.is_deep_gemm()
                or runner_backend.is_flashinfer_trtllm()
                or runner_backend.is_flashinfer_trtllm_routed()
            )
            and hidden_states.dtype != torch.bfloat16
        ):
            raise TypeError(
                "FlashInfer A2A FP8 MoE combine payload must be BF16, but "
                f"received {hidden_states.dtype}."
            )
        output_hidden_size = hidden_states.shape[-1]
        hidden_states = self.active_moe_a2a.combine(
            hidden_states.view(
                self.ep_size, self.runtime_max_tokens_per_rank, output_hidden_size
            ),
            self.runtime_max_tokens_per_rank,
            payload_in_workspace=self.payload_in_workspace,
        )

        del self.runtime_max_tokens_per_rank
        del self.active_moe_a2a
        return hidden_states

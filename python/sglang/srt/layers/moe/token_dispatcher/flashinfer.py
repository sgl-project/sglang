from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
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
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput
from sglang.srt.layers.moe.utils import get_moe_runner_backend
from sglang.srt.runtime_context import get_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_int_env_var

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
        self.invalid_token_expert_id = (
            -1
            if get_moe_runner_backend().is_flashinfer_trtllm_routed()
            else self.num_experts
        )
        # TODO: Can other moe runners use payload_in_workspace too?
        self.payload_in_workspace = get_moe_runner_backend().is_flashinfer_cutlass()

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
        cps = get_server_args().chunked_prefill_size
        default_max_tokens = max(cps if cps and cps > 0 else 4096, 4096)
        self.max_num_tokens = get_int_env_var(
            "SGLANG_FLASHINFER_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
            default_max_tokens,
        )

        # Calculate workspace size. For eagle mode, use the larger workspace size since nextn layer will be unquantized.
        speculative_algo = SpeculativeAlgorithm.from_string(
            get_server_args().speculative_algorithm
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
        self.moe_a2a = MoeAlltoAll(
            mapping=self.mapping,
            max_num_tokens=self.max_num_tokens,
            top_k=self.router_topk,
            num_experts=self.num_experts,
            workspace_size_per_rank=self.workspace_size,
            mnnvl_config=MnnvlConfig(comm_backend=TorchDistributedCommBackend(group)),
        )

    @debug_kernel_api
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> FlashinferDispatchOutput:
        output_dtype = hidden_states.dtype
        x = hidden_states
        x_sf = None
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
        # Case 1 — max(dp_global): DP attention feeding EP. The scheduler
        #   all-gathers per-DP-rank token counts into dp_global (length dp_size,
        #   identical on every rank), which differ across ranks, so we must take
        #   the max. FlashInfer A2A forces require_mlp_tp_gather=True (see
        #   require_mlp_tp_gather()), so: eager reads the live list; capture sees
        #   [num_tokens] * dp_size (uniform capture bs) and bakes max() == the
        #   bucket; replay reuses that baked value and every rank replays the same
        #   bucket because the decode graph runner sizes it from the cross-rank
        #   max. Without this, per-rank buckets could diverge -> geometry mismatch
        #   -> illegal memory access (issue #30242).
        #
        # Case 2 — x.shape[0]: no per-rank DP list (dp_global absent or scalar).
        #   This is SP attention feeding EP (tokens are sequence-parallel scattered
        #   uniformly, so x.shape[0] is already identical on every EP rank), a
        #   single EP rank, or CUDA-graph capture of those. x.shape[0] is
        #   rank-invariant here, so it is both correct and right-sized.
        dp_global = get_dp_global_num_tokens()
        if dp_global is not None and len(dp_global) > 1:
            # Case 1
            self.runtime_max_tokens_per_rank = max(dp_global)
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

        # The recv buffer reserves runtime_max_tokens_per_rank slots for THIS
        # rank, so it must cover this rank's own tokens. This holds in both cases
        # (Case 1: max(dp_global) >= the local count; Case 2: exactly x.shape[0]),
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
        recv_tensors = self.moe_a2a.dispatch(
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
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                self.runtime_max_tokens_per_rank, self.hidden_size, output_dtype
            ).view(-1, self.hidden_size)
        return FlashinferDispatchOutput(
            x,
            x_sf,
            StandardTopKOutput(topk_weights, topk_ids, topk_output.router_logits),
            moe_output,
        )

    @debug_kernel_api
    def combine(self, combine_input: FlashinferCombineInput) -> torch.Tensor:
        hidden_states = combine_input.hidden_states
        output_hidden_size = hidden_states.shape[-1]
        hidden_states = self.moe_a2a.combine(
            hidden_states.view(
                self.ep_size, self.runtime_max_tokens_per_rank, output_hidden_size
            ),
            self.runtime_max_tokens_per_rank,
            payload_in_workspace=self.payload_in_workspace,
        )

        del self.runtime_max_tokens_per_rank
        return hidden_states

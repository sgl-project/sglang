from __future__ import annotations

import enum
import logging
from typing import NamedTuple, Optional

import torch

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.arg_groups.overrides import flashinfer_a2a_max_dispatch_tokens_per_rank
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
from sglang.srt.layers.moe.utils import (
    FlashinferA2ADispatchType,
    get_flashinfer_a2a_dispatch_type,
)
from sglang.srt.runtime_context import (
    get_flags,
    get_forward,
    get_parallel,
    max_prefill_buffer_tokens,
)

try:
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpCommParams,
        NVLinkOneSidedAlltoAll,
        NVLinkOneSidedConfig,
        create_communication,
    )

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    use_flashinfer = True
except ImportError:
    use_flashinfer = False

logger = logging.getLogger(__name__)


# FlashInfer shares one workspace among communication instances with the same
# geometry, and those instances must not interleave dispatch/combine rounds.
# Padding the token capacity by the namespace gives concurrently live paths
# distinct persistent workspaces without extra token work.
class _WorkspaceNamespace(enum.IntEnum):
    DECODE = 0
    SPECULATIVE = 1
    SP_PREFILL = 2


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


def _max_tokens_for_namespace(
    max_num_tokens: int,
    *,
    namespace: Optional[_WorkspaceNamespace] = None,
    speculative: Optional[bool] = None,
) -> int:
    if namespace is not None and speculative is not None:
        raise ValueError("Specify either namespace or speculative, not both.")
    if namespace is None:
        namespace = (
            _WorkspaceNamespace.SPECULATIVE
            if speculative
            else _WorkspaceNamespace.DECODE
        )
    return max_num_tokens + int(namespace)


class FlashinferDispatchOutput(NamedTuple):
    """Flashinfer EP dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: StandardTopKOutput
    # Provide an output tensor to fused_moe so it writes directly to our buffer
    moe_output: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None

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
        enable_sp_prefill_a2a: bool = False,
    ):
        super().__init__()
        if not use_flashinfer:
            raise ImportError(
                "Flashinfer is not installed or does not support A2A. "
                "Please install the appropriate version of Flashinfer."
            )

        self.runner_backend = (
            moe_runner_config.runner_backend if moe_runner_config is not None else None
        )
        self.ep_size = group.size()
        self.ep_rank = group.rank()
        self.router_topk = router_topk
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        runner_backend = self.runner_backend
        self.invalid_token_expert_id = (
            -1
            if (
                runner_backend is not None
                and (
                    runner_backend.is_deep_gemm()
                    or runner_backend.is_flashinfer_trtllm()
                    or runner_backend.is_flashinfer_trtllm_routed()
                )
            )
            else self.num_experts
        )
        # TODO: Can other moe runners use payload_in_workspace too?
        self.payload_in_workspace = (
            self.runner_backend is not None
            and self.runner_backend.is_flashinfer_cutlass()
        )
        self.dispatch_type = get_flashinfer_a2a_dispatch_type()
        if moe_runner_config is None:
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

            moe_runner_config = MoeRunnerConfig(
                num_experts=num_experts,
                num_local_experts=num_local_experts,
                num_fused_shared_experts=0,
                hidden_size=hidden_size,
                top_k=router_topk,
            )
        self.prefill_dispatcher = StandardDispatcher(moe_runner_config)

        # FlashInfer sizes the workspace from the maximum dispatched tokens per
        # EP rank: its dispatch receive buffer is ep_size * max_tokens_per_rank rows.
        #
        # Cover both the largest eager prefill chunk (including PP dynamic
        # chunking) and the largest decode batch using upstream's shared bound.
        self.max_num_tokens = flashinfer_a2a_max_dispatch_tokens_per_rank(
            max_prefill_buffer_tokens()
        )
        bootstrap = BootstrapConfig(
            world_size=self.ep_size, rank=self.ep_rank, process_group=group
        )
        a2a_config = NVLinkOneSidedConfig(
            comm_backend=TorchDistributedCommBackend(group)
        )
        is_speculative_model = get_flags().moe.speculative_context

        def make_moe_a2a(namespace: _WorkspaceNamespace) -> NVLinkOneSidedAlltoAll:
            params = MoEEpCommParams(
                num_experts=self.num_experts,
                top_k=self.router_topk,
                max_tokens_per_rank=_max_tokens_for_namespace(
                    self.max_num_tokens, namespace=namespace
                ),
                hidden_size=self.hidden_size,
                # No dispatch_format: BF16 rows bound every quantized format and
                # cover layers whose dispatch falls back to BF16 (draft, EAGLE).
                dtype=torch.bfloat16,
                invalid_expert_id=self.invalid_token_expert_id,
            )
            # FlashInfer's host-side workspace metadata must stay on the CPU even
            # when layers are built under a CUDA default-device context.
            with torch.device("cpu"):
                return create_communication(bootstrap, params, backend=a2a_config)

        decode_namespace = (
            _WorkspaceNamespace.SPECULATIVE
            if is_speculative_model
            else _WorkspaceNamespace.DECODE
        )
        self.decode_a2a_namespace = decode_namespace
        self.moe_a2a = make_moe_a2a(decode_namespace)
        self.sp_prefill_moe_a2a = (
            make_moe_a2a(_WorkspaceNamespace.SP_PREFILL)
            if enable_sp_prefill_a2a
            else None
        )

    def _start_a2a(self, *, sp_prefill: bool) -> None:
        if hasattr(self, "active_moe_a2a"):
            raise RuntimeError(
                "FlashInfer A2A dispatch is already active; combine or abort it "
                "before starting another dispatch."
            )
        if sp_prefill and self.sp_prefill_moe_a2a is None:
            raise RuntimeError(
                "FlashInfer SP-prefill A2A is not enabled for this dispatcher."
            )
        self.active_moe_a2a = self.sp_prefill_moe_a2a if sp_prefill else self.moe_a2a
        self.active_a2a_namespace = (
            _WorkspaceNamespace.SP_PREFILL if sp_prefill else self.decode_a2a_namespace
        )

    def _clear_active_a2a(self) -> None:
        for attr in ("active_moe_a2a", "active_a2a_namespace"):
            if hasattr(self, attr):
                delattr(self, attr)

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

    def _effective_dispatch_type(self) -> FlashinferA2ADispatchType:
        if self.dispatch_type == FlashinferA2ADispatchType.NVFP4:
            global_scale = (self.quant_config or {}).get("input_global_scale", None)
            if global_scale is None:
                return FlashinferA2ADispatchType.BF16
        elif self.dispatch_type == FlashinferA2ADispatchType.MXFP8:
            # Draft/NextN or mixed layers may not be MXFP8 even when the
            # process-wide default is MXFP8.
            if not (self.quant_config or {}).get("use_mxfp8", False):
                return FlashinferA2ADispatchType.BF16
        return self.dispatch_type

    @debug_kernel_api
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> FlashinferDispatchOutput | StandardDispatchOutput:
        is_extend = get_is_extend_in_batch()
        is_sp_prefill = is_extend and get_forward().sp_active
        if is_extend and not is_sp_prefill:
            return self._dispatch_prefill_allgather(hidden_states, topk_output)
        if is_sp_prefill and hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer SP prefill A2A requires BF16 hidden states, got "
                f"{hidden_states.dtype}."
            )
        # Block-wise FP8 runners quantize before GEMM, so keep dispatch/combine BF16;
        # FP4 retains its packed wire path keyed by input_global_scale.
        runner_backend = self.runner_backend
        weight_dtype = self.quant_config.get("weight_dtype")
        uses_bf16_fp8_payload = weight_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ) and (
            runner_backend is not None
            and (
                runner_backend.is_deep_gemm()
                or runner_backend.is_flashinfer_trtllm()
                or runner_backend.is_flashinfer_trtllm_routed()
            )
        )
        if uses_bf16_fp8_payload and hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "FlashInfer A2A with an FP8 DeepGEMM/TRT-LLM Gen MoE runner "
                "requires BF16 dispatch and combine payloads, but received "
                f"{hidden_states.dtype}."
            )

        output_dtype = hidden_states.dtype
        dispatch_type = self._effective_dispatch_type()
        x = hidden_states
        x_sf = None
        # FlashInfer dispatch requires materialized top-k IDs and weights.
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        # FlashInfer NVLinkOneSidedAlltoAll's expert-ID ABI is int32. This
        # dispatcher is only selected for moe_a2a_backend="flashinfer".
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights

        if dispatch_type == FlashinferA2ADispatchType.NVFP4:
            global_scale = (self.quant_config or {}).get("input_global_scale", None)
            assert global_scale is not None
            if x.shape[0] > 0:
                x, x_sf = fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
            else:
                x = torch.zeros(
                    0, self.hidden_size // 2, dtype=torch.uint8, device=x.device
                )
                x_sf = torch.zeros(
                    0, self.hidden_size // 16, dtype=torch.uint8, device=x.device
                )
        elif dispatch_type == FlashinferA2ADispatchType.MXFP8:
            if x.shape[0] > 0:
                from flashinfer import mxfp8_quantize

                x, x_sf = mxfp8_quantize(x, False)
                x_sf = x_sf.view(torch.uint8).reshape(
                    x.shape[0], self.hidden_size // 32
                )
            else:
                x = torch.zeros(
                    0,
                    self.hidden_size,
                    dtype=torch.float8_e4m3fn,
                    device=x.device,
                )
                x_sf = torch.zeros(
                    0, self.hidden_size // 32, dtype=torch.uint8, device=x.device
                )

        # runtime_max_tokens_per_rank selection
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NVLinkOneSidedAlltoAll uses fixed-geometry buffers shaped
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
            runtime_max_tokens_per_rank = _max_tokens_per_scattered_source(
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
            runtime_max_tokens_per_rank = x.shape[0]

        # The FlashInfer workspace is sized once for max_num_tokens; reject larger
        # runtime geometry here with a sizing hint rather than FlashInfer's error.
        assert runtime_max_tokens_per_rank <= self.max_num_tokens, (
            "FlashInfer A2A runtime token geometry exceeds its fixed workspace: "
            f"runtime_max_tokens_per_rank={runtime_max_tokens_per_rank} > "
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
        assert runtime_max_tokens_per_rank >= x.shape[0], (
            f"runtime_max_tokens_per_rank={runtime_max_tokens_per_rank} < "
            f"x.shape[0]={x.shape[0]}: A2A recv buffer would overflow."
        )

        # The recv buffer has shape [ep_size * runtime_max_tokens_per_rank, ...],
        # so any rank below max leaves padding slots; FlashInfer sanitizes their
        # expert ids to MoEEpCommParams.invalid_expert_id so that downstream MoE
        # compute skips them.
        self._start_a2a(sp_prefill=is_sp_prefill)
        try:
            received = self.active_moe_a2a.dispatch(
                x,
                topk_ids,
                topk_weights,
                hidden_states_scale=x_sf,
                max_tokens_per_rank=runtime_max_tokens_per_rank,
            )
            x = received.hidden_states
            x_sf = received.hidden_states_scale
            # TODO: fuse interleave into cutlass moe
            if (
                dispatch_type == FlashinferA2ADispatchType.NVFP4
                and self.runner_backend is not None
                and self.runner_backend.is_flashinfer_cutlass()
            ):
                x_sf = nvfp4_block_scale_interleave(x_sf)

            # Provide an output tensor to fused_moe so it writes directly to our buffer
            moe_output = None
            if self.payload_in_workspace:
                moe_output = self.active_moe_a2a.get_combine_input_buffer(output_dtype)
            return FlashinferDispatchOutput(
                x,
                x_sf,
                StandardTopKOutput(
                    received.topk_weights, received.topk_ids, topk_output.router_logits
                ),
                moe_output,
                output_dtype,
            )
        except Exception:
            self._clear_active_a2a()
            raise

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

        if not hasattr(self, "active_moe_a2a"):
            raise RuntimeError(
                "FlashInfer A2A combine requires a matching active dispatch."
            )

        weight_dtype = self.quant_config.get("weight_dtype")
        runner_backend = self.runner_backend
        if (
            weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and (
                runner_backend is not None
                and (
                    runner_backend.is_deep_gemm()
                    or runner_backend.is_flashinfer_trtllm()
                    or runner_backend.is_flashinfer_trtllm_routed()
                )
            )
            and hidden_states.dtype != torch.bfloat16
        ):
            raise TypeError(
                "FlashInfer A2A FP8 MoE combine payload must be BF16, but "
                f"received {hidden_states.dtype}."
            )
        try:
            # FlashInfer detects a payload already in its combine input buffer.
            return self.active_moe_a2a.combine(hidden_states)
        finally:
            self._clear_active_a2a()

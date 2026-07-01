from __future__ import annotations

import logging
from enum import Enum, auto
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPV2OutputDtype,
    DeepEPV2RunnerCapability,
    get_deepep_v2_runner_capability,
)

logger = logging.getLogger(__name__)

_SCALE_BLOCK_SIZE = 128
_deepep_v2_import_error: Optional[BaseException] = None
_fp8_quant_import_error: Optional[BaseException] = None
sglang_per_token_group_quant_fp8 = None

try:
    from deep_ep import ElasticBuffer

    use_deepep_v2 = True
except (ImportError, OSError) as exc:
    use_deepep_v2 = False
    _deepep_v2_import_error = exc

if use_deepep_v2:
    try:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
    except (ImportError, OSError) as exc:
        _fp8_quant_import_error = exc


class DeepEPV2DispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]
    psum_num_recv_tokens_per_expert: Optional[torch.Tensor] = None
    is_expanded: bool = False
    hidden_states_scale_tma_aligned: bool = False
    use_masked_gemm: bool = False
    expected_m: int = 0
    masked_max_m: int = 0
    total_expanded: int = 0
    expert_alignment: int = 128

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_V2


class DeepEPV2CombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: Optional[torch.Tensor]
    topk_weights: Optional[torch.Tensor]

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_V2


assert isinstance(DeepEPV2DispatchOutput, DispatchOutput)
assert isinstance(DeepEPV2CombineInput, CombineInput)


def _raise_deepep_v2_import_error() -> None:
    detail = (
        f" Original import error: {_deepep_v2_import_error}"
        if _deepep_v2_import_error is not None
        else ""
    )
    raise ImportError(
        "DeepEP v2 (ElasticBuffer) is not available. Install DeepEP v2 from "
        "https://github.com/deepseek-ai/DeepEP." + detail
    )


def _ensure_deepep_v2_available() -> None:
    if not use_deepep_v2:
        _raise_deepep_v2_import_error()


def _ensure_fp8_quant_available() -> None:
    _ensure_deepep_v2_available()
    if sglang_per_token_group_quant_fp8 is None:
        detail = (
            f" Original import error: {_fp8_quant_import_error}"
            if _fp8_quant_import_error is not None
            else ""
        )
        raise ImportError(
            "DeepEP v2 FP8 dispatch requires the SGLang FP8 quantization kernel."
            + detail
        )


def _get_allow_hybrid_mode() -> bool:
    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except ValueError:
        # Synthetic/unit tests can instantiate the dispatcher without ServerArgs.
        return envs.SGLANG_DEEPEP_V2_ALLOW_HYBRID_MODE.get()

    deepep_v2_mode = getattr(server_args, "deepep_v2_mode", None)
    if deepep_v2_mode is None:
        return envs.SGLANG_DEEPEP_V2_ALLOW_HYBRID_MODE.get()
    return deepep_v2_mode == "hybrid"


def _quantize_for_deepep_v2_dispatch(
    hidden_states: torch.Tensor, capability: DeepEPV2RunnerCapability
):
    _ensure_fp8_quant_available()
    return sglang_per_token_group_quant_fp8(
        hidden_states,
        _SCALE_BLOCK_SIZE,
        column_major_scales=capability.fp8_scale_tma_aligned,
        scale_tma_aligned=capability.fp8_scale_tma_aligned,
        scale_ue8m0=capability.fp8_scale_ue8m0,
    )


class DeepEPV2Buffer:
    _buffer: Optional[ElasticBuffer] = None
    _buffer_key: Optional[Tuple] = None

    @classmethod
    def get_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        router_topk: int,
        num_max_dispatch_tokens_per_rank: int,
        use_fp8_dispatch: bool,
    ) -> ElasticBuffer:
        _ensure_deepep_v2_available()

        allow_hybrid_mode = _get_allow_hybrid_mode()
        key = (
            id(group),
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
            dist.get_world_size(group),
        )
        if cls._buffer is not None and cls._buffer_key == key:
            return cls._buffer

        if cls._buffer is not None:
            cls.destroy()

        cls._buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=router_topk,
            use_fp8_dispatch=use_fp8_dispatch,
            allow_hybrid_mode=allow_hybrid_mode,
            sl_idx=0,
            prefer_overlap_with_compute=False,
        )
        cls._buffer_key = key
        logger.info(
            "Initialized DeepEP v2 ElasticBuffer: world_size=%s hidden_size=%s "
            "num_topk=%s max_dispatch_tokens_per_rank=%s use_fp8_dispatch=%s "
            "allow_hybrid_mode=%s num_bytes=%s",
            dist.get_world_size(group),
            hidden_size,
            router_topk,
            num_max_dispatch_tokens_per_rank,
            use_fp8_dispatch,
            allow_hybrid_mode,
            cls._buffer.num_bytes,
        )
        return cls._buffer

    @classmethod
    def destroy(cls) -> None:
        cls._buffer = None
        cls._buffer_key = None


class _DeepEPV2Impl:
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        capability: DeepEPV2RunnerCapability,
        num_max_dispatch_tokens_per_rank: int,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.capability = capability
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.rank = dist.get_rank(group)
        self._handle = None

    def set_runner_capability(self, capability: DeepEPV2RunnerCapability) -> None:
        if self.capability != capability:
            self._destroy_handle()
            self.capability = capability

    def _uses_fp8_dispatch_output(self) -> bool:
        return self.capability.output_dtype == DeepEPV2OutputDtype.FP8

    def _destroy_handle(self) -> None:
        self._handle = None

    def _get_buffer(self) -> ElasticBuffer:
        return DeepEPV2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_max_dispatch_tokens_per_rank,
            self._uses_fp8_dispatch_output(),
        )

    def _validate_common(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor
    ) -> None:
        if hidden_states.shape[0] > self.num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"DeepEP v2 dispatch input exceeds the per-rank buffer capacity "
                f"{self.num_max_dispatch_tokens_per_rank}, got {hidden_states.shape[0]}. "
                "Increase SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"DeepEP v2 hidden size mismatch: expected {self.hidden_size}, "
                f"got {hidden_states.shape[1]}"
            )
        if (
            self._uses_fp8_dispatch_output()
            and self.hidden_size % _SCALE_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "DeepEP v2 FP8 dispatch requires hidden_size multiple of "
                f"{_SCALE_BLOCK_SIZE}, got {self.hidden_size}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                f"DeepEP v2 topk mismatch: expected {self.router_topk}, "
                f"got {topk_ids.shape[1]}"
            )

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        _ensure_deepep_v2_available()
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)
        self._validate_common(hidden_states, topk_ids)
        # DeepEP v2's native expanded layout is profitable for decode-like DeepGEMM
        # FP8 workloads but regresses prefill-like ones, so layout is chosen by
        # inference PHASE, independently of the comm mode (direct/hybrid is a topology
        # knob fixed at server init): decode (non-extend) -> native expanded layout;
        # prefill/extend -> non-expanded contiguous layout. This decouples the
        # masked-GEMM + CUDA-graph decode fast path from the comm mode, so it is
        # available under multi-node `hybrid` too.
        use_expand_layout = (
            self.capability.use_expanded_layout and not get_is_extend_in_batch()
        )
        # masked GEMM is built from the expanded layout (expand_to_masked_slab), so
        # masked <=> expanded. async dispatch (cpu_sync=False) gives a static
        # capturable recv shape; the masked GEMM bounds compute by masked_m, so the
        # full (safe) cap costs no extra GEMM.
        use_masked = use_expand_layout

        # ElasticBuffer requires >=1 token per rank on the non-masked (contiguous /
        # extend) path: DeepEP's own ElasticBuffer test pads every rank to
        # `max(1, num_tokens)` (tests/elastic/test_ep.py). An idle DP rank with 0
        # tokens never fires the dispatch notify / scale-up-reduction warps, so no
        # rank's recv count becomes "ready" and the do_cpu_sync CPU readback times
        # out ("Dispatch CPU wait", buffer.hpp:1032). Pad an empty local batch to a
        # single dummy token (routed to local expert 0); the contiguous slice in
        # dispatch_b yields 0 real rows and combine_b drops it back to an empty
        # output. The masked decode path tolerates empty (do_cpu_sync=False), so it
        # is left untouched.
        self._pad_empty_combine = (not use_masked) and hidden_states.shape[0] == 0
        if self._pad_empty_combine:
            hidden_states = hidden_states.new_zeros((1, hidden_states.shape[-1]))
            # A token's top-k experts must be DISTINCT valid ids: duplicates (e.g.
            # all-zero -> expert 0 repeated) fault the dispatch kernel. Route the
            # dummy to experts [0, 1, ..., topk-1] with zero weights so it
            # contributes nothing even before combine_b slices it off.
            topk_ids = torch.arange(
                topk_ids.shape[-1], dtype=topk_ids.dtype, device=topk_ids.device
            ).unsqueeze(0)
            topk_weights = topk_weights.new_zeros((1, topk_weights.shape[-1]))

        if self._uses_fp8_dispatch_output():
            _ensure_fp8_quant_available()
            if use_masked:
                # _run_masked_gemm consumes plain per-token-group fp32 scales and
                # does its own e8m0/tma-major alignment, so dispatch a plain
                # row-major scale (no col-major, no tma, no e8m0 pre-pack).
                dispatch_x = sglang_per_token_group_quant_fp8(
                    hidden_states,
                    _SCALE_BLOCK_SIZE,
                    column_major_scales=False,
                    scale_tma_aligned=False,
                    scale_ue8m0=False,
                )
                use_tma_aligned_col_major_sf = False
            else:
                dispatch_x = _quantize_for_deepep_v2_dispatch(
                    hidden_states, self.capability
                )
                use_tma_aligned_col_major_sf = self.capability.fp8_scale_tma_aligned
        else:
            dispatch_x = hidden_states
            use_tma_aligned_col_major_sf = False

        # num_max_tokens_per_rank is a COLLECTIVE dispatch arg (ElasticBuffer
        # requires the same value on all ranks). Keep it at the fixed buffer cap
        # (class-level, cross-rank-consistent), matching DeepEP LL which uses a
        # fixed _num_max_dispatch_tokens_per_rank rather than a per-forward token
        # count. Do NOT derive it from the local hidden_states.shape[0]: under
        # ragged DP load (or TP attention) the ranks would disagree on this
        # collective arg. (The masked slab max_m below is likewise fixed at
        # cap * ep_group_size for the same cross-rank / overflow safety; only
        # expected_m, a per-rank-local GEMM schedule hint, uses the actual batch.)
        num_max_tokens = self.num_max_dispatch_tokens_per_rank
        # Non-masked (hybrid / direct-extend) path reads exact per-expert recv
        # counts on the CPU, so it must wait for the GPU to finish writing them
        # (matches the DeepEP elastic test which passes do_cpu_sync=1). Leaving
        # it None lets the CPU read zeros on multi-node (scaleup) dispatch. Only
        # the masked decode path keeps do_cpu_sync=False for graph capturability.
        do_cpu_sync_val = True
        if use_masked:
            do_cpu_sync_val = False

        buffer = self._get_buffer()
        self._destroy_handle()
        # num_sms/num_qps are NOT auto-resolved by ElasticBuffer when left at 0;
        # 0 means "0 SMs / 0 QPs". Multi-node RDMA dispatch needs real QPs, so
        # resolve them from the theoretical helpers (matches the DeepEP elastic
        # test harness). Single-node NVLink works with 0 QPs, hence the original
        # default only ever ran on a single node.
        _num_sms = envs.SGLANG_DEEPEP_V2_NUM_SMS.get()
        if _num_sms == 0:
            _num_sms = buffer.get_theoretical_num_sms(
                self.num_experts, self.router_topk
            )
        _num_qps = buffer.get_theoretical_num_qps(_num_sms)
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            dispatch_x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=num_max_tokens,
            expert_alignment=self.capability.expert_alignment,
            num_sms=_num_sms,
            num_qps=_num_qps,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            do_cpu_sync=do_cpu_sync_val,
            do_expand=use_expand_layout,
        )
        self._handle = handle
        # NOTE: do NOT wait here. dispatch_b() does event.current_stream_wait(),
        # so dispatch() == dispatch_a()+dispatch_b() stays behavior-equivalent to
        # the old single-shot path, while TBO/SBO can later insert another
        # micro-batch's compute between dispatch_a and dispatch_b to overlap the
        # dispatch communication. local_tokens (hidden_states.shape[0]) is carried
        # so dispatch_b can compute the per-rank-local expected_m hint.
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            event,
            use_expand_layout,
            use_masked,
            use_tma_aligned_col_major_sf,
            hidden_states.shape[0],
        )

    def dispatch_b(
        self,
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        handle,
        event,
        use_expand_layout,
        use_masked,
        use_tma_aligned_col_major_sf,
        local_tokens,
    ):
        if event.event is not None:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            recv_hidden_states, recv_hidden_states_scale = recv_x
        else:
            recv_hidden_states = recv_x
            recv_hidden_states_scale = None

        if use_expand_layout:
            # Expanded layout already has one row per local expert slot. There is
            # no recv_topk_idx tensor in this native layout; combine uses handle
            # metadata and expects top-k weights to be applied before combine.
            # Avoid exact-count CPU reads that are only needed by non-expanded
            # slicing/scatter paths.
            local_topk_ids = None
            num_recv_tokens_per_expert = []
        else:
            num_recv_tokens = int(
                handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
            )
            recv_topk_idx = recv_topk_idx[:num_recv_tokens]
            recv_topk_weights = recv_topk_weights[:num_recv_tokens]
            recv_hidden_states = recv_hidden_states[:num_recv_tokens]
            if recv_hidden_states_scale is not None:
                recv_hidden_states_scale = recv_hidden_states_scale[:num_recv_tokens]

            # Elastic dispatch epilogue already converts global expert ids to local
            # expert ids and marks non-local choices as -1. Keep it on-GPU and avoid
            # an unnecessary max().item() synchronization in the decode path.
            local_topk_ids = recv_topk_idx
            num_recv_tokens_per_expert = list(handle.num_recv_tokens_per_expert_list)

        expected_m = 0
        masked_max_m = 0
        total_expanded = 0
        if use_masked:
            # expected_m: average tokens-per-expert across the EP group, a
            # per-rank-local schedule hint for the masked GEMM (NOT a hard bound;
            # the real per-expert bound is masked_m on the GPU). Derive it from
            # the actual local batch * EP group size, matching DeepEP LL
            # (deepep.py dispatch_a uses hidden_states.shape[0]). Per-rank-local,
            # so the actual batch is safe here even under ragged DP. group size
            # == ep world size == num_experts // num_local_experts.
            ep_group_size = max(1, self.num_experts // self.num_local_experts)
            expected_m = max(
                1,
                (local_tokens * ep_group_size * self.router_topk + self.num_experts)
                // self.num_experts,
            )
            # Size the masked slab to the FIXED worst case cap * ep_group_size,
            # matching DeepEP LL's fixed buffer. A local expert receives the sum
            # over all ranks of the tokens routed to it; each rank sends at most
            # `cap` tokens (enforced by the dispatch-entry assert), so the count
            # is bounded by cap * ep_group_size regardless of DP padding mode
            # (MAX_LEN / SUM_LEN / skewed). Using the local batch for the slab
            # would be unsafe: under skewed SUM_LEN decode another rank's larger
            # batch could overflow this rank's slab.
            masked_max_m = self.num_max_dispatch_tokens_per_rank * ep_group_size
            total_expanded = recv_hidden_states.shape[0]

        return DeepEPV2DispatchOutput(
            recv_hidden_states,
            recv_hidden_states_scale,
            local_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            handle.psum_num_recv_tokens_per_expert,
            use_expand_layout,
            use_tma_aligned_col_major_sf,
            use_masked,
            expected_m,
            masked_max_m,
            total_expanded,
            self.capability.expert_alignment,
        )

    def dispatch(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        # Single-shot = dispatch_a + dispatch_b (behavior-equivalent to the
        # pre-split path; non-TBO callers use this).
        return self.dispatch_b(*self.dispatch_a(hidden_states, topk_output))

    def combine_a(self, combine_input: DeepEPV2CombineInput):
        if self._handle is None:
            raise RuntimeError(
                "DeepEP v2 combine called without a valid dispatch handle"
            )
        buffer = self._get_buffer()
        # Async: do NOT wait here; combine_b() waits. Lets TBO/SBO overlap the
        # combine communication with another micro-batch's compute.
        _num_sms = envs.SGLANG_DEEPEP_V2_NUM_SMS.get()
        if _num_sms == 0:
            _num_sms = buffer.get_theoretical_num_sms(
                self.num_experts, self.router_topk
            )
        _num_qps = buffer.get_theoretical_num_qps(_num_sms)
        combined_x, _, event = buffer.combine(
            combine_input.hidden_states,
            handle=self._handle,
            topk_weights=combine_input.topk_weights,
            num_sms=_num_sms,
            num_qps=_num_qps,
        )
        return combined_x, event

    def combine_b(self, combined_x, event):
        try:
            if event.event is not None:
                event.current_stream_wait()
            if getattr(self, "_pad_empty_combine", False):
                # Drop the dummy token padded onto an empty local batch in
                # dispatch_a so this idle rank's combined output is empty again.
                combined_x = combined_x[:0]
            return combined_x
        finally:
            self._pad_empty_combine = False
            self._destroy_handle()

    def combine(self, combine_input: DeepEPV2CombineInput) -> torch.Tensor:
        return self.combine_b(*self.combine_a(combine_input))


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH = auto()


class DeepEPV2Dispatcher(BaseDispatcher):
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "DeepEP v2 dispatch adapter currently expects BF16 model activations, "
                f"got {params_dtype}"
            )
        self.quant_config = {}
        capability = get_deepep_v2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self._impl = _DeepEPV2Impl(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            capability=capability,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
        )
        self._stage = _Stage.INITIAL
        self._dispatch_state = None
        self._combine_state = None

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        capability = get_deepep_v2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self._impl.set_runner_capability(capability)

    def dispatch_a(self, hidden_states: torch.Tensor, topk_output: TopKOutput) -> None:
        if self._stage != _Stage.INITIAL:
            raise RuntimeError(
                f"DeepEP v2 dispatch called in invalid stage: {self._stage}"
            )
        self._dispatch_state = self._impl.dispatch_a(hidden_states, topk_output)

    def dispatch_b(self) -> DispatchOutput:
        if self._dispatch_state is None:
            raise RuntimeError(
                "DeepEP v2 dispatch_b() called without a preceding dispatch_a()"
            )
        out = self._impl.dispatch_b(*self._dispatch_state)
        self._dispatch_state = None
        self._stage = _Stage.AFTER_DISPATCH
        return out

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        # Single-shot = dispatch_a + dispatch_b (behavior-equivalent to pre-split;
        # non-TBO callers use this). TBO/SBO call dispatch_a / dispatch_b
        # separately and insert another micro-batch's compute in between.
        self.dispatch_a(hidden_states, topk_output)
        return self.dispatch_b()

    def combine_a(self, combine_input: CombineInput) -> None:
        if self._stage != _Stage.AFTER_DISPATCH:
            raise RuntimeError(
                f"DeepEP v2 combine called in invalid stage: {self._stage}"
            )
        if combine_input.format != CombineInputFormat.DEEPEP_V2:
            raise TypeError(
                f"Expected DeepEP v2 combine input, got {combine_input.format}"
            )
        self._combine_state = self._impl.combine_a(combine_input)

    def combine_b(self) -> torch.Tensor:
        if self._combine_state is None:
            raise RuntimeError(
                "DeepEP v2 combine_b() called without a preceding combine_a()"
            )
        try:
            return self._impl.combine_b(*self._combine_state)
        finally:
            self._combine_state = None
            self._stage = _Stage.INITIAL

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()

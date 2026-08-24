from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional, Tuple

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
    DeepEPv2Fp8ScaleFormat,
    get_deepep_v2_fp8_scale_format,
)

logger = logging.getLogger(__name__)

_SCALE_BLOCK_SIZE = 128
# Per-expert row alignment requested from ElasticBuffer. Must equal DeepGEMM's
# get_m_alignment_for_contiguous_layout() so the non-expand psum doubles as a
# valid group-start table for the contiguous grouped GEMM.
_EXPERT_ALIGNMENT = 128
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
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
    except (ImportError, OSError) as exc:
        _fp8_quant_import_error = exc


class DeepEPv2DispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    topk_weights: torch.Tensor
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


class DeepEPv2CombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_weights: Optional[torch.Tensor]

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_V2


assert isinstance(DeepEPv2DispatchOutput, DispatchOutput)
assert isinstance(DeepEPv2CombineInput, CombineInput)


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
    # direct/hybrid is a communication-topology knob fixed at server init.
    # Callers without a running server (synthetic/unit tests) pass
    # allow_hybrid_mode to DeepEPv2Buffer.get_buffer instead, since the
    # process-wide config is not set up for them.
    from sglang.srt.runtime_context import get_exec

    return get_exec().moe.deepep_v2_mode == "hybrid"


def _quantize_for_deepep_v2_dispatch(
    hidden_states: torch.Tensor, scale_format: DeepEPv2Fp8ScaleFormat
):
    _ensure_fp8_quant_available()
    return sglang_per_token_group_quant_fp8(
        hidden_states,
        _SCALE_BLOCK_SIZE,
        column_major_scales=scale_format.tma_aligned,
        scale_tma_aligned=scale_format.tma_aligned,
        scale_ue8m0=scale_format.ue8m0,
    )


class DeepEPv2Buffer:
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
        allow_hybrid_mode: Optional[bool] = None,
    ) -> ElasticBuffer:
        _ensure_deepep_v2_available()

        if allow_hybrid_mode is None:
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

        # Reusing the process group's NCCL communicator (DeepEP's default) needs a
        # device-bound group, which SGLang's shared init does not give: it segfaults
        # in ncclTeamWorld. Let DeepEP create its own; a user value still wins.
        os.environ.setdefault("EP_REUSE_NCCL_COMM", "0")
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


class _DeepEPv2Impl:
    def __init__(
        self,
        group: dist.ProcessGroup,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        scale_format: DeepEPv2Fp8ScaleFormat,
        num_max_dispatch_tokens_per_rank: int,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.scale_format = scale_format
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.rank = dist.get_rank(group)
        self._handle = None
        self._pad_empty_combine = False

    def _destroy_handle(self) -> None:
        self._handle = None

    def _get_buffer(self) -> ElasticBuffer:
        return DeepEPv2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_max_dispatch_tokens_per_rank,
            True,  # deepep_v2 always dispatches FP8 activations + scales
        )

    def _resolve_num_sms_qps(self, buffer: ElasticBuffer) -> Tuple[int, int]:
        # ElasticBuffer reads 0 as "zero SMs/QPs", not "auto", so resolve both.
        # Multi-node RDMA needs the real QPs. Both helpers are host-only and cached,
        # so this stays capture-safe on the decode path.
        num_sms = envs.SGLANG_DEEPEP_V2_NUM_SMS.get()
        if num_sms == 0:
            num_sms = buffer.get_theoretical_num_sms(self.num_experts, self.router_topk)
        num_qps = buffer.get_theoretical_num_qps(num_sms)
        return num_sms, num_qps

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
        if self.hidden_size % _SCALE_BLOCK_SIZE != 0:
            raise ValueError(
                "DeepEP v2 FP8 dispatch requires hidden_size multiple of "
                f"{_SCALE_BLOCK_SIZE}, got {self.hidden_size}"
            )
        if topk_ids.shape[1] != self.router_topk:
            raise ValueError(
                f"DeepEP v2 topk mismatch: expected {self.router_topk}, "
                f"got {topk_ids.shape[1]}"
            )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DeepEPv2DispatchOutput:
        # Guard-first (before the import check) so misuse is reportable without
        # DeepEP installed.
        if self._handle is not None:
            raise RuntimeError(
                "DeepEP v2 dispatch called while the previous dispatch handle is "
                "still unconsumed (missing combine)"
            )
        _ensure_deepep_v2_available()
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids.to(torch.int64)
        self._validate_common(hidden_states, topk_ids)
        # Layout follows inference phase, not comm mode: expanded wins on decode and
        # regresses prefill. Keeping it off the comm mode is what makes the
        # masked-GEMM + CUDA-graph decode path available under hybrid too.
        use_expand_layout = not get_is_extend_in_batch()
        # masked GEMM is built from the expanded layout (expand_to_masked_slab), so
        # masked <=> expanded. async dispatch (cpu_sync=False) gives a static
        # capturable recv shape; the masked GEMM bounds compute by masked_m, so the
        # full (safe) cap costs no extra GEMM.
        use_masked = use_expand_layout

        # An idle rank with 0 tokens never fires the dispatch notify warps, so no
        # rank's recv count becomes ready and the do_cpu_sync readback hangs. Pad to
        # one dummy token; combine() slices it back off. The masked path does not
        # read counts back, so it tolerates empty.
        self._pad_empty_combine = (not use_masked) and hidden_states.shape[0] == 0
        if self._pad_empty_combine:
            hidden_states = hidden_states.new_zeros((1, hidden_states.shape[-1]))
            # A token's top-k experts must be DISTINCT valid ids: duplicates (e.g.
            # all-zero -> expert 0 repeated) fault the dispatch kernel. Route the
            # dummy to experts [0, 1, ..., topk-1] with zero weights so it
            # contributes nothing even before combine() slices it off.
            topk_ids = torch.arange(
                topk_ids.shape[-1], dtype=topk_ids.dtype, device=topk_ids.device
            ).unsqueeze(0)
            topk_weights = topk_weights.new_zeros((1, topk_weights.shape[-1]))

        _ensure_fp8_quant_available()
        if use_masked:
            # Follow the hardware scale format (DEEPGEMM_SCALE_UE8M0 via
            # scale_format.ue8m0). Hopper (False): plain row-major fp32 scale,
            # and _run_masked_gemm does its own e8m0/tma-major alignment.
            # Blackwell (True): pre-quantize the activation against a col-major
            # UE8M0 scale so it already matches the layout the masked GEMM
            # consumes.
            _ue8m0 = self.scale_format.ue8m0
            dispatch_x = sglang_per_token_group_quant_fp8(
                hidden_states,
                _SCALE_BLOCK_SIZE,
                column_major_scales=_ue8m0,
                scale_tma_aligned=_ue8m0,
                scale_ue8m0=_ue8m0,
            )
            use_tma_aligned_col_major_sf = _ue8m0
        else:
            dispatch_x = _quantize_for_deepep_v2_dispatch(
                hidden_states, self.scale_format
            )
            use_tma_aligned_col_major_sf = self.scale_format.tma_aligned

        # Collective arg: every rank must pass the same value, so use the fixed cap.
        # Deriving it from the local batch would disagree across ranks under ragged DP.
        num_max_tokens = self.num_max_dispatch_tokens_per_rank
        # The contiguous path reads exact per-expert counts on the CPU and must wait
        # for the GPU to write them; without this the CPU reads zeros on multi-node
        # dispatch. The masked path stays async so it can be captured.
        do_cpu_sync_val = True
        if use_masked:
            do_cpu_sync_val = False

        buffer = self._get_buffer()
        _num_sms, _num_qps = self._resolve_num_sms_qps(buffer)
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
            dispatch_x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=num_max_tokens,
            expert_alignment=_EXPERT_ALIGNMENT,
            num_sms=_num_sms,
            num_qps=_num_qps,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            do_cpu_sync=do_cpu_sync_val,
            do_expand=use_expand_layout,
        )
        self._handle = handle
        local_tokens = hidden_states.shape[0]
        # event.current_stream_wait() is a GPU stream dependency (not a CPU
        # sync); the do_cpu_sync=False masked decode path stays CUDA-graph
        # capturable.
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

        expected_m = 0
        masked_max_m = 0
        total_expanded = 0
        if use_masked:
            # expected_m is a schedule hint, not a bound -- masked_m on the GPU is the
            # real per-expert bound -- so deriving it from the local batch is safe.
            ep_group_size = max(1, self.num_experts // self.num_local_experts)
            expected_m = max(
                1,
                (local_tokens * ep_group_size * self.router_topk + self.num_experts)
                // self.num_experts,
            )
            # A local expert receives from every rank and each sends at most `cap`, so
            # cap * ep_group_size is the worst case. Sizing the slab from the local batch
            # would overflow under skewed DP.
            masked_max_m = self.num_max_dispatch_tokens_per_rank * ep_group_size
            total_expanded = recv_hidden_states.shape[0]

        return DeepEPv2DispatchOutput(
            recv_hidden_states,
            recv_hidden_states_scale,
            local_topk_ids,
            recv_topk_weights,
            handle.psum_num_recv_tokens_per_expert,
            use_expand_layout,
            use_tma_aligned_col_major_sf,
            use_masked,
            expected_m,
            masked_max_m,
            total_expanded,
            _EXPERT_ALIGNMENT,
        )

    def combine(self, combine_input: DeepEPv2CombineInput) -> torch.Tensor:
        # Guard-first (before any DeepEP work) so misuse is reportable without
        # DeepEP installed.
        if self._handle is None:
            raise RuntimeError(
                "DeepEP v2 combine called without a valid dispatch handle"
            )
        # The handle is single-use: release it whether combine succeeds or
        # raises, so a failed step cannot poison the next dispatch.
        try:
            buffer = self._get_buffer()
            _num_sms, _num_qps = self._resolve_num_sms_qps(buffer)
            combined_x, _, event = buffer.combine(
                combine_input.hidden_states,
                handle=self._handle,
                topk_weights=combine_input.topk_weights,
                num_sms=_num_sms,
                num_qps=_num_qps,
            )
            # Stream dependency, not a CPU sync (graph-safe).
            if event.event is not None:
                event.current_stream_wait()
            if self._pad_empty_combine:
                # Drop the dummy token padded onto an empty local batch in
                # dispatch so this idle rank's combined output is empty again.
                combined_x = combined_x[:0]
            return combined_x
        finally:
            self._pad_empty_combine = False
            self._destroy_handle()


class DeepEPv2Dispatcher(BaseDispatcher):
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
        scale_format = get_deepep_v2_fp8_scale_format()
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self._impl = _DeepEPv2Impl(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            scale_format=scale_format,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
        )

    # Single-shot only: TBO/SBO are rejected at server start.
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        return self._impl.dispatch(hidden_states, topk_output)

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if combine_input.format != CombineInputFormat.DEEPEP_V2:
            raise TypeError(
                f"Expected DeepEP v2 combine input, got {combine_input.format}"
            )
        return self._impl.combine(combine_input)

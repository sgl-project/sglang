from __future__ import annotations

import logging
import os
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
    DeepEPv2OutputDtype,
    DeepEPv2RunnerCapability,
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


class DeepEPv2CombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: Optional[torch.Tensor]
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
    # direct/hybrid is a communication-topology knob resolved from ServerArgs.
    # Callers without a running server (synthetic/unit tests) must pass
    # allow_hybrid_mode explicitly instead (get_server_args() raises when the
    # process-wide ServerArgs is not set).
    from sglang.srt.runtime_context import get_server_args

    return get_server_args().deepep_v2_mode == "hybrid"


def _quantize_for_deepep_v2_dispatch(
    hidden_states: torch.Tensor, capability: DeepEPv2RunnerCapability
):
    _ensure_fp8_quant_available()
    return sglang_per_token_group_quant_fp8(
        hidden_states,
        _SCALE_BLOCK_SIZE,
        column_major_scales=capability.fp8_scale_tma_aligned,
        scale_tma_aligned=capability.fp8_scale_tma_aligned,
        scale_ue8m0=capability.fp8_scale_ue8m0,
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

        # DeepEP reuses the torch process group's internal NCCL communicator
        # when EP_REUSE_NCCL_COMM=1 (its default). That path requires the group
        # to be device-bound at init_process_group time (eager comm init),
        # which SGLang's shared init does not do -- reusing then reads an
        # uninitialized communicator and ElasticBuffer sizing segfaults in
        # ncclTeamWorld. Default to letting DeepEP create its own communicator
        # (it binds to the already-set current device); setdefault keeps any
        # explicit user override.
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
        capability: DeepEPv2RunnerCapability,
        num_max_dispatch_tokens_per_rank: int,
        allow_hybrid_mode: Optional[bool] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.capability = capability
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        # Prefill and decode have different static-shape requirements.  A large
        # one-pass prefill needs a correspondingly large ElasticBuffer, but the
        # decode masked-GEMM slab only needs to cover the largest decode batch.
        # Reusing the prefill cap for that slab can allocate multiple GiB per
        # expert during CUDA graph capture.  Keep the communication buffer cap
        # unchanged and allow a smaller, fixed, cross-rank decode slab cap.
        masked_cap = int(
            os.environ.get(
                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
                str(num_max_dispatch_tokens_per_rank),
            )
        )
        if masked_cap < 1 or masked_cap > num_max_dispatch_tokens_per_rank:
            raise ValueError(
                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK "
                f"must be in [1, {num_max_dispatch_tokens_per_rank}], got "
                f"{masked_cap}"
            )
        self.masked_num_max_dispatch_tokens_per_rank = masked_cap
        self.allow_hybrid_mode = allow_hybrid_mode
        self.rank = dist.get_rank(group)
        self._handle = None
        self._pad_empty_combine = False
        self._dispatch_seq = 0

    def set_runner_capability(self, capability: DeepEPv2RunnerCapability) -> None:
        if self.capability != capability:
            self._destroy_handle()
            self.capability = capability

    def _uses_fp8_dispatch_output(self) -> bool:
        return self.capability.output_dtype == DeepEPv2OutputDtype.FP8

    def _destroy_handle(self) -> None:
        self._handle = None

    def _get_buffer(self) -> ElasticBuffer:
        return DeepEPv2Buffer.get_buffer(
            self.group,
            self.hidden_size,
            self.router_topk,
            self.num_max_dispatch_tokens_per_rank,
            self._uses_fp8_dispatch_output(),
            allow_hybrid_mode=self.allow_hybrid_mode,
        )

    def _resolve_num_sms_qps(self, buffer: ElasticBuffer) -> Tuple[int, int]:
        # num_sms/num_qps are NOT auto-resolved by ElasticBuffer when left at 0;
        # 0 means "0 SMs / 0 QPs". Multi-node RDMA dispatch needs real QPs, so
        # resolve them from the theoretical helpers (matches the DeepEP elastic
        # test harness). Single-node NVLink works with 0 QPs.
        # get_theoretical_num_sms is @weak_lru-cached in DeepEP with fixed inputs
        # here, and its first (modeling) call happens during eager warmup -- so on
        # the CUDA-graph decode path this is a cache lookup: pure host work, no
        # device sync, capture-safe.
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

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DeepEPv2DispatchOutput:
        # Handle lifecycle: dispatch produces exactly one handle that the next
        # combine() consumes. Guard-first (before the import check) so misuse is
        # reportable without DeepEP installed.
        if self._handle is not None:
            raise RuntimeError(
                "DeepEP v2 dispatch called while the previous dispatch handle is "
                "still unconsumed (missing combine)"
            )
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
        force_expand_prefill = (
            os.environ.get("SGLANG_DEEPEP_V2_EXPAND_PREFILL") == "1"
        )
        use_expand_layout = self.capability.use_expanded_layout and (
            force_expand_prefill or not get_is_extend_in_batch()
        )
        # Decode uses the graph-oriented expanded -> masked-GEMM bridge.  Qwen
        # prefill uses DeepEP's expanded communication layout but consumes it via
        # the existing m_indices/contiguous-GEMM adapter; this is the split that
        # existed before layout selection was changed to decode-vs-prefill in
        # 941b17a9d.  Treating "expanded prefill" as "masked decode" allocates a
        # cap-sized slab and is not the Qwen prefill contract.
        use_masked = use_expand_layout and not get_is_extend_in_batch()
        if (
            use_masked
            and hidden_states.shape[0]
            > self.masked_num_max_dispatch_tokens_per_rank
        ):
            raise ValueError(
                "DeepEP v2 masked decode input exceeds the per-rank slab "
                f"capacity {self.masked_num_max_dispatch_tokens_per_rank}, got "
                f"{hidden_states.shape[0]}. Increase "
                "SGLANG_DEEPEP_V2_MASKED_NUM_MAX_DISPATCH_TOKENS_PER_RANK."
            )

        self._dispatch_seq += 1
        trace_dispatch = os.environ.get("SGLANG_DEEPEP_V2_TRACE_DISPATCH") == "1"
        if trace_dispatch:
            logger.warning(
                "DeepEP v2 dispatch enter: ep_rank=%s seq=%s tokens=%s "
                "is_extend_in_batch=%s use_expand_layout=%s use_masked=%s",
                self.rank,
                self._dispatch_seq,
                hidden_states.shape[0],
                get_is_extend_in_batch(),
                use_expand_layout,
                use_masked,
            )

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
        # ElasticBuffer's expanded layout handles an empty sender even with
        # do_cpu_sync=True (validated on DEP16).  Padding is required only for
        # the non-expanded exact-count path.  Keying this off `not use_masked`
        # incorrectly pads expanded-prefill ranks and routes every dummy to
        # global experts 0..topk-1, creating a large artificial hotspot on EP0.
        self._pad_empty_combine = (
            not use_expand_layout and hidden_states.shape[0] == 0
        )
        if self._pad_empty_combine:
            empty_pad_tokens = int(
                os.environ.get("SGLANG_DEEPEP_V2_EMPTY_PAD_TOKENS", "1")
            )
            if empty_pad_tokens < 1:
                raise ValueError(
                    "SGLANG_DEEPEP_V2_EMPTY_PAD_TOKENS must be at least 1"
                )
            hidden_states = hidden_states.new_zeros(
                (empty_pad_tokens, hidden_states.shape[-1])
            )
            # A token's top-k experts must be DISTINCT valid ids: duplicates (e.g.
            # all-zero -> expert 0 repeated) fault the dispatch kernel. Route the
            # dummy to experts [0, 1, ..., topk-1] with zero weights so it
            # contributes nothing even before combine_b slices it off.
            topk_ids = torch.arange(
                topk_ids.shape[-1], dtype=topk_ids.dtype, device=topk_ids.device
            ).unsqueeze(0).expand(empty_pad_tokens, -1).contiguous()
            topk_weights = topk_weights.new_zeros(
                (empty_pad_tokens, topk_weights.shape[-1])
            )
            if trace_dispatch:
                logger.warning(
                    "DeepEP v2 padded empty dispatch: ep_rank=%s seq=%s "
                    "dispatch_tokens=%s",
                    self.rank,
                    self._dispatch_seq,
                    empty_pad_tokens,
                )

        # Deterministic mixed-route reproducer. Some production warmups leave
        # a subset of DP ranks idle; the adapter pads those ranks with zero
        # tokens routed to experts [0..topk). Scheduler timing makes the idle
        # subset nondeterministic, so this diagnostic can reproduce the same
        # payload shape on a fixed suffix of ranks while keeping all collective
        # arguments and tensor shapes identical.
        dummy_rank_from = int(
            os.environ.get("SGLANG_DEEPEP_V2_DUMMY_RANK_FROM", "-1")
        )
        if (
            self._dispatch_seq == 1
            and dummy_rank_from >= 0
            and self.rank >= dummy_rank_from
        ):
            hidden_states = torch.zeros_like(hidden_states)
            topk_ids = (
                torch.arange(
                    topk_ids.shape[-1],
                    dtype=topk_ids.dtype,
                    device=topk_ids.device,
                )
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
                .contiguous()
            )
            topk_weights = torch.zeros_like(topk_weights)
            if trace_dispatch:
                logger.warning(
                    "DeepEP v2 forced dummy route: ep_rank=%s seq=%s tokens=%s",
                    self.rank,
                    self._dispatch_seq,
                    hidden_states.shape[0],
                )
        elif (
            self._dispatch_seq == 1
            and dummy_rank_from >= 0
            and self.rank < dummy_rank_from
            and self._pad_empty_combine
        ):
            # If scheduler timing happened to leave a designated "real" rank
            # idle, replace its already-padded dummy payload with deterministic
            # nonzero data and distributed valid routes. combine() will still
            # slice the synthetic local output back to zero rows.
            hidden_states = torch.ones_like(hidden_states)
            token_offsets = torch.arange(
                hidden_states.shape[0], device=topk_ids.device, dtype=topk_ids.dtype
            ).unsqueeze(1)
            expert_offsets = torch.arange(
                topk_ids.shape[-1], device=topk_ids.device, dtype=topk_ids.dtype
            ).unsqueeze(0)
            topk_ids = (
                self.rank * self.router_topk
                + token_offsets * self.router_topk
                + expert_offsets
            ) % self.num_experts
            topk_weights = torch.full_like(
                topk_weights, 1.0 / self.router_topk
            )
            if trace_dispatch:
                logger.warning(
                    "DeepEP v2 synthesized real route: ep_rank=%s seq=%s tokens=%s",
                    self.rank,
                    self._dispatch_seq,
                    hidden_states.shape[0],
                )

        if self._uses_fp8_dispatch_output():
            _ensure_fp8_quant_available()
            if use_masked:
                # Follow the hardware scale format (DEEPGEMM_SCALE_UE8M0 via
                # capability.fp8_scale_ue8m0). Hopper (False): plain row-major
                # fp32 scale, and _run_masked_gemm does its own e8m0/tma-major
                # alignment. Blackwell (True): pre-quantize the activation
                # against a col-major UE8M0 scale so it already matches the
                # layout the masked GEMM consumes.
                _ue8m0 = self.capability.fp8_scale_ue8m0
                dispatch_x = sglang_per_token_group_quant_fp8(
                    hidden_states,
                    _SCALE_BLOCK_SIZE,
                    column_major_scales=_ue8m0,
                    scale_tma_aligned=_ue8m0,
                    scale_ue8m0=_ue8m0,
                )
                use_tma_aligned_col_major_sf = _ue8m0
            else:
                # Diagnostic A/B: DeepEP's elastic unit test exercises the
                # non-expanded FP8 path with ordinary row-major scales, whereas
                # the GB300 SGLang path normally supplies packed UE8M0,
                # TMA-aligned column-major scales. Keep overrides opt-in so
                # every other dispatcher input remains identical.
                force_rowmajor_fp8 = (
                    os.environ.get("SGLANG_DEEPEP_V2_FORCE_ROWMAJOR_FP8") == "1"
                )
                if force_rowmajor_fp8:
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
                    layout_ab = os.environ.get(
                        "SGLANG_DEEPEP_V2_FP8_LAYOUT_AB", "col_in_col_out"
                    )
                    if layout_ab not in {
                        "col_in_col_out",
                        "col_in_row_out",
                        "row_in_col_out",
                        "row_in_row_out",
                    }:
                        raise ValueError(
                            "Invalid SGLANG_DEEPEP_V2_FP8_LAYOUT_AB="
                            f"{layout_ab}"
                        )
                    if layout_ab.startswith("row_in_"):
                        dispatch_x = (dispatch_x[0], dispatch_x[1].contiguous())
                    use_tma_aligned_col_major_sf = layout_ab.endswith("_col_out")
                    if trace_dispatch:
                        logger.warning(
                            "DeepEP v2 FP8 layout A/B: ep_rank=%s seq=%s "
                            "mode=%s sf_shape=%s sf_stride=%s sf_dtype=%s "
                            "col_major_output=%s",
                            self.rank,
                            self._dispatch_seq,
                            layout_ab,
                            tuple(dispatch_x[1].shape),
                            dispatch_x[1].stride(),
                            dispatch_x[1].dtype,
                            use_tma_aligned_col_major_sf,
                        )
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
        # Diagnostic for the Qwen prefill-expanded contract.  The upstream
        # ElasticBuffer test exercises do_expand with CPU synchronization by
        # default, while the initial SGLang integration tied expanded layout to
        # the graph-oriented decode path and therefore forced async metadata.
        # Keep decode unchanged, but allow an eager/BCG-breakable prefill to use
        # the synchronized expanded path so we can distinguish an unfinished
        # dispatch epilogue from the downstream masked-slab kernel.
        if (
            force_expand_prefill
            and get_is_extend_in_batch()
            and os.environ.get("SGLANG_DEEPEP_V2_EXPAND_PREFILL_CPU_SYNC") == "1"
        ):
            do_cpu_sync_val = True

        buffer = self._get_buffer()
        _num_sms, _num_qps = self._resolve_num_sms_qps(buffer)
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
        if trace_dispatch:
            logger.warning(
                "DeepEP v2 dispatch returned: ep_rank=%s seq=%s "
                "use_expand_layout=%s do_cpu_sync=%s",
                self.rank,
                self._dispatch_seq,
                use_expand_layout,
                do_cpu_sync_val,
            )
        self._handle = handle
        local_tokens = hidden_states.shape[0]
        # event.current_stream_wait() is a GPU stream dependency (not a CPU
        # sync); the do_cpu_sync=False masked decode path stays CUDA-graph
        # capturable.
        if event.event is not None:
            event.current_stream_wait()

        if os.environ.get("SGLANG_DEEPEP_V2_TRACE_PSUM") == "1":
            trace_recv = recv_x[0] if isinstance(recv_x, tuple) else recv_x
            trace_scale = recv_x[1] if isinstance(recv_x, tuple) else None
            logger.warning(
                "DeepEP v2 dispatch metadata: ep_rank=%s seq=%s recv_x_shape=%s "
                "recv_x_stride=%s scale_shape=%s scale_stride=%s scale_dtype=%s "
                "psum=%s",
                self.rank,
                self._dispatch_seq,
                tuple(trace_recv.shape),
                trace_recv.stride(),
                None if trace_scale is None else tuple(trace_scale.shape),
                None if trace_scale is None else trace_scale.stride(),
                None if trace_scale is None else trace_scale.dtype,
                handle.psum_num_recv_tokens_per_expert.detach().cpu().tolist(),
            )

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
            masked_max_m = (
                self.masked_num_max_dispatch_tokens_per_rank * ep_group_size
            )
            total_expanded = recv_hidden_states.shape[0]

        return DeepEPv2DispatchOutput(
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
            trace_contig = os.environ.get("SGLANG_DEEPEP_V2_TRACE_CONTIG") == "1"
            if trace_contig:
                torch.cuda.synchronize()
                logger.warning(
                    "DeepEP v2 combine enter: ep_rank=%s seq=%s hidden=%s",
                    self.rank,
                    self._dispatch_seq,
                    tuple(combine_input.hidden_states.shape),
                )
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
            if trace_contig:
                torch.cuda.synchronize()
                logger.warning(
                    "DeepEP v2 combine returned: ep_rank=%s seq=%s output=%s "
                    "stride=%s dtype=%s contiguous=%s",
                    self.rank,
                    self._dispatch_seq,
                    tuple(combined_x.shape),
                    combined_x.stride(),
                    combined_x.dtype,
                    combined_x.is_contiguous(),
                )
            if os.environ.get("SGLANG_DEEPEP_V2_CLONE_COMBINE_OUTPUT") == "1":
                if trace_contig:
                    logger.warning(
                        "DeepEP v2 combine clone enter: ep_rank=%s seq=%s",
                        self.rank,
                        self._dispatch_seq,
                    )
                combined_x = combined_x.clone()
                torch.cuda.synchronize()
                if trace_contig:
                    logger.warning(
                        "DeepEP v2 combine clone returned: ep_rank=%s seq=%s",
                        self.rank,
                        self._dispatch_seq,
                    )
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
        allow_hybrid_mode: Optional[bool] = None,
    ):
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise NotImplementedError(
                "DeepEP v2 dispatch adapter currently expects BF16 model activations, "
                f"got {params_dtype}"
            )
        capability = get_deepep_v2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        self._impl = _DeepEPv2Impl(
            group=group,
            router_topk=router_topk,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            capability=capability,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            allow_hybrid_mode=allow_hybrid_mode,
        )

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config
        capability = get_deepep_v2_runner_capability(self)
        self.output_dtype = capability.output_dtype
        self._impl.set_runner_capability(capability)

    # This backend intentionally exposes only single-shot dispatch()/combine():
    # TBO/SBO are rejected at server start, and our overlap PoC showed the naive
    # two-phase split cannot overlap anyway (ElasticBuffer.dispatch is
    # host-blocking); a split API will land together with real TBO support.
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

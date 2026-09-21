from __future__ import annotations

"""MORI EPv2 intranode dispatcher using cco-LSA and FlyDSL kernels."""

import logging
import os
from enum import Enum, auto
from functools import lru_cache
from typing import NamedTuple, Optional

import torch

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, is_tbo_enabled
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hip

logger = logging.getLogger(__name__)
MXFP4_BLOCK_SIZE = 32
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


class _MoriEPv2TBOGeometry(NamedTuple):
    dispatch_block_num: Optional[int]
    dispatch_warp_num_per_block: Optional[int]
    combine_block_num: Optional[int]
    combine_warp_num_per_block: Optional[int]


def _resolve_tbo_geometry(
    *,
    tbo_enabled: bool,
    dispatch_block_num: int,
    combine_block_num: int,
    dispatch_warp_num_per_block: int,
    combine_warp_num_per_block: int,
) -> _MoriEPv2TBOGeometry:
    if not tbo_enabled:
        return _MoriEPv2TBOGeometry(None, None, None, None)
    values = {
        "SGLANG_MORI_EPV2_TBO_DISPATCH_BLOCK_NUM": dispatch_block_num,
        "SGLANG_MORI_EPV2_TBO_COMBINE_BLOCK_NUM": combine_block_num,
        "SGLANG_MORI_EPV2_TBO_DISPATCH_WARP_NUM_PER_BLOCK": (
            dispatch_warp_num_per_block
        ),
        "SGLANG_MORI_EPV2_TBO_COMBINE_WARP_NUM_PER_BLOCK": (combine_warp_num_per_block),
    }
    for env_name, value in values.items():
        if value <= 0:
            raise ValueError(f"{env_name} must be positive; got {value}")
    return _MoriEPv2TBOGeometry(
        dispatch_block_num,
        dispatch_warp_num_per_block,
        combine_block_num,
        combine_warp_num_per_block,
    )


class _MoriEPv2CommStreamPool:
    _streams = {}

    @classmethod
    def get(cls, group, priority: int = 0) -> torch.cuda.Stream:
        key = (torch.cuda.current_device(), id(group), priority)
        stream = cls._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(priority=priority)
            cls._streams[key] = stream
            logger.info(
                "[MORI EPv2 TBO] configured shared comm stream priority=%d",
                priority,
            )
        return stream


def _get_tbo_comm_stream(group, *, tbo_enabled: bool, async_finish: bool):
    if not (tbo_enabled and async_finish):
        return None
    if not get_bool_env_var("SGLANG_MORI_EPV2_TBO_USE_COMM_STREAM", "true"):
        return None
    priority = get_int_env_var("SGLANG_MORI_EPV2_TBO_COMM_STREAM_PRIORITY", 0)
    return _MoriEPv2CommStreamPool.get(group, priority)


class MoriEPv2NormalDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor | None
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype
    # Optional destination for AITER's post-expert result. Borrowed from this
    # dispatcher's MORI EPv2 arena, which stays alive through graph replay/combine.
    expert_output: torch.Tensor | None = None
    # Safe row bound for AITER input trimming; 0 disables it.
    recv_cap: int = 0

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


assert isinstance(MoriEPv2NormalDispatchOutput, DispatchOutput)


@lru_cache(maxsize=4)
def _init_cco_communicator(group, instance_id: int):
    from mori.cco import Communicator

    parallel = get_parallel()
    world_size = parallel.moe_ep_size
    rank = parallel.moe_ep_rank
    if world_size > 8:
        raise ValueError(
            f"MORI EPv2 is currently intranode-only (world_size<=8); got {world_size}"
        )

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = group.broadcast_object(uid, src=0)
    per_rank_vmm_gb = get_int_env_var("SGLANG_MORI_EPV2_PER_RANK_VMM_GB", 4)
    if per_rank_vmm_gb <= 0:
        raise ValueError("SGLANG_MORI_EPV2_PER_RANK_VMM_GB must be positive")
    comm = Communicator.init(
        world_size,
        rank,
        uid,
        per_rank_vmm=per_rank_vmm_gb * (1 << 30),
    )
    logger.info(
        "[MORI EPv2 init] cco communicator world=%d rank=%d instance=%d "
        "per_rank_vmm_gb=%d",
        world_size,
        rank,
        instance_id,
        per_rank_vmm_gb,
    )
    return comm


@lru_cache(maxsize=4)
def init_mori_epv2_op(
    group,
    router_topk: int,
    num_experts: int,
    num_local_experts: int,
    hidden_size: int,
    params_dtype: torch.dtype,
    max_tokens_per_rank: int,
    instance_id: int = 0,
    max_total_recv_tokens: int = 0,
    dispatch_dtype: torch.dtype = torch.bfloat16,
    geometry: _MoriEPv2TBOGeometry = _MoriEPv2TBOGeometry(None, None, None, None),
):
    from mori.ops.dispatch_combine_v2 import (
        EpDispatchCombineConfig,
        EpDispatchCombineOp,
    )

    if params_dtype != torch.bfloat16:
        raise ValueError(
            "MORI EPv2 requires BF16 expert/combine output; "
            f"got params_dtype={params_dtype}"
        )
    if dispatch_dtype not in (torch.bfloat16, torch.float4_e2m1fn_x2):
        raise ValueError(f"unsupported MORI EPv2 dispatch dtype: {dispatch_dtype}")

    parallel = get_parallel()
    world_size = parallel.moe_ep_size
    rank = parallel.moe_ep_rank
    comm = _init_cco_communicator(group, instance_id)
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_size,
        max_num_inp_token_per_rank=max_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=router_topk,
        data_type=torch.bfloat16,
        dispatch_data_type=(
            dispatch_dtype if dispatch_dtype != torch.bfloat16 else None
        ),
        combine_data_type=(
            torch.bfloat16 if dispatch_dtype != torch.bfloat16 else None
        ),
        scale_dim=(
            hidden_size // MXFP4_BLOCK_SIZE if dispatch_dtype != torch.bfloat16 else 0
        ),
        scale_type_size=1 if dispatch_dtype != torch.bfloat16 else 0,
        combine_mode="gather",
        max_total_recv_tokens=max_total_recv_tokens,
        dispatch_block_num=geometry.dispatch_block_num,
        combine_block_num=geometry.combine_block_num,
        warp_num_per_block=geometry.dispatch_warp_num_per_block,
        combine_warp_num_per_block=geometry.combine_warp_num_per_block,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    logger.info(
        "[MORI EPv2 init] world=%d rank=%d hidden=%d experts=%d local_experts=%d "
        "topk=%d max_tokens=%d recv_cap=%d dispatch_dtype=%s geometry=%s schedule=%s",
        world_size,
        rank,
        hidden_size,
        num_experts,
        num_local_experts,
        router_topk,
        max_tokens_per_rank,
        cfg.effective_max_recv,
        dispatch_dtype,
        geometry,
        cfg.schedule,
    )
    return op


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class MoriEPv2Dispatcher(BaseDispatcher):
    """MORI EPv2 dispatcher, including two-child TBO orchestration."""

    def __init__(
        self,
        group,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        params_dtype: torch.dtype | None = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        instance_id: int = 0,
    ):
        super().__init__()
        del permute_fusion, return_recv_hook
        if not deepep_mode.enable_normal() or deepep_mode.enable_low_latency():
            raise ValueError("MORI EPv2 currently supports normal mode only")
        try:
            from mori.cco import Communicator  # noqa: F401
            from mori.ops.dispatch_combine_v2 import EpDispatchCombineOp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MORI EPv2 requires a MORI build with cco and dispatch_combine_v2"
            ) from exc

        # EPv2 forwards exactly top-k routed experts and has no fake expert slot.
        os.environ.setdefault("AITER_FLYDSL_EP_NO_FAKE_EXPERT", "1")
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.instance_id = instance_id
        self.async_finish = async_finish
        self.dispatch_dtype = torch.bfloat16
        tbo_enabled = is_tbo_enabled()
        # Receive trimming falls back when sender metadata is unavailable,
        # including TBO children without per-rank token counts.
        self._trim_recv = get_bool_env_var("SGLANG_MORI_EPV2_TRIM_RECV", "true")
        # Disable direct output for TBO to preserve buffer/stream ownership.
        self._direct_output = not tbo_enabled and get_bool_env_var(
            "SGLANG_MORI_EPV2_AITER_DIRECT_OUTPUT", "true"
        )
        self._comm_stream = _get_tbo_comm_stream(
            group, tbo_enabled=tbo_enabled, async_finish=async_finish
        )
        self._geometry = _resolve_tbo_geometry(
            tbo_enabled=tbo_enabled,
            dispatch_block_num=get_int_env_var(
                "SGLANG_MORI_EPV2_TBO_DISPATCH_BLOCK_NUM", 32
            ),
            combine_block_num=get_int_env_var(
                "SGLANG_MORI_EPV2_TBO_COMBINE_BLOCK_NUM", 48
            ),
            dispatch_warp_num_per_block=get_int_env_var(
                "SGLANG_MORI_EPV2_TBO_DISPATCH_WARP_NUM_PER_BLOCK", 4
            ),
            combine_warp_num_per_block=get_int_env_var(
                "SGLANG_MORI_EPV2_TBO_COMBINE_WARP_NUM_PER_BLOCK", 4
            ),
        )
        self.max_tokens_per_rank = get_int_env_var(
            "SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )
        self._op = None
        self._stage = _Stage.INITIAL
        self.fp4_quant_func = None
        if _use_aiter:
            from aiter import QuantType, get_hip_quant

            self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)

        parallel = get_parallel()
        self.expert_mask_gpu = torch.zeros(
            num_experts,
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        start = parallel.moe_ep_rank * num_local_experts
        self.expert_mask_gpu[start : start + num_local_experts] = 1

    def _initialize_op(self):
        # set_quant_config runs during model loading, before CUDA graph capture,
        # so collective CCO window creation and graph-tier JIT happen safely.
        self._op = init_mori_epv2_op(
            self.group,
            self.router_topk,
            self.num_experts,
            self.num_local_experts,
            self.hidden_size,
            self.params_dtype,
            self.max_tokens_per_rank,
            self.instance_id,
            get_int_env_var("SGLANG_MORI_EPV2_PREALLOC_MAX_RECV_TOKENS", 0),
            self.dispatch_dtype,
            self._geometry,
        )
        prepare_recv_cap = getattr(self._op, "prepare_recv_cap", None)
        if prepare_recv_cap is None:
            return
        graph_cap_max = get_int_env_var("SGLANG_MORI_EPV2_GRAPH_RECV_CAP_MAX", 8192)
        graph_cap = 32
        while graph_cap <= min(graph_cap_max, self._op.cfg.effective_max_recv):
            prepare_recv_cap(graph_cap)
            graph_cap *= 2

    @property
    def op(self):
        if self._op is None:
            self._initialize_op()
        return self._op

    def _select_recv_cap(self, eager_cluster_rows: Optional[int] = None):
        if eager_cluster_rows is not None:
            # Keep the optional MORI API independent from the separate FlyDSL
            # dispatcher, which is not part of this SGLang source tree.
            cluster_rows = int(eager_cluster_rows)
            physical_cap = self.op.cfg.effective_max_recv
            if cluster_rows < 0 or physical_cap <= 0:
                raise ValueError(
                    f"Invalid MORI EPv2 capacity: {cluster_rows=}, {physical_cap=}"
                )
            eager_cap = min(
                physical_cap, max(32, 1 << (max(1, cluster_rows) - 1).bit_length())
            )
            if not eager_cap & (eager_cap - 1):
                return eager_cap
        if not self._trim_recv:
            return self.op.cfg.effective_max_recv
        # Reuse the MORI EP/AITER receive bound for EPv2's token-major layout.
        # EPv2 can also pass this bound to MORI builds with a native recv_cap API.
        from sglang.srt.layers.moe.moe_runner.aiter import _mori_decode_recv_bound

        return (
            _mori_decode_recv_bound(
                self.op.cfg.effective_max_recv,
                self.router_topk,
                local_rows=self._num_tokens,
                is_epv2=True,
            )
            or self.op.cfg.effective_max_recv
        )

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        weight_dtype = quant_config.get("weight_dtype")
        self.dispatch_dtype = (
            torch.float4_e2m1fn_x2
            if weight_dtype == torch.float4_e2m1fn_x2
            else torch.bfloat16
        )
        override = os.environ.get("SGLANG_MORI_EPV2_DISPATCH_DTYPE", "").lower()
        if override:
            if override not in ("bf16", "fp4"):
                raise ValueError(
                    "SGLANG_MORI_EPV2_DISPATCH_DTYPE must be bf16 or fp4; "
                    f"got {override!r}"
                )
            self.dispatch_dtype = (
                torch.float4_e2m1fn_x2 if override == "fp4" else torch.bfloat16
            )
        if (
            self.dispatch_dtype == torch.float4_e2m1fn_x2
            and self.fp4_quant_func is None
        ):
            raise RuntimeError(
                "MORI EPv2 FP4 dispatch requires Aiter per_1x32 quantization"
            )
        if self._op is None:
            self._initialize_op()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        dynamic_recv_cluster_rows: Optional[int] = None,
    ) -> DispatchOutput:
        self.dispatch_a(
            hidden_states,
            topk_output,
            dynamic_recv_cluster_rows=dynamic_recv_cluster_rows,
        )
        return self.dispatch_b()

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        dynamic_recv_cluster_rows: Optional[int] = None,
    ) -> None:
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "MORI EPv2 expects BF16 activations before transport quantization; "
                f"hidden_states.dtype={hidden_states.dtype}"
            )
        self._num_tokens = hidden_states.shape[0]
        self._dynamic_recv_cluster_rows = dynamic_recv_cluster_rows
        output_dtype = hidden_states.dtype
        scale = None
        if self.dispatch_dtype == torch.float4_e2m1fn_x2:
            if self._num_tokens:
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=hidden_states.device,
                )
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=hidden_states.device,
                )
        ready_event = None
        if self._comm_stream is not None:
            ready_event = torch.cuda.Event(blocking=False, interprocess=False)
            ready_event.record(torch.cuda.current_stream())
        self._dispatch_intermediate_state = (
            hidden_states,
            topk_output.topk_weights.to(torch.float32),
            topk_output.topk_ids.to(torch.int32),
            scale,
            output_dtype,
            ready_event,
        )

    def dispatch_b(self) -> DispatchOutput:
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        hidden_states, topk_weights, topk_ids, scale, output_dtype, ready_event = (
            self._dispatch_intermediate_state
        )
        del self._dispatch_intermediate_state
        recv_cap = self._select_recv_cap(self._dynamic_recv_cluster_rows)
        kwargs = {"return_routing": True}
        if hasattr(self.op, "prepare_recv_cap"):
            kwargs.update(recv_cap=recv_cap, clone_routing=False)
        if self._comm_stream is None:
            result = self.op.dispatch(
                hidden_states,
                topk_weights,
                scale,
                topk_ids,
                **kwargs,
            )
        else:
            compute_stream = torch.cuda.current_stream()
            keepalive = (hidden_states, topk_weights, topk_ids, scale)
            with torch.cuda.stream(self._comm_stream):
                assert ready_event is not None
                self._comm_stream.wait_event(ready_event)
                result = self.op.dispatch(
                    hidden_states,
                    topk_weights,
                    scale,
                    topk_ids,
                    **kwargs,
                )
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(self._comm_stream)
            compute_stream.wait_event(done_event)
            del keepalive
        (
            recv_hidden,
            recv_weights,
            recv_scales,
            recv_indices,
            total_recv,
            routing,
        ) = result
        if recv_scales is not None and self.dispatch_dtype == torch.float4_e2m1fn_x2:
            recv_scales = recv_scales.view(torch.float8_e8m0fnu)[
                :, : self.hidden_size // MXFP4_BLOCK_SIZE
            ]
        self._routing = routing
        self._recv_topk_ids = recv_indices
        self._recv_cap = recv_cap
        expert_output = None
        if self._direct_output:
            combine_in_view = getattr(self.op, "combine_in_view", None)
            if combine_in_view is not None:
                expert_output = combine_in_view()[: recv_hidden.shape[0]]
        return MoriEPv2NormalDispatchOutput(
            hidden_states=recv_hidden,
            hidden_states_scale=recv_scales,
            topk_ids=recv_indices,
            topk_weights=recv_weights,
            num_recv_tokens_per_expert=total_recv,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
            expert_output=expert_output,
            recv_cap=recv_cap if self._trim_recv else 0,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()[: self._num_tokens]

    def combine_a(self, combine_input: CombineInput) -> None:
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        ready_event = None
        if self._comm_stream is not None:
            ready_event = torch.cuda.Event(blocking=False, interprocess=False)
            ready_event.record(torch.cuda.current_stream())
        self._combine_intermediate_state = (*tuple(combine_input), ready_event)

    def combine_b(self) -> torch.Tensor:
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        hidden_states, _topk_ids, _topk_weights, ready_event = (
            self._combine_intermediate_state
        )
        del self._combine_intermediate_state
        if self._comm_stream is None:
            out, _ = self.op.combine(
                hidden_states,
                None,
                self._recv_topk_ids,
                routing=self._routing,
            )
        else:
            compute_stream = torch.cuda.current_stream()
            keepalive = (
                hidden_states,
                _topk_ids,
                _topk_weights,
                self._recv_topk_ids,
                self._routing,
            )
            with torch.cuda.stream(self._comm_stream):
                assert ready_event is not None
                self._comm_stream.wait_event(ready_event)
                out, _ = self.op.combine(
                    hidden_states,
                    None,
                    self._recv_topk_ids,
                    routing=self._routing,
                )
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(self._comm_stream)
            compute_stream.wait_event(done_event)
            del keepalive
        if not torch.cuda.is_current_stream_capturing():
            del self._routing
        return out

    def _update_stage(self, old_stage: _Stage, new_stage: _Stage) -> None:
        assert self._stage == old_stage, f"stage {self._stage} != expected {old_stage}"
        self._stage = new_stage

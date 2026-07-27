from __future__ import annotations

"""FlyDSL intranode EP dispatcher backed by aiter's FlyDSL all-to-all op."""

import logging
import os
from enum import Enum, auto
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, is_tbo_enabled
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_hip
from sglang.srt.utils.bounded_telemetry import BoundedTelemetryLogger

logger = logging.getLogger(__name__)

FP8_BLOCK_SIZE = 128
MXFP4_BLOCK_SIZE = 32

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_flydsl_telemetry = BoundedTelemetryLogger(
    logger,
    "[SGLANG_FLYDSL_TBO_TELEMETRY]",
    enabled=get_bool_env_var("SGLANG_FLYDSL_TBO_TELEMETRY", "false"),
    max_events=256,
)
_flydsl_sync_values_telemetry = BoundedTelemetryLogger(
    logger,
    "[SGLANG_FLYDSL_TBO_TELEMETRY_SYNC_VALUES]",
    enabled=_flydsl_telemetry.enabled
    and get_bool_env_var("SGLANG_FLYDSL_TBO_TELEMETRY_SYNC_VALUES", "false"),
    max_events=128,
    rank_zero_only=False,
)


class _FlyDSLTBOGeometry(NamedTuple):
    dispatch_block_num: Optional[int]
    dispatch_warp_num_per_block: Optional[int]
    combine_block_num: Optional[int]
    combine_warp_num_per_block: Optional[int]


def _resolve_tbo_geometry(
    *,
    tbo_enabled: bool,
    shared_block_num: int,
    dispatch_block_num: int,
    combine_block_num: int,
) -> _FlyDSLTBOGeometry:
    """Resolve phase-specific TBO launch pins without reading process state."""
    if not tbo_enabled:
        return _FlyDSLTBOGeometry(None, None, None, None)

    values = {
        "SGLANG_FLYDSL_TBO_BLOCK_NUM": shared_block_num,
        "SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM": dispatch_block_num,
        "SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM": combine_block_num,
    }
    for env_name, value in values.items():
        if value < 0:
            raise ValueError(f"{env_name} must be non-negative; got {value}")

    resolved_dispatch = dispatch_block_num or shared_block_num or None
    resolved_combine = combine_block_num or shared_block_num or None
    return _FlyDSLTBOGeometry(
        dispatch_block_num=resolved_dispatch,
        dispatch_warp_num_per_block=4 if resolved_dispatch is not None else None,
        combine_block_num=resolved_combine,
        combine_warp_num_per_block=4 if resolved_combine is not None else None,
    )


def _resolved_geometry_from_cache(op, phase: str):
    """Read the first-call resolved geometry after the op has populated its cache."""
    cache = getattr(op, f"_{'disp' if phase == 'dispatch' else 'comb'}_jit_cache", {})
    key = next(reversed(cache), None)
    return (key[-3], key[-2]) if key is not None else (None, None)


def _stream_telemetry(stream):
    if stream is None:
        return None, None, None
    return (
        id(stream),
        getattr(stream, "cuda_stream", None),
        getattr(stream, "priority", None),
    )


def _validate_stream_priority(priority: int, priority_range) -> None:
    """Reject priorities outside the range reported by the active torch backend."""
    if priority_range is None:
        return
    least_priority, greatest_priority = priority_range
    low, high = sorted((least_priority, greatest_priority))
    if not low <= priority <= high:
        raise ValueError(
            "SGLANG_FLYDSL_TBO_COMM_STREAM_PRIORITY="
            f"{priority} is outside torch's supported stream priority range "
            f"[{low}, {high}]"
        )


def _should_sync_recv_values(
    *,
    telemetry_enabled: bool,
    sync_values_enabled: bool,
    pending: bool,
) -> bool:
    """Pure gate for the intentionally synchronizing receive-count probe."""
    return telemetry_enabled and sync_values_enabled and pending


def _recv_count_values(
    out_idx: torch.Tensor,
    total_recv: torch.Tensor,
    *,
    local_expert_start: int,
    num_local_experts: int,
) -> tuple[list[int], int]:
    """Materialize local expert assignment counts with one device-to-host wait."""
    row_is_valid = torch.arange(out_idx.shape[0], device=out_idx.device) < total_recv[0]
    expert_ids = out_idx[row_is_valid].reshape(-1).to(torch.int64)
    local_ids = expert_ids - local_expert_start
    local_ids = local_ids[(local_ids >= 0) & (local_ids < num_local_experts)]
    counts = torch.bincount(local_ids, minlength=num_local_experts).cpu().tolist()
    return counts, sum(counts)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _resolve_tbo_child_cluster_rows(
    parent_global_num_tokens: Optional[Sequence[int]],
    child_padded_rows_by_rank: Optional[Sequence[Sequence[int]]],
) -> Optional[tuple[int, ...]]:
    """Resolve rank-consistent child row totals from synchronized host metadata."""
    if not parent_global_num_tokens or not child_padded_rows_by_rank:
        return None
    parent_rows = tuple(int(value) for value in parent_global_num_tokens)
    child_rows = tuple(
        tuple(int(value) for value in rank_rows)
        for rank_rows in child_padded_rows_by_rank
    )
    if len(child_rows) != len(parent_rows) or not child_rows:
        return None
    num_children = len(child_rows[0])
    if num_children == 0 or any(len(rows) != num_children for rows in child_rows):
        return None
    if any(value < 0 for value in parent_rows) or any(
        value < 0 for rows in child_rows for value in rows
    ):
        raise ValueError("FlyDSL TBO token counts must be non-negative")
    # Splitting and per-child alignment may add padding, but cannot lose a
    # parent's dispatch rows on any rank.
    if any(sum(rows) < parent for rows, parent in zip(child_rows, parent_rows)):
        return None
    return tuple(
        sum(rows[child] for rows in child_rows) for child in range(num_children)
    )


def _resolve_eager_recv_cap(cluster_rows: int, physical_cap: int) -> Optional[int]:
    """Return a power-of-two eager bucket, or None for conservative fallback."""
    if cluster_rows < 0 or physical_cap <= 0:
        raise ValueError(
            f"invalid FlyDSL recv-cap inputs: {cluster_rows=}, {physical_cap=}"
        )
    candidate = min(physical_cap, max(32, _next_power_of_two(cluster_rows)))
    # Dynamic JIT variants are power-of-two only. A non-power-of-two physical
    # clamp cannot satisfy both the requested formula and that invariant, so
    # retain the existing full physical cap instead of shrinking unsafely.
    if candidate & (candidate - 1):
        return None
    return candidate


def _validate_all_rank_recv_cap(
    candidate_caps: Sequence[int],
    actual_local_dispatch_rows: Sequence[int],
) -> None:
    """Validate one child's all-rank candidate and actual dispatch row total."""
    candidates = tuple(int(value) for value in candidate_caps)
    actual_rows = tuple(int(value) for value in actual_local_dispatch_rows)
    if not candidates or len(candidates) != len(actual_rows):
        raise RuntimeError(
            "FlyDSL dynamic recv-cap validation requires one cap and row count "
            "from every EP rank"
        )
    if len(set(candidates)) != 1:
        raise RuntimeError(
            f"FlyDSL dynamic recv-cap mismatch across ranks: {candidates}"
        )
    if any(value < 0 for value in actual_rows):
        raise RuntimeError(
            f"FlyDSL dynamic recv-cap saw negative dispatch rows: {actual_rows}"
        )
    actual_cluster_rows = sum(actual_rows)
    if candidates[0] < actual_cluster_rows:
        raise RuntimeError(
            "FlyDSL dynamic recv-cap underbound: "
            f"cap={candidates[0]} actual_cluster_rows={actual_cluster_rows} "
            f"actual_local_dispatch_rows={actual_rows}"
        )


def _configured_physical_recv_cap(world_size: int) -> int:
    max_per_rank = get_int_env_var(
        "SGLANG_FLYDSL_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
    )
    configured_total = get_int_env_var("SGLANG_FLYDSL_PREALLOC_MAX_RECV_TOKENS", 0)
    if configured_total <= 0:
        effective_per_rank = max_per_rank
    else:
        effective_per_rank = min(
            (configured_total + world_size - 1) // world_size, max_per_rank
        )
    return world_size * effective_per_rank


def prepare_tbo_eager_recv_cap_metadata(
    *,
    parent_global_num_tokens: Optional[Sequence[int]],
    children: Sequence,
    group,
) -> bool:
    """Populate two TBO children once per eager forward.

    Returns False without collectives when required synchronized parent metadata
    is unavailable. Exact per-rank child padding needs one small host all-gather;
    the optional validator uses one additional diagnostic-only all-gather.
    """
    if not get_bool_env_var("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "true"):
        return False
    world_size = int(group.world_size)
    if (
        not parent_global_num_tokens
        or len(parent_global_num_tokens) != world_size
        or not children
        or any(getattr(child, "tbo_padded_len", None) is None for child in children)
    ):
        return False

    local_rows = tuple(int(child.tbo_padded_len) for child in children)
    local_tensor = torch.tensor(local_rows, dtype=torch.int64)
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor, group=group.cpu_group)
    rows_by_rank = tuple(
        tuple(int(value) for value in tensor.tolist()) for tensor in gathered
    )
    cluster_rows = _resolve_tbo_child_cluster_rows(
        parent_global_num_tokens, rows_by_rank
    )
    if cluster_rows is None:
        return False

    for child, rows in zip(children, cluster_rows, strict=True):
        child.flydsl_tbo_cluster_dispatch_rows = rows

    if get_bool_env_var("SGLANG_FLYDSL_DYNAMIC_RECV_CAP_VALIDATE", "false"):
        physical_cap = _configured_physical_recv_cap(world_size)
        local_candidates = tuple(
            _resolve_eager_recv_cap(rows, physical_cap) for rows in cluster_rows
        )
        if any(candidate is None for candidate in local_candidates):
            raise RuntimeError(
                "FlyDSL dynamic recv-cap validation requires a power-of-two "
                f"physical clamp; got physical_cap={physical_cap}"
            )
        diagnostic = torch.tensor((*local_candidates, *local_rows), dtype=torch.int64)
        diagnostic_gathered = [torch.empty_like(diagnostic) for _ in range(world_size)]
        dist.all_gather(diagnostic_gathered, diagnostic, group=group.cpu_group)
        values_by_rank = [tensor.tolist() for tensor in diagnostic_gathered]
        for child_index in range(len(children)):
            _validate_all_rank_recv_cap(
                [values[child_index] for values in values_by_rank],
                [values[len(children) + child_index] for values in values_by_rank],
            )
    return True


class DispatchDtype(Enum):
    bf16 = "bfloat16"
    fp8 = "float8"
    fp4 = "mxfp4"


class CombineDtype(Enum):
    bf16 = "bfloat16"
    fp8_direct_cast = "float8_direct_cast"


class _FlyDSLCommStreamPool:
    """One shared comm stream per device/process-group.

    Both TBO inner dispatchers must enqueue collectives in the same order on the
    same stream, while the primary compute stream runs the other ubatch.
    """

    _streams = {}

    @classmethod
    def get(cls, group, priority: int = 0) -> torch.cuda.Stream:
        key = (torch.cuda.current_device(), id(group), priority)
        stream = cls._streams.get(key)
        if stream is None:
            priority_range = None
            get_priority_range = getattr(torch.cuda, "get_stream_priority_range", None)
            if priority != 0 and get_priority_range is not None:
                priority_range = get_priority_range()
            _validate_stream_priority(priority, priority_range)
            stream = torch.cuda.Stream(priority=priority)
            cls._streams[key] = stream
            logger.info(
                "[FlyDSL TBO] configured comm stream priority=%d actual_priority=%s",
                priority,
                getattr(stream, "priority", None),
            )
        return stream


def _get_tbo_comm_stream(group, *, tbo_enabled: bool, async_finish: bool):
    if not (tbo_enabled and async_finish):
        return None
    use_comm_stream = get_bool_env_var("SGLANG_FLYDSL_TBO_USE_COMM_STREAM", "true")
    logger.info("[FlyDSL TBO] dedicated comm stream enabled=%s", use_comm_stream)
    if not use_comm_stream:
        return None
    priority = get_int_env_var("SGLANG_FLYDSL_TBO_COMM_STREAM_PRIORITY", 0)
    return _FlyDSLCommStreamPool.get(group, priority=priority)


@lru_cache(maxsize=4)
def init_flydsl_op(
    group,
    router_topk,
    num_experts,
    num_local_experts,
    hidden_size,
    params_dtype,
    num_max_dispatch_tokens_per_rank,
    instance_id=0,
    dispatch_dtype=DispatchDtype.bf16,
    combine_dtype=CombineDtype.bf16,
):
    """Initialize one SGLang-vendored FlyDSL dispatch/combine op per config."""
    import mori.shmem as ms

    from sglang.kernels.third_party.flydsl_a2a import (
        FlyDSLDispatchCombineConfig,
        FlyDSLDispatchCombineIntraNodeOp,
    )

    world_size = get_parallel().moe_ep_size
    rank = get_parallel().moe_ep_rank
    if world_size > 8:
        raise ValueError(
            f"FlyDSL a2a is intranode-only (world_size<=8); got {world_size}"
        )

    group_name = "mori"
    try:
        torch._C._distributed_c10d._register_process_group(group_name, group.cpu_group)
    except Exception as exc:
        if "already registered" not in str(exc):
            raise
        logger.info("[FlyDSL init] process group already registered: %s", exc)
    else:
        ms.shmem_torch_process_group_init(group_name)

    scale_dim = 0
    scale_type_size = 0
    if dispatch_dtype == DispatchDtype.fp8:
        scale_dim = hidden_size // FP8_BLOCK_SIZE
        scale_type_size = torch.float32.itemsize
    elif dispatch_dtype == DispatchDtype.fp4:
        scale_dim = hidden_size // MXFP4_BLOCK_SIZE
        scale_type_size = torch.float8_e8m0fnu.itemsize

    quant_type = (
        "fp8_direct_cast" if combine_dtype == CombineDtype.fp8_direct_cast else "none"
    )
    tbo_enabled = is_tbo_enabled()
    if tbo_enabled:
        tbo_geometry = _resolve_tbo_geometry(
            tbo_enabled=True,
            shared_block_num=get_int_env_var("SGLANG_FLYDSL_TBO_BLOCK_NUM", 0),
            dispatch_block_num=get_int_env_var(
                "SGLANG_FLYDSL_TBO_DISPATCH_BLOCK_NUM", 0
            ),
            combine_block_num=get_int_env_var("SGLANG_FLYDSL_TBO_COMBINE_BLOCK_NUM", 0),
        )
    else:
        # TBO launch controls have no effect, including malformed values, when
        # two-batch overlap is disabled.
        tbo_geometry = _resolve_tbo_geometry(
            tbo_enabled=False,
            shared_block_num=0,
            dispatch_block_num=0,
            combine_block_num=0,
        )
    logger.info(
        "[FlyDSL init] world=%d rank=%d hidden=%d max_tokens=%d "
        "local_experts=%d topk=%d dispatch=%s combine=%s "
        "tbo_dispatch_block=%s tbo_dispatch_warps=%s "
        "tbo_combine_block=%s tbo_combine_warps=%s",
        world_size,
        rank,
        hidden_size,
        num_max_dispatch_tokens_per_rank,
        num_local_experts,
        router_topk,
        dispatch_dtype,
        combine_dtype,
        tbo_geometry.dispatch_block_num,
        tbo_geometry.dispatch_warp_num_per_block,
        tbo_geometry.combine_block_num,
        tbo_geometry.combine_warp_num_per_block,
    )
    return FlyDSLDispatchCombineIntraNodeOp(
        FlyDSLDispatchCombineConfig(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_size,
            max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=router_topk,
            # Allocate for the largest external row type. dispatch() still
            # specializes on its launch-time bf16/fp8/fp4 dtype.
            data_type=params_dtype,
            max_token_type_size=params_dtype.itemsize,
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            quant_type=quant_type,
            enable_std_moe=False,
            dispatch_block_num=tbo_geometry.dispatch_block_num,
            dispatch_warp_num_per_block=tbo_geometry.dispatch_warp_num_per_block,
            combine_block_num=tbo_geometry.combine_block_num,
            combine_warp_num_per_block=tbo_geometry.combine_warp_num_per_block,
            max_total_recv_tokens=get_int_env_var(
                "SGLANG_FLYDSL_PREALLOC_MAX_RECV_TOKENS", 0
            ),
        )
    )


class FlyDSLEPNormalDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: torch.Tensor
    origin_topk_ids: torch.Tensor
    origin_topk_weights: torch.Tensor
    out_dtype: torch.dtype

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


assert isinstance(FlyDSLEPNormalDispatchOutput, DispatchOutput)


class FlyDSLEPNormalCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


assert isinstance(FlyDSLEPNormalCombineInput, CombineInput)


class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class FlyDSLEPDispatcher(BaseDispatcher):
    """Plain token-major FlyDSL all-to-all for intranode expert parallelism."""

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        instance_id: int = 0,
    ):
        super().__init__()
        try:
            import flydsl  # noqa: F401
            import mori  # noqa: F401

            import aiter  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FlyDSL EP requires the aiter, flydsl, and mori packages"
            ) from exc

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.instance_id = instance_id
        self._telemetry_dispatcher_id = id(self)
        self.async_finish = async_finish
        # TBO-only control: non-overlapped paths do not read the priority env.
        self._comm_stream = _get_tbo_comm_stream(
            group, tbo_enabled=is_tbo_enabled(), async_finish=async_finish
        )
        self.num_max_dispatch_tokens_per_rank = get_int_env_var(
            "SGLANG_FLYDSL_NUM_MAX_DISPATCH_TOKENS_PER_RANK", 4096
        )

        self.dispatch_dtype = DispatchDtype.bf16
        self.combine_dtype = CombineDtype.bf16
        self._flydsl_op = None
        self._stage = _Stage.INITIAL
        self._telemetry_dispatch_pending = _flydsl_telemetry.enabled
        self._telemetry_combine_pending = _flydsl_telemetry.enabled
        self._telemetry_sync_values_pending = _flydsl_sync_values_telemetry.enabled

        self.fp8_quant_func = None
        self.fp4_quant_func = None
        self.expert_mask_gpu = None
        if _use_aiter:
            from aiter import QuantType, get_hip_quant

            self.fp8_quant_func = get_hip_quant(QuantType.per_1x128)
            self.fp4_quant_func = get_hip_quant(QuantType.per_1x32)
            if num_experts is not None and num_local_experts is not None:
                ep_rank = get_parallel().moe_ep_rank
                self.expert_mask_gpu = torch.zeros(
                    num_experts,
                    device=torch.cuda.current_device(),
                    dtype=torch.int32,
                )
                start = ep_rank * num_local_experts
                self.expert_mask_gpu[start : start + num_local_experts] = 1

    @property
    def flydsl_op(self):
        if self._flydsl_op is None:
            self._apply_dtype_overrides()
            self._flydsl_op = init_flydsl_op(
                self.group,
                self.router_topk,
                self.num_experts,
                self.num_local_experts,
                self.hidden_size,
                self.params_dtype,
                self.num_max_dispatch_tokens_per_rank,
                self.instance_id,
                self.dispatch_dtype,
                self.combine_dtype,
            )
        return self._flydsl_op

    def _apply_dtype_overrides(self):
        dispatch = os.environ.get("SGLANG_FLYDSL_DISPATCH_DTYPE", "").lower()
        if dispatch == "fp8":
            self.dispatch_dtype = DispatchDtype.fp8
        elif dispatch == "fp4":
            self.dispatch_dtype = DispatchDtype.fp4
        elif dispatch == "bf16":
            self.dispatch_dtype = DispatchDtype.bf16

        combine = os.environ.get("SGLANG_FLYDSL_COMBINE_DTYPE", "").lower()
        if combine == "fp8_direct_cast":
            self.combine_dtype = CombineDtype.fp8_direct_cast
        elif combine == "bf16":
            self.combine_dtype = CombineDtype.bf16

    def set_quant_config(self, quant_config: dict) -> None:
        super().set_quant_config(quant_config)
        weight_dtype = quant_config.get("weight_dtype")
        if weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            self.dispatch_dtype = DispatchDtype.fp8
        elif weight_dtype == torch.float4_e2m1fn_x2:
            self.dispatch_dtype = DispatchDtype.fp4
        else:
            self.dispatch_dtype = DispatchDtype.bf16
        self.combine_dtype = CombineDtype.bf16
        self._apply_dtype_overrides()

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
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        self._num_tokens = hidden_states.shape[0]
        self._op_cur_tok = hidden_states.shape[0]
        self._op_cluster_dispatch_rows = dynamic_recv_cluster_rows
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        output_dtype = hidden_states.dtype
        scale = None
        device = hidden_states.device

        if self.dispatch_dtype == DispatchDtype.fp8 and self.fp8_quant_func:
            from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

            if self._num_tokens:
                hidden_states, scale = self.fp8_quant_func(
                    hidden_states, quant_dtype=fp8_dtype
                )
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size), dtype=fp8_dtype, device=device
                )
                scale = torch.empty(
                    (0, self.hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
        elif self.dispatch_dtype == DispatchDtype.fp4 and self.fp4_quant_func:
            if self._num_tokens:
                hidden_states, scale = self.fp4_quant_func(hidden_states, shuffle=False)
            else:
                hidden_states = torch.empty(
                    (0, self.hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=device,
                )
                scale = torch.empty(
                    (0, self.hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=device,
                )

        # Keep all dispatch inputs ready before the comm stream consumes them.
        topk_weights = topk_weights.to(torch.float32)
        ready_event = None
        if self._comm_stream is not None:
            ready_event = torch.cuda.Event(blocking=False, interprocess=False)
            ready_event.record(torch.cuda.current_stream())

        self._dispatch_intermediate_state = (
            hidden_states,
            topk_weights,
            topk_ids,
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

        op = self.flydsl_op
        recv_cap = self._resolve_dynamic_recv_cap(
            op.cfg.effective_max_recv,
            eager_cluster_rows=self._op_cluster_dispatch_rows,
        )
        self._op_recv_cap = recv_cap
        if self._comm_stream is None:
            out_tok, out_wts, out_scales, out_idx, total_recv = op.dispatch(
                hidden_states,
                topk_weights,
                scale,
                topk_ids,
                recv_cap=recv_cap,
            )
        else:
            compute_stream = torch.cuda.current_stream()
            # Event + Python-ref lifetime: avoid record_stream(comm), whose
            # deferred frees caused the historical DP-TBO allocator fragmentation.
            keepalive = (hidden_states, topk_weights, topk_ids, scale)
            with torch.cuda.stream(self._comm_stream):
                assert ready_event is not None
                self._comm_stream.wait_event(ready_event)
                out_tok, out_wts, out_scales, out_idx, total_recv = op.dispatch(
                    hidden_states,
                    topk_weights,
                    scale,
                    topk_ids,
                    recv_cap=recv_cap,
                )
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(self._comm_stream)
            compute_stream.wait_event(done_event)
            # The wait is now ordered before these compute-stream-owned tensors
            # can be reclaimed.
            del keepalive
        if self._telemetry_sync_values_pending:
            self._telemetry_sync_values_pending = False
            should_sync_recv_values = _should_sync_recv_values(
                telemetry_enabled=_flydsl_telemetry.enabled,
                sync_values_enabled=_flydsl_sync_values_telemetry.enabled,
                pending=True,
            )
        else:
            should_sync_recv_values = False
        if should_sync_recv_values:
            # This runs after the compute stream's comm-stream wait. The final
            # cpu() is the probe's sole host synchronization and perturbs timing.
            parallel = get_parallel()
            recv_counts_per_expert, actual_total_recv = _recv_count_values(
                out_idx,
                total_recv,
                local_expert_start=parallel.moe_ep_rank * self.num_local_experts,
                num_local_experts=self.num_local_experts,
            )
            _flydsl_sync_values_telemetry.log(
                ("dispatch_recv_values", self._telemetry_dispatcher_id),
                "dispatch_recv_values",
                global_rank=parallel.world_rank,
                moe_ep_rank=parallel.moe_ep_rank,
                dispatcher_id=self._telemetry_dispatcher_id,
                child_id=self.instance_id,
                physical_recv_cap=op.cfg.effective_max_recv,
                effective_recv_cap=recv_cap,
                recv_counts_per_expert=recv_counts_per_expert,
                actual_total_recv=actual_total_recv,
                sync_values_diagnostic_perturbs_timing=True,
                sync_values_not_for_performance_benchmark_traces=True,
            )
        if self._telemetry_dispatch_pending:
            dispatch_blocks, dispatch_warps = _resolved_geometry_from_cache(
                op, "dispatch"
            )
            stream_id, stream_handle, stream_priority = _stream_telemetry(
                self._comm_stream
            )
            _flydsl_telemetry.log(
                ("dispatch", self._telemetry_dispatcher_id),
                "dispatch",
                dispatcher_id=self._telemetry_dispatcher_id,
                instance_id=self.instance_id,
                child_id=self.instance_id,
                local_input_rows=self._op_cur_tok,
                cluster_dispatch_rows=self._op_cluster_dispatch_rows,
                physical_recv_cap=op.cfg.effective_max_recv,
                effective_recv_cap=recv_cap,
                dispatched_output_rows=out_tok.shape[0],
                total_recv=total_recv,
                configured_dispatch_blocks=op.cfg.dispatch_block_num,
                configured_dispatch_warps_per_block=(
                    op.cfg.dispatch_warp_num_per_block
                ),
                configured_combine_blocks=op.cfg.combine_block_num,
                configured_combine_warps_per_block=(op.cfg.combine_warp_num_per_block),
                dispatch_blocks=dispatch_blocks,
                dispatch_warps_per_block=dispatch_warps,
                comm_stream_id=stream_id,
                comm_stream_handle=stream_handle,
                comm_stream_priority=stream_priority,
            )
            self._telemetry_dispatch_pending = False
        self._recv_topk_ids = out_idx
        return FlyDSLEPNormalDispatchOutput(
            hidden_states=out_tok,
            hidden_states_scale=out_scales,
            topk_ids=out_idx,
            topk_weights=out_wts,
            num_recv_tokens_per_expert=total_recv,
            origin_topk_ids=topk_ids,
            origin_topk_weights=topk_weights,
            out_dtype=output_dtype,
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        self.combine_a(combine_input)
        return self.combine_b()[: self._num_tokens]

    def combine_a(self, combine_input: CombineInput):
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
            out_tok, _ = self.flydsl_op.combine(
                hidden_states,
                None,
                self._recv_topk_ids,
                cur_tok=self._op_cur_tok,
                recv_cap=self._op_recv_cap,
            )
        else:
            compute_stream = torch.cuda.current_stream()
            keepalive = (
                hidden_states,
                _topk_ids,
                _topk_weights,
                self._recv_topk_ids,
            )
            with torch.cuda.stream(self._comm_stream):
                assert ready_event is not None
                self._comm_stream.wait_event(ready_event)
                out_tok, _ = self.flydsl_op.combine(
                    hidden_states,
                    None,
                    self._recv_topk_ids,
                    cur_tok=self._op_cur_tok,
                    recv_cap=self._op_recv_cap,
                )
                done_event = torch.cuda.Event(blocking=False, interprocess=False)
                done_event.record(self._comm_stream)
            compute_stream.wait_event(done_event)
            del keepalive
        if self._telemetry_combine_pending:
            combine_blocks, combine_warps = _resolved_geometry_from_cache(
                self.flydsl_op, "combine"
            )
            _flydsl_telemetry.log(
                ("combine", self._telemetry_dispatcher_id),
                "combine",
                dispatcher_id=self._telemetry_dispatcher_id,
                instance_id=self.instance_id,
                child_id=self.instance_id,
                combine_input_rows=hidden_states.shape[0],
                effective_recv_cap=self._op_recv_cap,
                configured_dispatch_blocks=self.flydsl_op.cfg.dispatch_block_num,
                configured_dispatch_warps_per_block=(
                    self.flydsl_op.cfg.dispatch_warp_num_per_block
                ),
                configured_combine_blocks=self.flydsl_op.cfg.combine_block_num,
                configured_combine_warps_per_block=(
                    self.flydsl_op.cfg.combine_warp_num_per_block
                ),
                combine_blocks=combine_blocks,
                combine_warps_per_block=combine_warps,
            )
            self._telemetry_combine_pending = False
        return out_tok

    def _resolve_dynamic_recv_cap(
        self, physical_cap: int, eager_cluster_rows: Optional[int] = None
    ) -> int:
        if eager_cluster_rows is not None and get_bool_env_var(
            "SGLANG_FLYDSL_DYNAMIC_RECV_CAP_EAGER", "true"
        ):
            eager_cap = _resolve_eager_recv_cap(int(eager_cluster_rows), physical_cap)
            if eager_cap is not None:
                return eager_cap
        if not get_bool_env_var("SGLANG_FLYDSL_DYNAMIC_RECV_CAP", "false"):
            return physical_cap
        try:
            from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
            from sglang.srt.model_executor.runner import get_is_capture_mode
        except Exception:
            return physical_cap
        if not get_is_capture_mode():
            return physical_cap
        dp_global = get_dp_global_num_tokens()
        if dp_global is None or len(dp_global) <= 1:
            return physical_cap
        global_capacity = max(int(n) for n in dp_global) * len(dp_global)
        if global_capacity <= 0:
            return physical_cap
        recv_cap = max(32, 1 << (global_capacity - 1).bit_length())
        return min(physical_cap, recv_cap)

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage, f"stage {self._stage} != expected {old_stage}"
        self._stage = new_stage

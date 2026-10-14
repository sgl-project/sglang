"""Step5 KV capacity accounting and construction checks."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Literal

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.pool_configurator import (
    HybridSWAPoolConfigurator,
    MemoryPoolConfig,
    MemoryPoolConfigurator,
    SWARequestCapPoolConfigurator,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool

logger = logging.getLogger(__name__)

_STEP5_INT32_MAX = (1 << 31) - 1

_STEP5_INT64_MAX = (1 << 63) - 1

_Step5Phase = Literal["profiled", "token_constrained"]

_Step5Scalar = str | int | float | bool | None

_Step5ModeInputs = tuple[tuple[str, _Step5Scalar], ...]


def _step5_integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= _STEP5_INT64_MAX:
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {_STEP5_INT64_MAX}]"
        )


def _step5_bytes(value: int) -> int:
    if value > _STEP5_INT64_MAX:
        raise OverflowError("Step5 KV byte accounting exceeds signed int64")
    return value


def _step5_capacity(name: str, value: int, page_size: int) -> None:
    _step5_integer(name, value, 1)
    if value % page_size:
        raise ValueError(f"{name} must contain complete pages")
    if value + page_size - 1 > _STEP5_INT32_MAX:
        raise ValueError(f"{name} including the dummy page exceeds int32 addresses")


class Step5KVReservations(msgspec.Struct, frozen=True, kw_only=True):
    """Mandatory additional bounds, disjoint from persistent payload/upstream.

    Explicit zero is an assumption, not evidence of zero producer workspace.
    Each bound must cover fixed B/row capacity and both pool capacity ceilings.
    Constructor minimum checks below prove only known individual lower bounds;
    the caller must still supply a bound for the complete constructor lifetime.
    """

    dsa_workspace_bytes: int
    constructor_extra_bytes: int
    allocator_runtime_extra_bytes: int
    non_dsa_runtime_extra_bytes: int
    allocation_rounding_extra_bytes: int
    full_capacity_ceiling: int
    swa_capacity_ceiling: int

    def __post_init__(self) -> None:
        for name, value in (
            ("dsa_workspace_bytes", self.dsa_workspace_bytes),
            ("constructor_extra_bytes", self.constructor_extra_bytes),
            ("allocator_runtime_extra_bytes", self.allocator_runtime_extra_bytes),
            ("non_dsa_runtime_extra_bytes", self.non_dsa_runtime_extra_bytes),
            ("allocation_rounding_extra_bytes", self.allocation_rounding_extra_bytes),
        ):
            _step5_integer(name, value)
        _step5_capacity("full_capacity_ceiling", self.full_capacity_ceiling, 64)
        _step5_capacity("swa_capacity_ceiling", self.swa_capacity_ceiling, 64)
        # mapping cat overlaps its final storage with zeros + sentinel inputs.
        # Each child clear may also overlap one old/new free-page array. These
        # are necessary individual bounds, not a proved simultaneous peak.
        minimum = max(
            8 * (self.full_capacity_ceiling + 64 + 1),
            8 * (self.full_capacity_ceiling // 64),
            8 * (self.swa_capacity_ceiling // 64),
        )
        if self.constructor_extra_bytes < minimum:
            raise ValueError("constructor reserve is below a known individual cost")
        _step5_bytes(self.total_bytes)

    @property
    def total_bytes(self) -> int:
        return (
            self.dsa_workspace_bytes
            + self.constructor_extra_bytes
            + self.allocator_runtime_extra_bytes
            + self.non_dsa_runtime_extra_bytes
            + self.allocation_rounding_extra_bytes
        )


class Step5KVGeometry(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved metadata snapshot, not an alternate model configuration."""

    target_full_ids: tuple[int, ...]
    target_swa_ids: tuple[int, ...]
    draft_swa_ids: tuple[int, ...]
    hidden_size: int
    local_kv_heads: int
    k_head_dim: int
    v_head_dim: int
    page_size: int
    region_size: int
    index_head_dim: int
    index_k_heads: int
    sliding_window: int
    full_kv_fp8: bool = False


class Step5RequestCapacity(msgspec.Struct, frozen=True, kw_only=True):
    max_running_requests: int
    context_len: int
    extra_context_len: int
    index_ring_size: int
    # Parsed once; PD slots belong to the same final request capacity.
    extra_slots: int = 0

    @property
    def row_capacity(self) -> int:
        return self.context_len + self.extra_context_len


_STEP5_MIB = 1 << 20


@dataclass(kw_only=True)
class Step5MemoryPoolConfig(MemoryPoolConfig):
    """Final native capacities, allocation costs and Step5 layout constraints.

    Target and draft runners share this exact configuration. There is no
    separate payload/result conversion on the allocation path.
    """

    geometry: Step5KVGeometry
    requests: Step5RequestCapacity
    reservations: Step5KVReservations
    index_state_bytes: int
    draft_boundary_stash_bytes: int
    persistent_payload_bytes: int
    budget_bytes: int = 0
    mode_inputs: _Step5ModeInputs = ()

    @property
    def num_req_slots(self) -> int:
        return self.max_running_requests + self.requests.extra_slots + 1

    @property
    def additional_reserved_bytes(self) -> int:
        return self.reservations.total_bytes

    @property
    def charged_bytes(self) -> int:
        value = self.persistent_payload_bytes + self.reservations.total_bytes
        if value > (1 << 63) - 1:
            raise OverflowError("Step5 KV byte accounting exceeds signed int64")
        return value

    @property
    def host_request_generation_bytes(self) -> int:
        return 8 * self.num_req_slots

    @property
    def remaining_bytes(self) -> int:
        return self.budget_bytes - self.charged_bytes


def _step5_full_kv_fp8() -> bool:
    return get_model().kv_cache_dtype == "fp8_e4m3"


def _create_capacity_policy(kvc):
    if SWARequestCapPoolConfigurator.is_applicable(kvc):
        policy = SWARequestCapPoolConfigurator(kvc)
    else:
        policy = HybridSWAPoolConfigurator(kvc)
    if _step5_full_kv_fp8():
        policy._full_per_token = _step5_full_kv_bytes_per_token(
            kvc.model_config.get_num_kv_heads(get_parallel().attn_tp_size),
            kvc.model_config.head_dim,
            kvc.model_config.v_head_dim,
            fp8=True,
        )
        policy._recompute_cell_size()
    return policy


def _step5_full_kv_bytes_per_token(heads, k_dim, v_dim, *, fp8):
    return heads * (k_dim + v_dim + 8) if fp8 else 2 * heads * (k_dim + v_dim)


def _capacity_from_full_tokens(policy, full_tokens, page_size, *, phase):
    if phase not in ("profiled", "token_constrained"):
        raise ValueError(f"Unknown SWA capacity sizing phase: {phase!r}")
    if phase == "profiled" and isinstance(policy, SWARequestCapPoolConfigurator):
        # Profiling retains the independent SWA cap; token constraints use the
        # native method below, including its unrounded SWA token limit.
        full = full_tokens // page_size * page_size
        swa = (policy.swa_token_cap + page_size - 1) // page_size * page_size
        return MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
        )
    return policy.calculate_pool_sizes_from_max_tokens(full_tokens, page_size)


class Step5PoolConfigurator(MemoryPoolConfigurator):
    """Native sizing policy including compressed index keys, raw ring and MTP buffer costs."""

    def __init__(self, kvc):
        self.kvc = kvc
        self.config = None

    def calculate_pool_sizes(self, available_bytes: int, page_size: int):
        self.capacity_policy = _create_capacity_policy(self.kvc)
        self.config, self.minimum = self._profile_capacity(available_bytes)
        self.config.mem_fraction_static = get_schedule().mem_fraction_static
        return self.config

    def _profile_capacity(self, budget_bytes: int):
        """Keep the native profiled/constrained policies and attach a priced payload.

        The upstream profiler already subtracts loaded weights and runtime slack.
        Additional allowances below are conservative startup headroom; actual peak
        and long-context workspace qualification remain required before production.
        This entry serves target operation with native device-cache owners and an
        MTP with native device prefix caching and scheduler overlap. MTP uses eager
        prefill and either eager or full-graph target verify and draft extend.
        Overlap uses the scheduler's shared-read barrier: Step5 declares POST_REPLAY,
        so shared mappings cannot be rewritten before the forward finishes reading.
        The general Step5 solver retains the wider cache/MTP sizing contracts.
        """

        kvc, policy = self.kvc, self.capacity_policy
        schedule = get_schedule()
        page = kvc.page_size
        mtp = self._validate_startup_mode(page=page)
        native = policy.calculate_pool_sizes(budget_bytes, page)
        upper = native.max_total_num_tokens
        # The native resolver handles both explicit per-DP limits and automatic
        # concurrency. Reserve workspace for the upper bound; the solver prices
        # request-owned buffers at each candidate's actual native concurrency.
        requests = kvc.resolve_max_num_reqs(upper)
        if requests < 1:
            raise ValueError(
                "Step5 capacity must admit at least one request per DP rank"
            )
        minimum = (
            page
            if schedule.max_running_requests is None
            else ((2 * requests + page - 1) // page) * page
        )
        if not SWARequestCapPoolConfigurator.is_applicable(kvc):
            swa_pages = kvc.sliding_window_size // page + 2
            ratio = schedule.swa_full_tokens_ratio
            if ratio <= 0:
                raise ValueError("SWA capacity ratio must be positive")
            swa_minimum = math.ceil(swa_pages / ratio) * page
            while int(swa_minimum * ratio) // page < swa_pages:
                swa_minimum += page
            minimum = max(minimum, swa_minimum)
        if upper < minimum:
            raise ValueError("Step5 KV budget cannot hold the requested concurrency")
        tokens = (
            max(schedule.max_prefill_tokens, schedule.chunked_prefill_size) + requests
        )
        mtp_workspace = (
            Step5PoolConfigurator._mtp_workspace_allowance(
                tokens=max(tokens, 4 * requests),
                requests=requests,
                hidden=kvc.model_config.hidden_size,
                vocab=kvc.model_config.vocab_size,
                images=kvc.model_config.is_multimodal,
            )
            if mtp
            else 0
        )
        reservations = Step5KVReservations(
            dsa_workspace_bytes=Step5PoolConfigurator._workspace_allowance(
                context=kvc.model_config.context_len,
                tokens=tokens,
                requests=requests,
            ),
            constructor_extra_bytes=64 * _STEP5_MIB
            + 16 * (upper + native.swa_max_total_num_tokens + 128),
            allocator_runtime_extra_bytes=64 * _STEP5_MIB + 32 * tokens,
            non_dsa_runtime_extra_bytes=512 * _STEP5_MIB + mtp_workspace,
            # Conservative independent-segment allowance for native K/V buffers
            # and the small number of whole-model DSA/request/mapping allocations.
            allocation_rounding_extra_bytes=20
            * _STEP5_MIB
            * (2 * kvc.layer_info.num_effective_layers + 16),
            full_capacity_ceiling=upper,
            swa_capacity_ceiling=native.swa_max_total_num_tokens,
        )
        result = self.solve(
            kvc=kvc,
            budget_bytes=budget_bytes,
            max_running_requests=requests,
            reservations=reservations,
            min_full_tokens=minimum,
            max_full_tokens=upper,
            policy=policy,
        )
        logger.info(
            "Step5 profiled KV budget: available=%d persistent=%d additional_reserved=%d "
            "FULL=%d SWA=%d requests=%d",
            result.budget_bytes,
            result.persistent_payload_bytes,
            result.additional_reserved_bytes,
            result.full_max_total_num_tokens,
            result.swa_max_total_num_tokens,
            result.requests.max_running_requests,
        )
        return result, minimum

    def calculate_pool_sizes_from_max_tokens(self, max_total_num_tokens, page_size):

        # The common KV entry point applies user/PP constraints after profiling.
        # Keep the same parsed geometry and reservation domain; do not re-solve
        # or spend the memory released by a cap.
        tokens = max_total_num_tokens // page_size * page_size
        if tokens < self.minimum:
            raise ValueError(
                "external token cap is below the declared workload minimum"
            )
        native = self.capacity_policy.calculate_pool_sizes_from_max_tokens(
            tokens, page_size
        )
        result = self.config
        payload = Step5PoolConfigurator.price_pool(
            native,
            result.geometry,
            msgspec.structs.replace(
                result.requests,
                max_running_requests=self.kvc.resolve_max_num_reqs(tokens),
            ),
            result.reservations,
        )
        if payload.charged_bytes > result.budget_bytes:
            raise ValueError("final native constrained payload exceeds the KV budget")
        self.config = replace(
            payload, budget_bytes=result.budget_bytes, mode_inputs=result.mode_inputs
        )
        self.config.mem_fraction_static = get_schedule().mem_fraction_static
        return self.config

    @staticmethod
    def _parallel_inputs() -> tuple[tuple[str, int], ...]:
        parallel = get_parallel()
        return (
            ("tp_size", parallel.tp_size),
            ("moe_ep_size", parallel.moe_ep_size),
            ("attn_tp_size", parallel.attn_tp_size),
            ("pp_size", parallel.pp_size),
            ("dp_size", parallel.dp_size),
            ("attn_dp_size", parallel.attn_dp_size),
            ("attn_cp_size", parallel.attn_cp_size),
            ("attn_dcp_size", parallel.attn_dcp_size),
        )

    @staticmethod
    def _mode_inputs() -> _Step5ModeInputs:
        schedule, spec, memory = get_schedule(), get_spec(), get_memory()
        return (
            ("max_running_requests", schedule.max_running_requests),
            ("page_size", schedule.page_size),
            ("chunked_prefill_size", schedule.chunked_prefill_size),
            ("max_prefill_tokens", schedule.max_prefill_tokens),
            ("disable_overlap_schedule", schedule.disable_overlap_schedule),
            ("disable_radix_cache", memory.disable_radix_cache),
            ("enable_unified_memory", memory.enable_unified_memory),
            ("enable_page_major_kv_layout", memory.enable_page_major_kv_layout),
            ("speculative_algorithm", spec.speculative_algorithm),
            ("speculative_num_steps", spec.speculative_num_steps),
            ("speculative_eagle_topk", spec.speculative_eagle_topk),
            ("speculative_num_draft_tokens", spec.speculative_num_draft_tokens),
            ("speculative_adaptive", spec.speculative_adaptive),
            ("disaggregation_mode", get_disagg().disaggregation_mode),
            (
                "disaggregation_transfer_backend",
                get_disagg().disaggregation_transfer_backend,
            ),
            (
                "disaggregation_decode_extra_slots",
                get_disagg().disaggregation_decode_extra_slots,
            ),
            ("swa_eviction_interval", envs.SGLANG_SWA_EVICTION_INTERVAL.get()),
            ("hnd_kvcache", envs.SGLANG_USE_HND_KVCACHE.get()),
            ("step5_fp8_full_kv", _step5_full_kv_fp8()),
            ("aiter_kv_cache_layout", envs.SGLANG_AITER_KV_CACHE_LAYOUT.get()),
            ("swa_full_tokens_ratio", schedule.swa_full_tokens_ratio),
        ) + Step5PoolConfigurator._parallel_inputs()

    @staticmethod
    def _validate_runtime(
        *, kvc: KVCacheConfigurator, config: Step5MemoryPoolConfig
    ) -> None:
        """Reject drift when either role consumes the target's frozen budget.

        This validates construction inputs, not target geometry on a draft worker.
        Target geometry is resolved once before byte accounting.
        """
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        if not isinstance(config, Step5MemoryPoolConfig):
            raise ValueError("an explicit resolved Step5 KV budget is required")
        if Step5PoolConfigurator._mode_inputs() != config.mode_inputs:
            raise ValueError("runtime inputs differ from the resolved Step5 KV budget")
        if kvc.spec_algorithm is not SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        ):
            raise ValueError(
                "worker speculative algorithm differs from the Step5 KV budget"
            )
        Step5PoolConfigurator._validate_parallel_topology(kvc)

    @staticmethod
    def _validate_parallel_topology(kvc: KVCacheConfigurator) -> None:
        """The worker and runtime must agree on one supported topology."""

        ps = get_parallel()
        declared = (
            ps.tp_size,
            ps.moe_ep_size,
            ps.attn_tp_size,
            ps.pp_size,
            ps.dp_size,
            ps.attn_dp_size,
            ps.attn_cp_size,
            ps.attn_dcp_size,
        )
        actual = tuple(value for _, value in Step5PoolConfigurator._parallel_inputs())
        for value in declared + actual:
            _step5_integer("parallel dimension", value, 1)
        if (
            declared
            not in (
                (4, 1, 4, 1, 1, 1, 1, 1),
                (4, 4, 4, 1, 1, 1, 1, 1),
                (8, 8, 8, 1, 1, 1, 1, 1),
                (8, 8, 4, 1, 1, 2, 1, 1),
            )
            or actual != declared
        ):
            raise ValueError(
                "runtime parallel topology must match TP4/EP1, TP4/EP4, TP8/EP8, or "
                "DP2/TP4/EP8 with PP1/CP1"
            )

    @staticmethod
    def _validate_worker_profile(
        kvc: KVCacheConfigurator,
        max_running_requests: int,
    ) -> None:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.step5 import Step5Config
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        if not isinstance(kvc, KVCacheConfigurator):
            raise ValueError("a real target KVCacheConfigurator is required")
        _step5_integer("max_running_requests", max_running_requests, 1)
        if max_running_requests > _STEP5_INT32_MAX:
            raise ValueError("request slot IDs exceed int32")
        mc = kvc.model_config
        if not isinstance(mc, ModelConfig) or not isinstance(
            mc.hf_text_config, Step5Config
        ):
            raise ValueError("a resolved target Step5 ModelConfig is required")
        hf = mc.hf_text_config
        if mc.is_draft_model or hf.architectures != ["Step5ForCausalLM"]:
            raise ValueError("a complete target Step5 ModelConfig is required")
        Step5PoolConfigurator._validate_parallel_topology(kvc)
        ps = get_parallel()
        schedule, memory, spec = get_schedule(), get_memory(), get_spec()
        if (
            kvc.is_draft_worker
            or kvc.post_capture_kv_active
            or memory.enable_unified_memory
            or memory.enable_page_major_kv_layout
            or envs.SGLANG_USE_HND_KVCACHE.get()
            or envs.SGLANG_AITER_KV_CACHE_LAYOUT.get() != "nhd"
            or spec.speculative_adaptive
            or not kvc.is_hybrid_swa
            or kvc.is_hybrid_swa_compress
            or kvc.use_mla_backend
            or kvc.mambaish_config is not None
            or kvc.kv_cache_dtype != torch.bfloat16
            or kvc.kv_cache_dtype_str not in (None, "auto", "bf16", "bfloat16")
            or kvc.model_dtype != torch.bfloat16
            or kvc.device not in ("cpu", "cuda")
        ):
            raise ValueError("unsupported Step5 target BF16 NHD sizing mode")
        disagg = get_disagg()
        pd_mtp_mode = (
            spec.speculative_algorithm in ("EAGLE", "EAGLE3")
            and spec.enable_multi_layer_eagle
            and not spec.speculative_use_rejection_sampling
            and (
                spec.speculative_num_steps,
                spec.speculative_eagle_topk,
                spec.speculative_num_draft_tokens,
            )
            == (3, 1, 4)
        )
        if (
            _step5_full_kv_fp8()
            and disagg.disaggregation_mode != "null"
            and disagg.disaggregation_transfer_backend != "mooncake"
        ):
            raise ValueError("Step5 FP8 FULL KV PD requires the mooncake backend")
        if disagg.disaggregation_mode != "null" and (
            disagg.disaggregation_transfer_backend not in ("mooncake", "nixl")
            or (spec.speculative_algorithm is not None and not pd_mtp_mode)
            or (
                memory.enable_hierarchical_cache
                and disagg.disaggregation_mode == "decode"
            )
        ):
            raise ValueError("unsupported Step5 PD sizing mode")
        if (
            kvc.page_size != 64
            or schedule.page_size != kvc.page_size
            or (
                schedule.max_running_requests is not None
                and schedule.max_running_requests // ps.attn_dp_size
                != max_running_requests
            )
            or kvc.sliding_window_size != 512
            or hf.sliding_window != kvc.sliding_window_size
            or (kvc.layer_info.start_layer, kvc.layer_info.end_layer) != (0, 92)
            or kvc.layer_info.num_effective_layers != 92
            or hf.num_hidden_layers != 92
            or hf.num_nextn_predict_layers != 3
        ):
            raise ValueError(
                "Step5 sizing requires explicit B, P64, W512 and all 92 layers"
            )

    @staticmethod
    def _resolve_geometry(kvc: KVCacheConfigurator) -> Step5KVGeometry:
        from sglang.srt.configs.step5 import parse_step5_index_geometry

        mc, ps = kvc.model_config, get_parallel()
        hf = mc.hf_text_config
        full, swa = (
            tuple(mc.full_attention_layer_ids),
            tuple(mc.swa_attention_layer_ids),
        )
        for layer_id in full + swa:
            _step5_integer("resolved layer ID", layer_id)
        if (
            len(full) != 23
            or len(swa) != 69
            or len(set(full + swa)) != 92
            or set(full + swa) != set(range(92))
            or hf.layer_types_with_mtp is None
            or len(hf.layer_types_with_mtp) != 95
            or any(hf.layer_types_with_mtp[i] != "full_attention" for i in full)
            or any(hf.layer_types_with_mtp[i] != "sliding_attention" for i in swa)
            or hf.layer_types_with_mtp[92:] != ["sliding_attention"] * 3
        ):
            raise ValueError(
                "invalid resolved target partition or three draft SWA layers"
            )
        heads = mc.get_num_kv_heads(ps.attn_tp_size, ps.attn_dcp_size)
        for name, value in (
            ("page size", kvc.page_size),
            ("window", kvc.sliding_window_size),
            ("model window", hf.sliding_window),
            ("model target depth", hf.num_hidden_layers),
            ("model MTP depth", hf.num_nextn_predict_layers),
            ("model hidden size", hf.hidden_size),
            ("local FULL KV heads", heads),
            ("FULL K width", mc.head_dim),
            ("FULL V width", mc.v_head_dim),
            ("local SWA KV heads", mc.get_swa_num_kv_heads(ps.attn_tp_size)),
            ("SWA K width", mc.swa_head_dim),
            ("SWA V width", mc.swa_v_head_dim),
        ):
            _step5_integer(name, value, 1)
        if (heads, mc.head_dim, mc.v_head_dim) != (1, 192, 192) or (
            mc.get_swa_num_kv_heads(ps.attn_tp_size),
            mc.swa_head_dim,
            mc.swa_v_head_dim,
        ) != (heads, mc.head_dim, mc.v_head_dim):
            raise ValueError("unsupported resolved Step5 FULL/SWA KV geometry")
        sparse = hf.sparse_config
        if (
            not isinstance(sparse, dict)
            or sparse["enabled"] is not True
            or sparse["apply_to_layer_types"] != ["full_attention"]
        ):
            raise ValueError("DSA layer IDs must equal the resolved target FULL IDs")
        index = parse_step5_index_geometry(hf)
        if kvc.page_size != 64:
            raise ValueError("Step5 requires page64")
        return Step5KVGeometry(
            target_full_ids=full,
            target_swa_ids=swa,
            draft_swa_ids=(
                tuple(
                    range(
                        hf.num_hidden_layers,
                        hf.num_hidden_layers + hf.num_nextn_predict_layers,
                    )
                )
                if not kvc.spec_algorithm.is_none()
                else ()
            ),
            local_kv_heads=heads,
            hidden_size=hf.hidden_size,
            k_head_dim=mc.head_dim,
            v_head_dim=mc.v_head_dim,
            page_size=kvc.page_size,
            region_size=index.region_size,
            index_head_dim=index.index_head_dim,
            index_k_heads=index.index_k_heads,
            sliding_window=kvc.sliding_window_size,
            full_kv_fp8=_step5_full_kv_fp8(),
        )

    @staticmethod
    def _resolve_requests(
        kvc: KVCacheConfigurator,
        max_running_requests: int,
    ) -> Step5RequestCapacity:
        from sglang.srt.mem_cache.allocation_sizing import (
            get_req_to_token_extra_context_len,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        spec, mc = get_spec(), kvc.model_config
        algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
        if algo is not kvc.spec_algorithm:
            raise ValueError("resolved speculative algorithm disagrees with the worker")
        if algo is SpeculativeAlgorithm.NONE:
            if kvc.spec_aux_config.eagle_draft_num_layers is not None:
                _step5_integer(
                    "inactive MTP depth", kvc.spec_aux_config.eagle_draft_num_layers
                )
            if kvc.spec_aux_config.eagle_draft_num_layers not in (None, 0):
                raise ValueError("MTP-off must not reserve active draft layers")
            if any(
                value is not None
                for value in (
                    spec.speculative_num_steps,
                    spec.speculative_eagle_topk,
                    spec.speculative_num_draft_tokens,
                )
            ):
                raise ValueError("MTP-off requires unset speculative sizing fields")
            index_ring_size = 8
        elif (
            algo in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3)
            and (
                spec.speculative_num_steps,
                spec.speculative_eagle_topk,
                spec.speculative_num_draft_tokens,
            )
            == (3, 1, 4)
            and kvc.spec_aux_config.eagle_draft_num_layers == 3
        ):
            from sglang.srt.mem_cache.step5_kv_pool import get_step5_ring_capacity

            index_ring_size = get_step5_ring_capacity(spec.speculative_num_draft_tokens)
        else:
            raise ValueError(
                "only MTP-off or fixed three-step/top1/four-row MTP is supported"
            )
        if not kvc.spec_algorithm.is_none():
            for name, value in (
                ("MTP steps", spec.speculative_num_steps),
                ("MTP topk", spec.speculative_eagle_topk),
                ("MTP tokens", spec.speculative_num_draft_tokens),
                ("MTP layers", kvc.spec_aux_config.eagle_draft_num_layers),
            ):
                _step5_integer(name, value, 1)
        _step5_integer("context_len", mc.context_len, 1)
        if mc.context_len > 1 << 20:
            raise ValueError("Step5 payload profile supports context length up to 1M")
        extra = get_req_to_token_extra_context_len()
        _step5_integer("request extra context", extra)
        row = mc.context_len + extra
        if row > _STEP5_INT32_MAX:
            raise ValueError("request row capacity exceeds int32")
        return Step5RequestCapacity(
            max_running_requests=max_running_requests,
            context_len=mc.context_len,
            extra_context_len=extra,
            index_ring_size=index_ring_size,
            extra_slots=(
                int(get_disagg().disaggregation_decode_extra_slots or 0)
                if get_disagg().disaggregation_mode == "decode"
                else 0
            ),
        )

    @staticmethod
    def _resolve(
        kvc: KVCacheConfigurator,
        max_running_requests: int,
    ) -> tuple[Step5KVGeometry, Step5RequestCapacity, _Step5ModeInputs]:
        mode = Step5PoolConfigurator._mode_inputs()
        Step5PoolConfigurator._validate_worker_profile(kvc, max_running_requests)
        requests = Step5PoolConfigurator._resolve_requests(kvc, max_running_requests)
        geometry = Step5PoolConfigurator._resolve_geometry(kvc)
        if Step5PoolConfigurator._mode_inputs() != mode:
            raise RuntimeError("runtime sizing inputs changed during resolution")
        return geometry, requests, mode

    @staticmethod
    def _validate_startup_mode(*, page: int) -> bool:
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        schedule, spec = get_schedule(), get_spec()
        if (
            (
                schedule.max_running_requests is not None
                and schedule.max_running_requests < 1
            )
            or schedule.chunked_prefill_size is None
            or schedule.chunked_prefill_size <= 0
        ):
            raise ValueError(
                "Step5 startup requires positive request capacity and chunked prefill"
            )
        # Speculative chain shape and PD compatibility are validated once by the
        # budget resolver. This boundary only checks execution/workspace needs.
        mtp = spec.speculative_algorithm is not None
        if page != 64:
            raise ValueError("Step5 startup requires page64")
        if mtp and (
            not spec.enable_multi_layer_eagle
            or schedule.enable_mixed_chunk
            or schedule.chunked_prefill_size < page + 3
            or not (
                check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
                or check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
            )
            or not (
                check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                or check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
            )
        ):
            raise ValueError(
                "Step5 MTP startup requires fixed multi-layer MTP3/top1/width4, eager or breakable prefill "
                "and eager/full-graph decode, "
                "unmixed batches and chunked-prefill-size >= 67"
            )
        return mtp

    @staticmethod
    def quote(
        *,
        kvc: KVCacheConfigurator,
        native_config: MemoryPoolConfig,
        max_running_requests: int,
        reservations: Step5KVReservations,
    ) -> Step5MemoryPoolConfig:
        """Price a supplied native config using real resolved metadata; no search."""

        geometry, requests, mode = Step5PoolConfigurator._resolve(
            kvc, max_running_requests
        )
        payload = Step5PoolConfigurator.price_pool(
            native_config, geometry, requests, reservations
        )
        if (
            kvc.resolve_max_num_reqs(payload.full_max_total_num_tokens)
            != max_running_requests
        ):
            raise ValueError("native request resolution changed the declared B")
        if Step5PoolConfigurator._mode_inputs() != mode:
            raise RuntimeError("runtime sizing inputs changed while quoting")
        return payload

    @staticmethod
    def solve(
        *,
        kvc: KVCacheConfigurator,
        budget_bytes: int,
        max_running_requests: int,
        reservations: Step5KVReservations,
        min_full_tokens: int,
        max_full_tokens: int,
        external_token_cap: int | None = None,
        policy: HybridSWAPoolConfigurator | None = None,
    ) -> Step5MemoryPoolConfig:
        """Resolve once, then apply native capacity policy to pure byte accounting."""

        geometry, requests, mode = Step5PoolConfigurator._resolve(
            kvc, max_running_requests
        )
        if policy is None:
            policy = _create_capacity_policy(kvc)
        payload = Step5PoolConfigurator.solve_capacity(
            geometry=geometry,
            requests=requests,
            policy=policy,
            resolve_max_num_reqs=kvc.resolve_max_num_reqs,
            automatic_requests=get_schedule().max_running_requests is None,
            budget_bytes=budget_bytes,
            reservations=reservations,
            min_full_tokens=min_full_tokens,
            max_full_tokens=max_full_tokens,
            external_token_cap=external_token_cap,
        )
        if Step5PoolConfigurator._mode_inputs() != mode:
            raise RuntimeError("runtime sizing inputs changed during the solve")
        return replace(payload, budget_bytes=budget_bytes, mode_inputs=mode)

    @staticmethod
    def price_pool(
        native_config: MemoryPoolConfig,
        geometry: Step5KVGeometry,
        requests: Step5RequestCapacity,
        reservations: Step5KVReservations,
    ) -> Step5MemoryPoolConfig:
        if not isinstance(native_config, MemoryPoolConfig):
            raise ValueError("a native MemoryPoolConfig is required")
        if not isinstance(reservations, Step5KVReservations):
            raise ValueError("explicit complete Step5 reservations are required")
        # Native MTP NHD copy warmup for K/V width 192 BF16 uses one int64[256]
        # dummy-location tensor. This is another necessary individual lower bound,
        # not a proved complete constructor peak and not a second charged term.
        if geometry.draft_swa_ids and reservations.constructor_extra_bytes < 2048:
            raise ValueError(
                "constructor reserve is below the native KV copy warmup cost"
            )
        full, swa = (
            native_config.full_max_total_num_tokens,
            native_config.swa_max_total_num_tokens,
        )
        page = geometry.page_size
        _step5_capacity("FULL capacity", full, page)
        _step5_capacity("SWA capacity", swa, page)
        _step5_integer("max_total_num_tokens", native_config.max_total_num_tokens, 1)
        if native_config.max_running_requests is not None:
            _step5_integer(
                "native max_running_requests", native_config.max_running_requests, 1
            )
        for value in (
            native_config.c4_max_total_num_tokens,
            native_config.c128_max_total_num_tokens,
            native_config.c4_state_pool_size,
            native_config.c128_state_pool_size,
        ):
            _step5_integer("non-Step5 capacity", value)
        if (
            native_config.max_total_num_tokens != full
            or native_config.max_running_requests
            not in (None, requests.max_running_requests)
            or any(
                (
                    native_config.c4_max_total_num_tokens,
                    native_config.c128_max_total_num_tokens,
                    native_config.c4_state_pool_size,
                    native_config.c128_state_pool_size,
                )
            )
        ):
            raise ValueError(
                "native pool config is not the requested plain FULL/SWA profile"
            )
        if full < 2 * requests.max_running_requests:
            raise ValueError("FULL capacity would reduce the requested B")
        # Native ratio sizing rejects equality at window + page. Chunk-cap's
        # generic constrained entry does not check it, so retain this admission
        # requirement explicitly after replaying an external Step5 token cap.
        if swa <= geometry.sliding_window + page:
            raise ValueError(
                "SWA capacity is below the native window-plus-page admission floor"
            )
        if (
            full > reservations.full_capacity_ceiling
            or swa > reservations.swa_capacity_ceiling
        ):
            raise ValueError("capacities exceed the declared reservation domain")
        state_request_capacity = requests.max_running_requests + requests.extra_slots
        if not 1 <= state_request_capacity <= _STEP5_INT32_MAX:
            raise ValueError("request slot IDs including PD extra slots exceed int32")
        slots = state_request_capacity + 1
        layers = len(geometry.target_full_ids)
        compressed_index_key_bytes = (
            layers
            * (full // page + 1)
            * (page // geometry.region_size)
            * geometry.index_k_heads
            * geometry.index_head_dim
        )
        ring_bytes = (
            layers
            * slots
            * (
                4
                * requests.index_ring_size
                * geometry.index_k_heads
                * geometry.index_head_dim
                + 8 * requests.index_ring_size
            )
        )
        per_token = _step5_full_kv_bytes_per_token(
            geometry.local_kv_heads,
            geometry.k_head_dim,
            geometry.v_head_dim,
            fp8=False,
        )
        full_per_token = _step5_full_kv_bytes_per_token(
            geometry.local_kv_heads,
            geometry.k_head_dim,
            geometry.v_head_dim,
            fp8=geometry.full_kv_fp8,
        )
        layer_count = len(
            geometry.target_full_ids + geometry.target_swa_ids + geometry.draft_swa_ids
        )
        # These two costs are consumed by allocation checks. The other costs are
        # local arithmetic, not a string-keyed ledger required by pool construction.
        index_state_bytes = _step5_bytes(compressed_index_key_bytes + ring_bytes)
        # Public two-row stash: int64 tokens, FP32 hidden states, int32 length.
        draft_boundary_stash_bytes = _step5_bytes(
            slots * (8 * geometry.hidden_size + 20) if geometry.draft_swa_ids else 0
        )
        persistent = _step5_bytes(
            (full + page) * len(geometry.target_full_ids) * full_per_token
            + (swa + page)
            * (len(geometry.target_swa_ids) + len(geometry.draft_swa_ids))
            * per_token
            + draft_boundary_stash_bytes
            # ReqToTokenPool: int32 slot-to-token table (including slot zero).
            + 4 * slots * requests.row_capacity
            # Shared int64 FULL->SWA map, including dummy page and -1 sentinel.
            + 8 * (full + page + 1)
            # Two native allocators: int64 free-page IDs, no dummy page entry.
            + 8 * (full // page + swa // page)
            # MHATokenToKVPool: K/V pointers, combined pointers and strides.
            # Each BF16 layer has (1 + 1 + 2 + 2) 64-bit entries.
            + 8 * (1 + 1 + 2 + 2) * layer_count
            + index_state_bytes
        )
        payload = Step5MemoryPoolConfig(
            max_total_num_tokens=full,
            full_max_total_num_tokens=full,
            swa_max_total_num_tokens=swa,
            max_running_requests=requests.max_running_requests,
            geometry=geometry,
            requests=requests,
            index_state_bytes=index_state_bytes,
            draft_boundary_stash_bytes=draft_boundary_stash_bytes,
            persistent_payload_bytes=persistent,
            reservations=reservations,
        )
        _step5_bytes(payload.charged_bytes)
        return payload

    @staticmethod
    def solve_capacity(
        *,
        geometry: Step5KVGeometry,
        requests: Step5RequestCapacity,
        policy: HybridSWAPoolConfigurator,
        resolve_max_num_reqs: Callable[[int], int],
        automatic_requests: bool,
        budget_bytes: int,
        reservations: Step5KVReservations,
        min_full_tokens: int,
        max_full_tokens: int,
        external_token_cap: int | None = None,
    ) -> Step5MemoryPoolConfig:
        """Maximize profiled FULL pages using native request-capacity resolution.

        Explicit request limits keep B fixed. With automatic concurrency, price
        request-owned buffers at each candidate's resolved B. Reservations must
        cover the entire interval and its maximum B, not just the winning page.
        An external aligned cap is replayed once in the native constrained phase;
        freed bytes are not re-spent. A later adapter must first resolve and pass
        user max-total-tokens and other native token constraints into this explicit
        cap; this function does not consume those ServerArgs limits itself.
        A result proves only payload plus declared
        reserves <= the caller's available KV budget, not a device peak bound.
        """
        _step5_integer("budget_bytes", budget_bytes)
        max_running_requests = requests.max_running_requests
        page = geometry.page_size
        _step5_capacity("min_full_tokens", min_full_tokens, page)
        _step5_capacity("max_full_tokens", max_full_tokens, page)
        if min_full_tokens > max_full_tokens:
            raise ValueError("invalid declared FULL interval")
        if automatic_requests:
            if resolve_max_num_reqs(max_full_tokens) != max_running_requests:
                raise ValueError("request reservation must cover the FULL upper bound")
        else:
            if min_full_tokens < 2 * max_running_requests:
                raise ValueError("invalid declared FULL interval for the fixed B")
            if resolve_max_num_reqs(min_full_tokens) != max_running_requests:
                raise ValueError("native request resolution changed the declared B")
        if external_token_cap is not None:
            _step5_capacity("external_token_cap", external_token_cap, page)
            if external_token_cap < min_full_tokens:
                raise ValueError(
                    "external token cap is below the declared workload minimum"
                )

        def project(
            full: int, phase: _Step5Phase = "profiled"
        ) -> Step5MemoryPoolConfig:
            config = _capacity_from_full_tokens(policy, full, page, phase=phase)
            candidate_requests = (
                msgspec.structs.replace(
                    requests, max_running_requests=resolve_max_num_reqs(full)
                )
                if automatic_requests
                else requests
            )
            return Step5PoolConfigurator.price_pool(
                config, geometry, candidate_requests, reservations
            )

        minimum = project(min_full_tokens)
        # Pricing the upper endpoint validates the entire finite reservation domain
        # and int64 arithmetic even if this endpoint is not affordable.
        project(max_full_tokens)
        if minimum.charged_bytes > budget_bytes:
            raise ValueError("declared minimum workload exceeds the KV budget")
        low, high = min_full_tokens // page, max_full_tokens // page
        while low < high:
            middle = (low + high + 1) // 2
            if project(middle * page).charged_bytes <= budget_bytes:
                low = middle
            else:
                high = middle - 1
        profiled = low * page
        payload = project(profiled)
        if profiled < max_full_tokens:
            next_bytes = project(profiled + page).charged_bytes
            if next_bytes <= budget_bytes:
                raise RuntimeError("profiled next-page maximality check failed")
        if external_token_cap is not None and external_token_cap < profiled:
            payload = project(external_token_cap, "token_constrained")
        if payload.charged_bytes > budget_bytes:
            raise ValueError("final native constrained payload exceeds the KV budget")
        if (
            resolve_max_num_reqs(payload.full_max_total_num_tokens)
            != payload.requests.max_running_requests
        ):
            raise ValueError("final native request resolution changed the declared B")
        return payload

    @staticmethod
    def _mtp_workspace_allowance(
        *, tokens: int, requests: int, hidden: int, vocab: int, images: bool = False
    ):
        # Allow eight live FP32 hidden-sized work areas and metadata vectors,
        # plus eight per-request vocabulary arrays for logits/probability work.
        # Persistent two-row history and private KV are charged by the solver.
        image_workspace = (
            # Shifted prefill embedding views/copies, plus the shared graph input,
            # rotation result and capture snapshot at four rows per request.
            4 * tokens * hidden * 2 + 3 * (4 * requests) * hidden * 2 if images else 0
        )
        return (
            8 * tokens * (4 * hidden + 64) + 8 * requests * vocab * 4 + image_workspace
        )

    @staticmethod
    def _workspace_allowance(*, context: int, tokens: int, requests: int) -> int:
        """Price eager tensor dimensions, plus explicit library/JIT scratch allowance.

        Region selection tiles query rows under the runtime's 512 MiB score-buffer
        bound. Price two tiles because the portable fallback keeps the scores and
        canonical-zero copy live together; the SM103 CUTE path needs only one.
        Include all selection/packing arrays and worst-case request-contained unions,
        even where lifetimes do not overlap. The extra 512 MiB is an allowance for
        library scratch/allocator effects, not a measured peak or a promise about
        arbitrary future backends.
        """
        from sglang.srt.layers.attention.step5.metadata import selection_tile_rows

        regions = (context + 7) // 8
        bins = 1 << (regions - 1).bit_length()
        groups = (tokens + 15) // 16 + requests
        # Same tile policy as the runtime workspace (region_selection.py).
        score_tile_rows = selection_tile_rows(rows=tokens, score_width=regions)
        # Two FP32 buffers cover scores plus the canonical-zero fallback copy.
        scores = 2 * score_tile_rows * regions * 4
        rows = tokens * (513 * (4 + 8 + 8 + 8) + 4096 + 1024 + 3072 + 64)
        unions = groups * (12 * min(8208, bins) + 4 * ((bins + 31) // 32) + 12)
        return scores + rows + unions + 512 * _STEP5_MIB

    @staticmethod
    def _validate_step5_worker(
        kvc: KVCacheConfigurator, config: Step5MemoryPoolConfig
    ) -> None:
        from sglang.srt.configs.step5 import Step5Config
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )
        from sglang.srt.runtime_context import (
            get_exec,
            get_memory,
            get_parallel,
            get_schedule,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        geometry, requests = config.geometry, config.requests
        mc = kvc.model_config
        if not isinstance(mc.hf_text_config, Step5Config):
            raise ValueError("Step5 construction requires the resolved Step5 config")
        if (
            kvc.post_capture_kv_active
            or get_memory().enable_unified_memory
            or get_memory().enable_page_major_kv_layout
            or get_memory().enable_hisparse
            or get_schedule().prefill_only_disable_kv_cache
            or not kvc.is_hybrid_swa
            or kvc.is_hybrid_swa_compress
            or kvc.use_mla_backend
            or kvc.mambaish_config is not None
            or kvc.hybrid_gdn_config is not None
            or kvc.model_dtype != torch.bfloat16
            or kvc.kv_cache_dtype != torch.bfloat16
            or kvc.kv_cache_dtype_str not in (None, "auto", "bf16", "bfloat16")
            or kvc.device not in ("cpu", "cuda")
            or (kvc.device == "cuda" and not current_platform.is_cuda())
            or current_platform.is_out_of_tree()
            or get_exec().kernel.attention_backend == "ascend"
        ):
            raise ValueError("unsupported Step5 native KV construction mode")
        if kvc.device == "cuda" and torch.cuda.current_device() != kvc.gpu_id:
            raise ValueError("current CUDA device differs from the Step5 worker gpu_id")
        if get_is_capture_mode() or (
            kvc.device == "cuda" and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("Step5 KV construction is forbidden during capture")
        parallel = get_parallel()
        _geometry_checks = {
            "page_size": (kvc.page_size, geometry.page_size),
            "pool_page_size": (kvc.pool_page_size, geometry.page_size),
            "sliding_window_size": (kvc.sliding_window_size, geometry.sliding_window),
            "hf_sliding_window": (
                mc.hf_text_config.sliding_window,
                geometry.sliding_window,
            ),
            "context_len": (mc.context_len, requests.context_len),
            "hidden_size": (mc.hf_text_config.hidden_size, geometry.hidden_size),
            "sparse_geometry": (
                (
                    mc.hf_text_config.sparse_config["region_block_size"],
                    mc.hf_text_config.index_head_dim,
                    mc.hf_text_config.sparse_config["sparse_indexer_num_k_heads"],
                ),
                (geometry.region_size, geometry.index_head_dim, geometry.index_k_heads),
            ),
            "num_kv_heads": (
                mc.get_num_kv_heads(parallel.attn_tp_size, parallel.attn_dcp_size),
                geometry.local_kv_heads,
            ),
            "swa_num_kv_heads": (
                mc.get_swa_num_kv_heads(parallel.attn_tp_size),
                geometry.local_kv_heads,
            ),
            "k_head_dim": ((mc.head_dim, mc.swa_head_dim), (geometry.k_head_dim,) * 2),
            "v_head_dim": (
                (mc.v_head_dim, mc.swa_v_head_dim),
                (geometry.v_head_dim,) * 2,
            ),
        }
        _geometry_mismatch = {k: v for k, v in _geometry_checks.items() if v[0] != v[1]}
        if _geometry_mismatch:
            raise ValueError(
                f"native worker geometry differs from the Step5 KV budget: "
                f"{_geometry_mismatch} (is_draft={mc.is_draft_model})"
            )
        # Native runner setup already resolved the layer partition. Construction
        # consumes that same model-owned selection rather than resolving it again.
        selection = kvc.layer_info.step5_layer_selection
        if (
            selection is None
            or selection is not kvc.model.layer_selection
            or selection.draft_model_idx != kvc.draft_model_idx
        ):
            raise ValueError(
                "native worker must consume the model's exact Step5 selection"
            )
        if kvc.is_draft_worker != (selection.draft_model_idx is not None):
            raise ValueError("Step5 selection and model/runner draft roles disagree")
        if kvc.is_draft_worker:
            depth = kvc.draft_model_idx
            if (
                type(depth) is not int
                or not 0 <= depth < len(geometry.draft_swa_ids)
                or tuple(mc.swa_attention_layer_ids) != geometry.draft_swa_ids
                or selection.swa_attention_layer_ids != (geometry.draft_swa_ids[depth],)
                or selection.full_attention_layer_ids
                or kvc.spec_algorithm
                not in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3)
            ):
                raise ValueError(
                    "draft worker differs from the three-SWA target budget"
                )
        elif (
            selection.full_attention_layer_ids != geometry.target_full_ids
            or selection.swa_attention_layer_ids != geometry.target_swa_ids
        ):
            raise ValueError("target selection differs from the Step5 KV budget")

    @staticmethod
    def _validate_step5_shared_resources(
        *,
        config: Step5MemoryPoolConfig,
        req_pool: ReqToTokenPool,
        allocator: BaseTokenToKVPoolAllocator,
        device: torch.device,
    ) -> None:
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.step5_kv_pool import Step5TokenToKVPool

        geometry, requests = config.geometry, config.requests
        if not isinstance(allocator, SWATokenToKVPoolAllocator):
            raise ValueError("Step5 requires an SWA allocator")
        # PD DecodeReqToTokenPool is not a ReqToTokenPool subclass. Both expose
        # this storage contract; include PD's preallocated rows in the budget.
        mapping = allocator.full_to_swa_index_mapping
        if (
            req_pool.size != requests.max_running_requests
            or req_pool.max_context_len != requests.row_capacity
            or tuple(req_pool.req_to_token.shape)
            != (config.num_req_slots, requests.row_capacity)
            or req_pool.req_to_token.dtype != torch.int32
            or req_pool.req_to_token.device != device
            or (allocator.size_full, allocator.size_swa)
            != (config.full_max_total_num_tokens, config.swa_max_total_num_tokens)
            or allocator.page_size != geometry.page_size
            or allocator.dtype != torch.bfloat16
            or tuple(mapping.shape)
            != (config.full_max_total_num_tokens + geometry.page_size + 1,)
            or mapping.dtype != torch.int64
            or mapping.device != device
        ):
            raise ValueError("native shared resources differ from the Step5 KV budget")
        target_pool = allocator.get_kvcache()
        if (
            not isinstance(target_pool, Step5TokenToKVPool)
            or target_pool.full_to_swa_index_mapping is not mapping
        ):
            raise ValueError("the native target pool must own the allocator's mapping")

    @staticmethod
    def _validate_step5_construction(
        *, kvc: KVCacheConfigurator, config: MemoryPoolConfig | None
    ) -> Step5MemoryPoolConfig:
        """Validate a prepared budget before allocation; never accept an unpriced pool."""

        if not isinstance(config, Step5MemoryPoolConfig):
            raise ValueError("Step5 construction requires a prepared MemoryPoolConfig")
        Step5PoolConfigurator._validate_runtime(kvc=kvc, config=config)
        capacities = (
            config.max_total_num_tokens,
            config.full_max_total_num_tokens,
            config.swa_max_total_num_tokens,
            config.max_running_requests,
        )
        if (
            any(type(value) is not int for value in capacities)
            or capacities
            != (
                config.full_max_total_num_tokens,
                config.full_max_total_num_tokens,
                config.swa_max_total_num_tokens,
                config.requests.max_running_requests,
            )
            or any(
                (
                    config.c4_max_total_num_tokens,
                    config.c128_max_total_num_tokens,
                    config.c4_state_pool_size,
                    config.c128_state_pool_size,
                )
            )
        ):
            raise ValueError(
                "native pool capacities differ from the prepared Step5 budget"
            )
        if (
            type(config.budget_bytes) is not int
            or config.budget_bytes < config.charged_bytes
        ):
            raise ValueError("prepared Step5 config exceeds its stated budget")
        Step5PoolConfigurator._validate_step5_worker(kvc, config)
        if kvc.is_draft_worker:
            Step5PoolConfigurator._validate_step5_shared_resources(
                config=config,
                req_pool=kvc.req_to_token_pool,
                allocator=kvc.token_to_kv_pool_allocator,
                device=(
                    torch.device(kvc.device, kvc.gpu_id)
                    if kvc.device == "cuda"
                    else torch.device("cpu")
                ),
            )
        else:
            if (
                kvc.req_to_token_pool is not None
                or kvc.token_to_kv_pool_allocator is not None
            ):
                raise ValueError(
                    "Step5 target KV resources may only be constructed once"
                )
            # Geometry and requests are the configurator's immutable config. Check
            # byte accounting without running model/capacity resolution again.
            actual = Step5PoolConfigurator.price_pool(
                config, config.geometry, config.requests, config.reservations
            )
            if (
                actual.index_state_bytes,
                actual.draft_boundary_stash_bytes,
                actual.persistent_payload_bytes,
            ) != (
                config.index_state_bytes,
                config.draft_boundary_stash_bytes,
                config.persistent_payload_bytes,
            ):
                raise ValueError(
                    "prepared Step5 config differs from resolved target metadata"
                )
        return config

    @staticmethod
    def _finish_step5_construction(
        *,
        kvc: KVCacheConfigurator,
        config: Step5MemoryPoolConfig,
        req_pool: ReqToTokenPool,
        kv_pool: KVCache,
        allocator: BaseTokenToKVPoolAllocator,
    ) -> None:
        """Check actual resources after the pre-allocation runtime admission check."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.mem_cache.step5_kv_pool import (
            Step5Fp8FullKVPool,
            Step5TokenToKVPool,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        device = (
            torch.device(kvc.device, kvc.gpu_id)
            if kvc.device == "cuda"
            else torch.device("cpu")
        )
        Step5PoolConfigurator._validate_step5_shared_resources(
            config=config, req_pool=req_pool, allocator=allocator, device=device
        )
        selection = kvc.layer_info.step5_layer_selection
        expected_mapping = {
            layer: (i, False)
            for i, layer in enumerate(selection.full_attention_layer_ids)
        }
        expected_mapping.update(
            {
                layer: (i, True)
                for i, layer in enumerate(selection.swa_attention_layer_ids)
            }
        )
        # Draft pools are plain SWA owners: a Step5TokenToKVPool is also an SWAKVPool
        # subclass, but carries target-only state that is not charged to a draft.
        # MHA subclasses include NoOp, FP4, page-major and unified storage. The byte
        # model below prices the native BF16 NHD allocation, not those other layouts.
        if (
            (
                type(kv_pool) is not SWAKVPool
                if kvc.is_draft_worker
                else not isinstance(kv_pool, Step5TokenToKVPool)
            )
            or (kv_pool.size, kv_pool.size_swa)
            != (config.full_max_total_num_tokens, config.swa_max_total_num_tokens)
            or kv_pool.page_size != config.geometry.page_size
            or kv_pool.layers_mapping != expected_mapping
            or kv_pool.full_to_swa_index_mapping
            is not allocator.full_to_swa_index_mapping
            or type(kv_pool.full_kv_pool)
            is not (
                Step5Fp8FullKVPool
                if config.geometry.full_kv_fp8 and not kvc.is_draft_worker
                else MHATokenToKVPool
            )
            or type(kv_pool.swa_kv_pool) is not MHATokenToKVPool
        ):
            raise ValueError(
                "constructed native KV pool differs from the Step5 selection/budget"
            )
        if kvc.is_draft_worker:
            if (
                req_pool is not kvc.req_to_token_pool
                or allocator is not kvc.token_to_kv_pool_allocator
            ):
                raise ValueError(
                    "a draft must reuse the exact target request pool and allocator"
                )
            return None
        if allocator.get_kvcache() is not kv_pool:
            raise ValueError("target allocator must retain this exact native KV pool")
        if (
            kv_pool.num_req_slots != config.num_req_slots
            or kv_pool.index_ring_size != config.requests.index_ring_size
        ):
            raise ValueError("Step5 state capacity differs from its resolved budget")
        expected = config.index_state_bytes
        if sum(kv_pool.get_dsa_size_bytes().values()) != expected:
            raise ValueError(
                "Step5 allocated state differs from its resolved byte budget"
            )

    @staticmethod
    def build_kv_pool(
        kvc,
        *,
        full_max_total_num_tokens,
        swa_max_total_num_tokens,
        req_to_token_pool,
        mha_pool_class,
    ):
        from sglang.srt.mem_cache.step5_kv_pool import (
            Step5TokenToKVPool,
            get_step5_ring_capacity,
        )
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.runtime_context import get_exec

        selection = kvc.layer_info.step5_layer_selection
        if selection is None:
            raise ValueError("Step5 native KV pool requires its model-owned selection")
        kwargs = {}
        pool_cls = SWAKVPool
        if not kvc.is_draft_worker:
            pool_cls = Step5TokenToKVPool
            kwargs.update(
                enable_memory_saver=get_exec().features.enable_memory_saver,
                num_req_slots=req_to_token_pool.req_to_token.shape[0],
                index_ring_size=get_step5_ring_capacity(
                    1
                    if kvc.spec_algorithm.is_none()
                    else get_spec().speculative_num_draft_tokens
                ),
                full_kv_fp8=_step5_full_kv_fp8(),
            )
        return pool_cls(
            size=full_max_total_num_tokens,
            size_swa=(
                full_max_total_num_tokens
                if kvc.draft_swa_full_capacity
                else swa_max_total_num_tokens
            ),
            page_size=kvc.pool_page_size,
            dtype=kvc.kv_cache_dtype,
            post_capture_active=kvc.post_capture_kv_active,
            head_num=kvc.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size, get_parallel().attn_dcp_size
            ),
            head_dim=kvc.model_config.head_dim,
            swa_attention_layer_ids=selection.swa_attention_layer_ids,
            full_attention_layer_ids=selection.full_attention_layer_ids,
            device=kvc.device,
            enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
            token_to_kv_pool_class=mha_pool_class,
            **kwargs,
        )

"""Readback and runtime updates for the scheduler's internal state."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.managers.io_struct import (
    GetInternalStateReq,
    GetInternalStateReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
)
from sglang.srt.managers.scheduler_components.memory_usage import build_memory_usage
from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_exec,
    get_parallel,
    get_spec,
)
from sglang.srt.server_args import compute_world_size
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.bundles import (
        KvCapacity,
        KvResources,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import (
        MetricsSink,
    )
    from sglang.srt.managers.scheduler_components.request_registry import (
        RequestRegistry,
    )
logger = logging.getLogger(__name__)

# The CUDA graph phases whose captured memory a PD router attributes to decode.
_DECODE_GRAPH_PHASES = ("decode", "target_verify", "draft_decode", "draft_extend")


class UpdateRejected(Exception):
    """A requested update is not applicable to this scheduler."""


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerInternalStateController:
    workers: Workers
    kv: KvResources
    capacity: KvCapacity
    startup_available_gpu_memory_gb: float
    metrics_sink: MetricsSink
    rust_server: Any
    request_registry: RequestRegistry
    startup_time: Any
    record_step_time: bool
    spec_algorithm: SpeculativeAlgorithm = field(
        init=False,
        default_factory=lambda: SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        ),
    )

    # ---------------------------------------------------------------- readback

    def get_internal_state(self, recv_req: GetInternalStateReq):
        # Resolved config (pristine server_args + post-publish overrides) so a
        # readback reflects values changed via /set_internal_state, not startup.
        state = get_context().resolved_server_args_dict()
        for contribute in (
            self._add_topology,
            self._add_memory_usage,
            self._add_disaggregation,
            self._add_elastic_ep,
            self._add_speculative,
            self._add_step_times,
            self._add_rust_server,
            self._add_env_vars,
        ):
            contribute(state)
        # A bound signal handler is not msgpack-serializable, and no reader
        # consumes it.
        state.pop("custom_sigquit_handler", None)
        return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(state))

    def _add_topology(self, state: Dict[str, Any]) -> None:
        state["world_size"] = compute_world_size(
            dp_size=get_parallel().dp_size,
            tp_size=get_parallel().tp_size,
            pp_size=get_parallel().pp_size,
        )
        state["startup_time"] = self.startup_time
        state["effective_max_running_requests_per_dp"] = (
            self.capacity.max_running_requests
        )
        state["last_gen_throughput"] = self.metrics_sink.last_gen_throughput

    def _add_memory_usage(self, state: Dict[str, Any]) -> None:
        draft_worker = self.workers.draft_worker
        state["memory_usage"] = build_memory_usage(
            weight_gb=self.workers.tp_worker.model_runner.weight_load_mem_usage,
            kv_cache_gb=self.kv.token_to_kv_pool_allocator.get_kvcache().mem_usage,
            startup_available_gb=self.startup_available_gpu_memory_gb,
            token_capacity=self.capacity.max_total_num_tokens,
            token_capacity_swa=self.capacity.swa_tokens_per_layer,
            target_graph_memory_usage=self.workers.tp_worker.graph_memory_usage,
            draft_graph_memory_usage=(
                None if draft_worker is None else draft_worker.graph_memory_usage
            ),
        )

    def _add_disaggregation(self, state: Dict[str, Any]) -> None:
        # PD role switch: report this instance's role and the decode CUDA graph
        # batch sizes it captured, which a router feeds back as
        # PdRoleSwitchReqInput.decode_cuda_graph_bs.
        mode = DisaggregationMode(get_disagg().disaggregation_mode)
        graph_memory = state["memory_usage"]["graph"]
        state["disaggregation_mode"] = mode.value
        state["decode_cuda_graph_bs"] = (
            self.workers.tp_worker.get_decode_cuda_graph_bs()
        )
        state["decode_cuda_graph_memory_gb"] = round(
            sum(graph_memory[phase] for phase in _DECODE_GRAPH_PHASES), 3
        )

    def _add_elastic_ep(self, state: Dict[str, Any]) -> None:
        if get_exec().moe.elastic_ep_backend is None:
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        state["is_scaling_elastic_ep"] = ElasticEPStateManager.is_scaling()
        state["effective_ep_size"] = ElasticEPStateManager.get_effective_ep_size()
        state["pending_ep_size"] = ElasticEPStateManager.get_pending_ep_size()
        state["scale_phase"] = ElasticEPStateManager.get_scale_phase()
        state["elastic_ep_last_error"] = ElasticEPStateManager.get_last_error()

    def _add_speculative(self, state: Dict[str, Any]) -> None:
        if self.spec_algorithm.is_none():
            return
        accept_length = self._average_accept_length()
        if accept_length is not None:
            state["avg_spec_accept_length"] = accept_length
        if self.spec_algorithm.is_dspark() and self.workers.draft_worker is not None:
            info_record = self.workers.draft_worker.dump_info_records()
            if info_record is not None:
                state["dspark_info_record"] = info_record

    def _add_step_times(self, state: Dict[str, Any]) -> None:
        if self.record_step_time:
            state["step_time_dict"] = self.metrics_sink.step_time_dict

    def _add_rust_server(self, state: Dict[str, Any]) -> None:
        if self.rust_server is not None:
            state["rust_mm_transport"] = self.rust_server.mm_transport_stats()
            state["rust_server_startup_time"] = self.rust_server.startup_time

    def _add_env_vars(self, state: Dict[str, Any]) -> None:
        if envs.SGLANG_EXPOSE_OWN_ENV_VARS.get():
            state["env_vars"] = exportable_env_vars()

    # ------------------------------------------------------------------ update

    def set_internal_state(self, recv_req: SetInternalStateReq):
        handlers = self._update_handlers()
        requested = recv_req.server_args
        try:
            for key, value in requested.items():
                handler = handlers.get(key)
                if handler is None:
                    raise UpdateRejected(f"Updating {key} is not supported.")
                handler.validate(key, value)
        except UpdateRejected as rejection:
            logger.warning(str(rejection))
            return SetInternalStateReqOutput(updated=False)

        # Worker commands apply directly; config keys land in one override.
        overrides = {}
        for key, value in requested.items():
            overrides.update(handlers[key].apply(key, value) or {})
        if overrides:
            get_context().override(source="update_server_args", **overrides)
            logger.info(f"Config updated via context override: {overrides}")
        return SetInternalStateReqOutput(updated=True)

    def _update_handlers(self) -> Dict[str, _UpdateHandler]:
        return {
            "pp_max_micro_batch_size": _UpdateHandler(
                self._validate_pp_micro_batch_size, _override
            ),
            "speculative_accept_threshold_single": _UpdateHandler(
                self._validate_accept_threshold, self._apply_accept_threshold
            ),
            "speculative_accept_threshold_acc": _UpdateHandler(
                self._validate_accept_threshold, self._apply_accept_threshold
            ),
            "dspark_force_budget_frac": _UpdateHandler(
                self._validate_dspark_budget_frac, self._apply_dspark_budget_frac
            ),
            "dspark_clear_info_records": _UpdateHandler(
                self._validate_dspark_clear_records, self._apply_dspark_clear_records
            ),
        }

    def _validate_pp_micro_batch_size(self, key: str, value: Any) -> None:
        upper = self.capacity.max_running_requests // get_parallel().pp_size
        if value < 1 or value > upper:
            raise UpdateRejected(
                f"Updating {key} to {value} is rejected because it is out of "
                f"the valid range [1, {upper}]."
            )

    def _validate_accept_threshold(self, key: str, value: Any) -> None:
        # A relaxed threshold stops acceptance from sampling exactly from the
        # target policy, which is what a captured sampling mask describes.
        if not self.spec_algorithm.is_dflash() or float(value) == 1.0:
            return
        if self._has_active_sampling_mask_request():
            raise UpdateRejected(
                f"Updating {key} is rejected while DFlash sampling-mask "
                "requests are active."
            )

    def _apply_accept_threshold(self, key: str, value: Any) -> Dict[str, Any]:
        # Acceptance statistics describe the previous threshold; start over.
        accept_length = self._average_accept_length()
        if accept_length is not None:
            logger.info(f"avg_spec_accept_length={accept_length}")
        metrics_sink = self.metrics_sink
        metrics_sink.spec_total_num_accept_tokens = (
            metrics_sink.spec_total_num_forward_ct
        ) = 0
        return {key: value}

    def _validate_dspark_budget_frac(self, key: str, value: Any) -> None:
        self._require_dspark_worker(key, "set_dspark_forced_budget_frac")
        if value is not None and not (0.0 < float(value) <= 1.0):
            raise UpdateRejected(f"{key} must be in (0, 1] or null, got {value}.")

    def _apply_dspark_budget_frac(self, key: str, value: Any) -> None:
        # A worker command, not a server arg: keep it out of the override.
        self.workers.draft_worker.set_dspark_forced_budget_frac(
            None if value is None else float(value)
        )

    def _validate_dspark_clear_records(self, key: str, value: Any) -> None:
        self._require_dspark_worker(key, "clear_info_records")

    def _apply_dspark_clear_records(self, key: str, value: Any) -> None:
        if value:
            self.workers.draft_worker.clear_info_records()

    def _require_dspark_worker(self, key: str, command: str) -> None:
        if not self.spec_algorithm.is_dspark() or not hasattr(
            self.workers.draft_worker, command
        ):
            raise UpdateRejected(f"{key} requires a DSpark draft worker.")

    # ------------------------------------------------------------------ shared

    def _average_accept_length(self) -> Optional[float]:
        metrics_sink = self.metrics_sink
        if self.spec_algorithm.is_none() or metrics_sink.spec_total_num_forward_ct <= 0:
            return None
        return (
            metrics_sink.spec_total_num_accept_tokens
            / metrics_sink.spec_total_num_forward_ct
        )

    def _has_active_sampling_mask_request(self) -> bool:
        """Whether any unfinished request holding a sampling mask is in a
        request list."""
        return any(
            req.return_sampling_mask and not req.finished()
            for entry in self.request_registry.request_lists().values()
            for req in entry.reqs
        )


class _UpdateHandler:
    def __init__(
        self,
        validate: Callable[[str, Any], None],
        apply: Callable[[str, Any], Optional[Dict[str, Any]]],
    ) -> None:
        self.validate = validate
        self.apply = apply


def _override(key: str, value: Any) -> Dict[str, Any]:
    return {key: value}

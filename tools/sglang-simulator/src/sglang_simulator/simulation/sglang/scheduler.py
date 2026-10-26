import heapq
import importlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial, wraps
from typing import Any

from sglang_simulator.compat import validate_simulator_server_args
from sglang_simulator.hook import (
    BaseHook,
    is_class_hook_matched,
    validate_required_class_hooks,
)
from sglang_simulator.hook.utils import get_obj_from_args
from sglang_simulator.simulation.manager import ConfigManager, Envs, StateManager
from sglang_simulator.simulation.sglang.dspark import (
    DSparkStepResult,
    dspark_result_from_native,
)
from sglang_simulator.simulation.sglang.prepared_batch import (
    predictor_batch_from_result,
)
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.utils import (
    resolve_model_info,
    resolve_scheduler_config,
)
from sglang_simulator.simulation.types import (
    RequestStats,
    SimulationMode,
)
from sglang_simulator.simulation.utils import (
    calc_iteration_metrics,
    calc_metrics,
)
from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    PredictorError,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.base import (
    ScheduleBatch as SimulationScheduleBatch,
)
from sglang_simulator.utils import get_logger
from sglang_simulator.utils.json import CustomJsonEncoder

logger = get_logger("sgl_simulator")

_DSPARK_ACCOUNTING = "_sglang_simulator_dspark_accounting"
_DSPARK_ACCOUNTED = "_sglang_simulator_dspark_accounted"
_DSPARK_WALL_CREDIT = "_sglang_simulator_dspark_wall_credit"
_DSPARK_ACCOUNTING_OWNER = "_sglang_simulator_dspark_accounting_owner"
_DSPARK_PENDING_RESULTS = "_sglang_simulator_dspark_pending_results"


@dataclass(frozen=True)
class _DSparkResultAccounting:
    owner: object
    prediction: DSparkStepResult
    forward_duration: float
    previous_inference_duration: float
    preprocess_duration: float


def _has_pending_dspark_work(owner, predictor):
    runtime = getattr(predictor, "dspark_runtime", None)
    driver = getattr(runtime, "reference_driver", None)
    if driver is not None:
        driver.check()
        ledger = driver.adapter.ledger
        if (
            driver.selection is not None
            or driver.processing is not None
            or ledger.prepared is not None
            or ledger.pending
        ):
            return True
    return bool(getattr(owner, _DSPARK_PENDING_RESULTS, 0))


def _validate_dspark_batch(prediction, batch):
    group = prediction.local_group
    source_mode = getattr(batch.forward_mode, "name", None)
    allowed_modes = {
        "DECODE": {"TARGET_VERIFY"},
        "TARGET_VERIFY": {"TARGET_VERIFY"},
        "EXTEND": {"EXTEND", "MIXED"},
        "MIXED": {"MIXED"},
        "IDLE": {"IDLE"},
    }
    if group.forward_mode not in allowed_modes.get(source_mode, set()):
        raise PredictorError(
            "invalid_dspark_step", "DSPark worker mode differs from the native batch"
        )
    if tuple(req.rid for req in batch.reqs) != tuple(
        request.rid for request in group.requests
    ):
        raise PredictorError(
            "invalid_dspark_step", "DSPark result does not match the native local batch"
        )


def _dspark_simulation_batch(prediction):
    group = prediction.local_group
    return SimulationScheduleBatch(
        reqs=[
            ScheduleRequest(r.extend_len, r.target_prefix_len) for r in group.requests
        ],
        forward_mode=group.forward_mode,
    )


def simulation_mode_log_message(mode: SimulationMode) -> str:
    return f"SGLang Simulator simulation mode: {mode.value}"


def _host_int(value):
    value = value.item() if hasattr(value, "item") else value
    if type(value) is not int:
        raise PredictorError("invalid_batch", "prepared token lengths must be integers")
    return value


def _request_prefix_len(req):
    value = getattr(req, "prefix_len", None)
    if value is None:
        value = len(req.prefix_indices)
    return _host_int(value)


def effective_l2_load_delay(
    load_duration: float,
    last_inference_duration: float,
    overlap_schedule: bool,
) -> float:
    if overlap_schedule:
        return max(load_duration - last_inference_duration, 0.0)
    return max(load_duration, 0.0)


def block_on_l2_load(mode: SimulationMode, delay: float) -> float:
    """Sleep for visible L2 load time and return actual blocked wall time."""
    if mode != SimulationMode.BLOCKING or delay <= 0:
        return 0.0
    start = time.perf_counter()
    time.sleep(delay)
    return time.perf_counter() - start


class C_SglangPrefillAdderHook(BaseHook):
    HOOK_CLASS_NAME = "PrefillAdder"
    HOOK_MODULE_NAME = "sglang.srt.managers.schedule_policy"

    @classmethod
    def hook(cls, target):
        original_add_one_req = target.add_one_req

        def wrapped_add_one_req(self, *args, **kwargs):
            req = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.Req",
                *args,
                **kwargs,
            )
            req_infos = request_stats_manager.get_req_stats(req.rid)
            req_infos.before_adder_device_hit_len = _request_prefix_len(req)
            req_infos.final_host_hit_len = req.host_hit_length

            return original_add_one_req(self, *args, **kwargs)

        target.add_one_req = wrapped_add_one_req


class ReqDispatcher:
    _instance = None
    _initialized = False

    def __new__(cls, mode):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, mode: SimulationMode):
        if self.__class__._initialized:
            return

        self.mode = mode
        # If the simulation mode is `BLOCKING`, all requests are released immediately.
        # If the simulation mode is `OFFLINE`, only control requests, such as `flush_cache`
        # and `server_info`, are released immediately.
        self.immediate_release_requests = []
        self.future_queue: list[
            tuple[float, int, Any]
        ] = []  # tuple(created time, salt, request)
        self.offline_recv_all_requests = False
        self.profile_active = False

    @staticmethod
    def simulation_created_time_s(simulation_args: dict) -> float:
        if "created_time_ms" in simulation_args:
            return simulation_args["created_time_ms"] / 1000.0
        return simulation_args["created_time"]

    def has_next(self) -> bool:
        return len(self.future_queue) > 0

    def next_req_from_future_ts(self) -> float:
        return self.future_queue[0][0]

    def reset(self) -> None:
        self.immediate_release_requests.clear()
        self.future_queue.clear()
        self.offline_recv_all_requests = False

    def add(self, reqs: list):
        if self.mode == SimulationMode.BLOCKING:
            self.immediate_release_requests.extend(reqs)
        elif self.mode == SimulationMode.OFFLINE:
            if self.offline_recv_all_requests:
                self.immediate_release_requests.extend(reqs)
                return

            gen_requests = []
            time.sleep(0.05)  # waiting requests

            for req in reqs:
                if req.__class__.__name__ == "TokenizedGenerateReqInput":
                    gen_requests.append(req)
                else:
                    # Such as: /profile_start, /flush_cache, etc.
                    self.immediate_release_requests.append(req)

            # Add requests to future queue
            for req in gen_requests:
                sim_params = None
                if req.sampling_params.custom_params is not None:
                    sim_params = req.sampling_params.custom_params.get("simulation")
                if sim_params is None:
                    # There are some warm-up requests when starting the server without --skip-server-warmup.
                    self.immediate_release_requests.append(req)
                    logger.warning(
                        "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                    )
                    continue
                if sim_params.get("queue_start"):
                    logger.debug(
                        "Add request to waiting queue with custom queue start timestamp."
                    )

                self.future_queue.append(
                    (
                        sim_params.get("queue_start")
                        or self.simulation_created_time_s(sim_params),
                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                        req,
                    )
                )

            if len(self.future_queue) != 0:
                _, _, gen_req = self.future_queue[-1]
                total_request = gen_req.sampling_params.custom_params["simulation"][
                    "total_request"
                ]

                if len(self.future_queue) == total_request:
                    self.offline_recv_all_requests = True
                    heapq.heapify(self.future_queue)
                    logger.info("All requests received. Starting simulation now.")
                else:
                    logger.info(
                        f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(self.future_queue)} requests so far."
                    )

    def dispatch(self) -> list:
        recv_reqs = []

        recv_reqs.extend(self.immediate_release_requests)
        self.immediate_release_requests.clear()

        if self.mode == SimulationMode.OFFLINE and self.offline_recv_all_requests:
            # Process the arrived requests only after all requests have been added to the future queue
            current_timestamp = StateManager.get_global_clock()
            while len(self.future_queue) > 0:
                enqueue_time, _, req = self.future_queue[0]
                if enqueue_time > current_timestamp:
                    break
                recv_reqs.append(req)
                heapq.heappop(self.future_queue)

        now = time.time()
        for req in recv_reqs:
            if req.__class__.__name__ in [
                "BatchTokenizedGenerateReqInput",
                "TokenizedGenerateReqInput",
            ]:
                simulation_args = None
                if req.sampling_params.custom_params is not None:
                    simulation_args = req.sampling_params.custom_params.get(
                        "simulation"
                    )
                # The warm-up request might not include any simulation arguments.
                if simulation_args is None:
                    if self.mode != SimulationMode.BLOCKING or not self.profile_active:
                        continue
                    simulation_args = {}
                req_stats = request_stats_manager.get_req_stats(req.rid)
                req_stats.rid = req.rid
                req_stats.input_length = len(req.input_ids)
                req_stats.output_length = req.sampling_params.max_new_tokens

                if self.mode == SimulationMode.BLOCKING:
                    req_stats.created_time = simulation_args.get(
                        "server_created_time", now
                    )
                    req_stats.last_event_time = req_stats.created_time
                    req_stats.queue_start = now
                elif self.mode == SimulationMode.OFFLINE:
                    req_stats.created_time = self.simulation_created_time_s(
                        simulation_args
                    )
                    req_stats.last_event_time = req_stats.created_time
                    # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                    queue_start = simulation_args.get("queue_start")
                    if queue_start is not None:
                        StateManager.set_global_clock(queue_start)
                    req_stats.queue_start = StateManager.get_global_clock()

        if recv_reqs and StateManager.get_last_real_time_ts() == 0:
            StateManager.set_last_real_time_ts(time.time())
            StateManager.set_global_clock(
                now if self.mode == SimulationMode.BLOCKING else 0
            )

        return recv_reqs


class C_SchedulerRequestReceiver(BaseHook):
    HOOK_CLASS_NAME = "SchedulerRequestReceiver"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler_components.request_receiver"

    # Older SGLang versions receive requests directly on Scheduler; that path is
    # patched by C_SchedulerHook instead.
    REQUIRED = False

    REQ_DISPATCHER: ReqDispatcher = ReqDispatcher(
        SimulationMode(Envs.simulation_mode())
    )

    @classmethod
    def hook(cls, target):
        original_recv_requests = target.recv_requests

        def wrapped_recv_requests(self, *args, **kwargs):
            if C_SchedulerHook.COORDINATOR is not None:
                return C_SchedulerHook.COORDINATOR.receive(
                    lambda: original_recv_requests(self, *args, **kwargs)
                )
            recv_reqs = original_recv_requests(self, *args, **kwargs)
            C_SchedulerRequestReceiver.REQ_DISPATCHER.add(recv_reqs)
            return C_SchedulerRequestReceiver.REQ_DISPATCHER.dispatch()

        target.recv_requests = wrapped_recv_requests


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None
    COORDINATOR = None

    ITERATION_STATS: list[dict] = []
    TOTAL_PREDICTOR_TIME_COST = 0
    CURRENT_PREDICTOR_TIME_COST = 0
    CURRENT_PREDICTED_LATENCY = 0
    GET_NEW_BATCH_PREFILL_TIME_COST = 0

    SIMULATION_BATCH: SimulationScheduleBatch = None
    OVERLAP_SCHEDULE: bool = False
    SIM_MODE = SimulationMode(Envs.simulation_mode())
    # Shared singleton instance with `C_SchedulerRequestReceiver.REQ_DISPATCHER`.
    REQ_DISPATCHER = ReqDispatcher(SIM_MODE)

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = getattr(target, "recv_requests", None)
        original_prefetch_kvcache = target._prefetch_kvcache
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal
        original_event_loop_overlap = getattr(target, "event_loop_overlap", None)
        original_overlap_decision = getattr(
            target, "is_disable_overlap_for_batch", None
        )
        original_init_request_dispatcher = target.init_request_dispatcher
        original_on_idle = target.on_idle
        original_release_aborted = getattr(target, "_release_aborted_request", None)
        from sglang_simulator.simulation.sglang.coordinated import (
            install_prefill_cadence_hooks,
        )

        install_prefill_cadence_hooks(target, lambda: C_SchedulerHook.COORDINATOR)
        original_get_next_batch = getattr(target, "get_next_batch_to_run", None)

        def reference_driver():
            runtime = getattr(
                C_SchedulerHook.INFERENCE_PREDICTOR, "dspark_runtime", None
            )
            return getattr(runtime, "reference_driver", None)

        def override_event_loop_overlap(self, *args, **kwargs):
            driver = reference_driver()
            if driver is not None:
                driver.bind(self)
                if original_event_loop_overlap is None:
                    raise PredictorError(
                        "invalid_reference_state", "native overlap loop is unavailable"
                    )
                return original_event_loop_overlap(self, *args, **kwargs)
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        @wraps(original_get_next_batch)
        def wrapped_get_next_batch(self, *args, **kwargs):
            driver = reference_driver()
            prior_requests = ()
            if driver is not None:
                prior_requests = tuple(
                    req
                    for native_batch in (
                        kwargs.get("running_batch"),
                        kwargs.get("last_batch"),
                        *args,
                    )
                    if native_batch is not None
                    for req in getattr(native_batch, "reqs", ())
                )
            plan = original_get_next_batch(self, *args, **kwargs)
            if driver is not None:
                started = time.perf_counter()
                try:
                    driver.selected(
                        self, plan.batch_to_run, prior_requests=prior_requests
                    )
                except Exception as error:
                    driver.fail("native selection/preparation", error)
                    raise
                setattr(
                    self,
                    _DSPARK_WALL_CREDIT,
                    getattr(self, _DSPARK_WALL_CREDIT, 0.0)
                    + time.perf_counter()
                    - started,
                )
            return plan

        def wrapped_overlap_decision(self, batch, *args, **kwargs):
            decision = original_overlap_decision(self, batch, *args, **kwargs)
            driver = reference_driver()
            if driver is not None:
                started = time.perf_counter()
                try:
                    driver.overlap_decision(batch, decision)
                except Exception as error:
                    driver.fail("native overlap decision", error)
                    raise
                setattr(
                    self,
                    _DSPARK_WALL_CREDIT,
                    getattr(self, _DSPARK_WALL_CREDIT, 0.0)
                    + time.perf_counter()
                    - started,
                )
            return decision

        def wrapped_release_aborted(self, req, *args, **kwargs):
            result = original_release_aborted(self, req, *args, **kwargs)
            driver = reference_driver()
            if driver is not None:
                # Native queued/chunk abort release can happen without another
                # result, and prefill does not use the decode finish callback.
                try:
                    C_SchedulerHook.INFERENCE_PREDICTOR.dspark_runtime.note_request_finished(
                        rid=req.rid, natural_stop=False
                    )
                except Exception as error:
                    driver.fail("native abort release", error)
                    raise
            return result

        def wrapped_init(self, *args, **kwargs):
            logger.info(simulation_mode_log_message(C_SchedulerHook.SIM_MODE))
            # Supported entry points prepare the final config before publication.
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            validate_simulator_server_args(server_args)
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )
            original_init(self, *args, **kwargs)
            validate_required_class_hooks()
            if original_recv_requests is None and not is_class_hook_matched(
                C_SchedulerRequestReceiver
            ):
                raise RuntimeError(
                    "SGLang Simulator could not hook a request receiver. The "
                    "simulator must be adapted to this SGLang revision."
                )

            try:
                if ConfigManager.get_model_info() is None:
                    model = resolve_model_info(self.model_config)
                    ConfigManager.set_model_info(model)

                model = ConfigManager.get_model_info()

                hw = ConfigManager.get_accelerator_info()

                if ConfigManager.get_scheduler_config() is None:
                    sched_config = resolve_scheduler_config(
                        server_args=self.server_args,
                        model_config=self.model_config,
                    )
                    ConfigManager.set_scheduler_config(sched_config)
                sched_config = ConfigManager.get_scheduler_config()

                predictor_options = {}
                native_dp_clock = False
                if Envs.coordinator_endpoint():
                    from sglang_simulator.simulation.sglang.dp_coordinated import (
                        native_dp_clock_enabled,
                    )

                    native_dp_clock = native_dp_clock_enabled()
                    if native_dp_clock:
                        predictor_options["attention_dp_rank"] = (
                            self.tp_group.rank_in_group
                        )
                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(
                        model, hw, sched_config, **predictor_options
                    )
                )
                if Envs.coordinator_endpoint():
                    from sglang_simulator.simulation.sglang.coordinated import (
                        CoordinatedScheduler,
                    )
                    from sglang_simulator.simulation.sglang.dp_coordinated import (
                        DPCoordinatedScheduler,
                        internal_endpoint,
                    )

                    def pending():
                        return _has_pending_dspark_work(
                            self, C_SchedulerHook.INFERENCE_PREDICTOR
                        )

                    if native_dp_clock:
                        ports = get_obj_from_args(
                            "sglang.srt.server_args.PortArgs", *args, **kwargs
                        )
                        C_SchedulerHook.COORDINATOR = DPCoordinatedScheduler(
                            scheduler=self,
                            tp_group=self.tp_group,
                            endpoint=internal_endpoint(ports),
                            pending_work=pending,
                        )
                    else:
                        C_SchedulerHook.COORDINATOR = CoordinatedScheduler(
                            scheduler=self,
                            endpoint=Envs.coordinator_endpoint(),
                            pending_work=pending,
                        )
            except Exception as e:
                logger.error(
                    f"Failed to initialize inference time predictor. Error: {e}"
                )
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            if C_SchedulerHook.COORDINATOR is not None:
                return C_SchedulerHook.COORDINATOR.receive(
                    lambda: original_recv_requests(self, *args, **kwargs)
                )
            recv_reqs = original_recv_requests(self, *args, **kwargs)
            C_SchedulerHook.REQ_DISPATCHER.add(recv_reqs)
            return C_SchedulerHook.REQ_DISPATCHER.dispatch()

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            start = time.perf_counter()
            result = original_get_new_batch_prefill(self, *args, **kwargs)
            C_SchedulerHook.GET_NEW_BATCH_PREFILL_TIME_COST = (
                time.perf_counter() - start
            )

            # Accept both a plan wrapper and a direct batch return value.
            new_batch = getattr(result, "batch_to_run", result)

            # A plan reports the running batch before self.running_batch is updated.
            running_batch = getattr(result, "running_batch", self.running_batch)

            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = request_stats_manager.get_req_stats(req.rid)
                    req_stats.final_device_hit_len = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
            elif C_SchedulerHook.COORDINATOR is not None:
                pass  # Only the coordinator may advance virtual time.
            elif len(running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                # Idle stage, there are some requests pendding in the future queue.
                if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE and (
                    C_SchedulerHook.REQ_DISPATCHER.has_next()
                    and len(running_batch.reqs) == 0
                ):
                    next_created_time = (
                        C_SchedulerHook.REQ_DISPATCHER.next_req_from_future_ts()
                    )
                    StateManager.set_global_clock(next_created_time + 1e-6)
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return result

        def wrapped_prefetch_kvcache(self, *args, **kwargs):
            original_prefetch_kvcache(self, *args, **kwargs)

            req = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.Req",
                *args,
                **kwargs,
            )
            req_stats = request_stats_manager.get_req_stats(req.rid)
            req_stats.recv_device_hit_len = _request_prefix_len(req)
            req_stats.recv_host_hit_len = req.host_hit_length

        def wrapped_run_batch(self, *args, **kwargs):
            C_SchedulerHook.CURRENT_PREDICTOR_TIME_COST = 0
            try:
                ret = original_run_batch(self, *args, **kwargs)
            except Exception as error:
                driver = reference_driver()
                if driver is not None:
                    driver.fail("native run_batch", error)
                raise

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )

            if ret.__class__.__name__ == "GenerationBatchResult":
                speculative_step = dspark_result_from_native(ret)
                prepared_batch = getattr(
                    C_SchedulerHook.INFERENCE_PREDICTOR,
                    "requires_prepared_batch",
                    False,
                )
                if speculative_step is not None:
                    if not getattr(
                        C_SchedulerHook.INFERENCE_PREDICTOR, "supports_dspark", False
                    ):
                        raise PredictorError(
                            "unsupported_execution_profile",
                            "predictor does not support DSPark steps",
                        )
                    _validate_dspark_batch(speculative_step, batch)
                    simulation_batch = _dspark_simulation_batch(speculative_step)
                    prepared_batch = True
                elif prepared_batch:
                    simulation_batch = predictor_batch_from_result(
                        ret, expected_batch_size=len(batch.reqs)
                    )
                else:
                    simulation_batch = SimulationScheduleBatch(reqs=[])
                    if batch.forward_mode.is_extend():
                        for req in batch.reqs:
                            extend_length = getattr(req, "extend_input_len", None)
                            if extend_length is None:
                                extend_length = req.extend_len
                            simulation_batch.reqs.append(
                                ScheduleRequest(
                                    extend_length=extend_length,
                                    past_kv_length=req.prefix_len + len(req.output_ids),
                                )
                            )
                    elif batch.forward_mode.is_decode():
                        for req in batch.reqs:
                            simulation_batch.reqs.append(
                                ScheduleRequest(
                                    extend_length=1,
                                    past_kv_length=req.prefix_len + len(req.output_ids),
                                )
                            )

                if not simulation_batch.is_empty() or speculative_step is not None:
                    if not prepared_batch:
                        StateManager.inc_iteration()
                    pred_start = time.perf_counter()
                    predicted_latency = (
                        speculative_step.timing.seconds
                        if speculative_step is not None
                        else C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                            simulation_batch
                        )
                    )
                    predictor_time_cost = (
                        speculative_step.prediction_wall_seconds
                        if speculative_step is not None
                        else time.perf_counter() - pred_start
                    )
                    C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST += predictor_time_cost
                    if prepared_batch:
                        C_SchedulerHook.CURRENT_PREDICTOR_TIME_COST = (
                            predictor_time_cost
                        )
                    predicted_latency = float(predicted_latency)
                    if prepared_batch:
                        StateManager.inc_iteration()
                    C_SchedulerHook.CURRENT_PREDICTED_LATENCY = predicted_latency

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING:
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - StateManager.get_last_real_time_ts()
                        StateManager.set_last_real_time_ts(now)
                    else:
                        forward_latency = predicted_latency

                    if speculative_step is not None:
                        owner = getattr(self, _DSPARK_ACCOUNTING_OWNER, None)
                        if owner is None:
                            owner = object()
                            setattr(self, _DSPARK_ACCOUNTING_OWNER, owner)
                        setattr(
                            ret,
                            _DSPARK_ACCOUNTING,
                            _DSparkResultAccounting(
                                owner,
                                speculative_step,
                                forward_latency,
                                StateManager.get_current_inference_dur(),
                                C_SchedulerHook.GET_NEW_BATCH_PREFILL_TIME_COST,
                            ),
                        )
                        setattr(ret, _DSPARK_ACCOUNTED, False)
                        setattr(
                            self,
                            _DSPARK_PENDING_RESULTS,
                            getattr(self, _DSPARK_PENDING_RESULTS, 0) + 1,
                        )
                        if (
                            C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE
                            and C_SchedulerHook.COORDINATOR is None
                        ):
                            # The next host accounting interval can contain several
                            # launches. Exclude their replacement work once there,
                            # rather than when each corresponding result is retired.
                            setattr(
                                self,
                                _DSPARK_WALL_CREDIT,
                                getattr(self, _DSPARK_WALL_CREDIT, 0.0)
                                + predictor_time_cost,
                            )
                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.SIMULATION_BATCH = simulation_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            native_result = args[1] if len(args) > 1 else kwargs.get("result")
            prediction = dspark_result_from_native(native_result)
            accounting = getattr(native_result, _DSPARK_ACCOUNTING, None)
            if prediction is not None or accounting is not None:
                if (
                    type(accounting) is not _DSparkResultAccounting
                    or accounting.prediction is not prediction
                    or accounting.owner
                    is not getattr(self, _DSPARK_ACCOUNTING_OWNER, None)
                    or getattr(self, _DSPARK_PENDING_RESULTS, 0) <= 0
                    or getattr(native_result, _DSPARK_ACCOUNTED, None) is not False
                ):
                    raise PredictorError(
                        "invalid_dspark_step",
                        "DSPark result accounting is missing, changed or already consumed",
                    )
                _validate_dspark_batch(prediction, batch)
            simulation_batch = (
                _dspark_simulation_batch(prediction)
                if accounting is not None
                else C_SchedulerHook.SIMULATION_BATCH
            )
            current_inference_dur = (
                accounting.forward_duration
                if accounting is not None
                else StateManager.get_current_inference_dur()
            )
            previous_inference_dur = (
                accounting.previous_inference_duration
                if accounting is not None
                else StateManager.get_last_inference_dur()
            )
            predicted_latency = (
                prediction.timing.seconds
                if accounting is not None
                else C_SchedulerHook.CURRENT_PREDICTED_LATENCY
            )
            preprocess_duration = (
                accounting.preprocess_duration
                if accounting is not None
                else C_SchedulerHook.GET_NEW_BATCH_PREFILL_TIME_COST
            )
            driver = reference_driver()
            native_output_before = None
            reference_wall = 0.0

            def process_native_result():
                nonlocal native_output_before, reference_wall
                if driver is not None:
                    started = time.perf_counter()
                    try:
                        native_output_before = {
                            id(req): (len(req.output_ids), req.finished())
                            for req in batch.reqs
                        }
                        driver.before_result(batch, native_result)
                    except Exception as error:
                        driver.fail("native result preparation", error)
                        raise
                    reference_wall += time.perf_counter() - started
                try:
                    ret = original_process_batch_result(self, *args, **kwargs)
                    if driver is not None:
                        started = time.perf_counter()
                        driver.after_result()
                        reference_wall += time.perf_counter() - started
                        reference_wall += driver.take_callback_wall_seconds()
                    return ret
                except BaseException as error:
                    if driver is not None:
                        driver.fail("native result processing", error)
                    raise

            def consume_accounting():
                if accounting is not None:
                    setattr(native_result, _DSPARK_ACCOUNTED, True)
                    setattr(
                        self,
                        _DSPARK_PENDING_RESULTS,
                        getattr(self, _DSPARK_PENDING_RESULTS) - 1,
                    )

            if C_SchedulerHook.COORDINATOR is not None:

                def process_and_account():
                    ret = process_native_result()
                    consume_accounting()
                    return ret

                try:
                    ret = C_SchedulerHook.COORDINATOR.process_batch_result(
                        batch,
                        native_result,
                        duration=current_inference_dur,
                        process=process_and_account,
                    )
                except BaseException as error:
                    if driver is not None:
                        driver.fail("coordinated native result", error)
                    raise
                # OFFLINE coordination excludes replacement CPU work entirely.
                setattr(self, _DSPARK_WALL_CREDIT, 0.0)
                return ret
            blocked_l2_wall_dur = 0.0
            if (
                C_SchedulerHook.SIM_MODE == SimulationMode.BLOCKING
                and batch is not None
                and batch.reqs
            ):
                # Native result processing sends responses. Pace cache loads
                # before delivery so a dependent request cannot arrive early.
                blocked_l2_wall_dur = block_on_l2_load(
                    C_SchedulerHook.SIM_MODE,
                    effective_l2_load_delay(
                        StateManager.get_hicache_l2_load_dur(),
                        previous_inference_dur,
                        C_SchedulerHook.OVERLAP_SCHEDULE,
                    ),
                )
            process_batch_result_start = time.perf_counter()
            ret = process_native_result()
            if driver is not None:
                setattr(
                    self,
                    _DSPARK_WALL_CREDIT,
                    getattr(self, _DSPARK_WALL_CREDIT, 0.0) + reference_wall,
                )
            process_batch_result_end = time.perf_counter()

            if batch is not None:
                if len(batch.reqs) == 0 and accounting is None:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_load_stats = StateManager.pop_hicache_l2_load_stats()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                visible_l2_load_dur = effective_l2_load_delay(
                    hicache_l2_load_dur,
                    previous_inference_dur,
                    C_SchedulerHook.OVERLAP_SCHEDULE,
                )
                StateManager.step_global_clock(visible_l2_load_dur)
                StateManager.step_global_clock(current_inference_dur)
                # Step CPU overhead BEFORE recording latencies,
                # so current iter's CPU time is reflected in current iter's TTFT.
                now = time.time()
                predictor_wall_dur = (
                    getattr(self, _DSPARK_WALL_CREDIT, 0.0)
                    + (
                        0.0
                        if accounting is not None
                        else C_SchedulerHook.CURRENT_PREDICTOR_TIME_COST
                    )
                    if C_SchedulerHook.SIM_MODE == SimulationMode.OFFLINE
                    else 0
                )
                cpu_overhead = max(
                    now
                    - StateManager.get_last_real_time_ts()
                    - blocked_l2_wall_dur
                    - predictor_wall_dur,
                    0.0,
                )
                StateManager.step_global_clock(cpu_overhead)
                StateManager.set_last_real_time_ts(now)
                setattr(self, _DSPARK_WALL_CREDIT, 0.0)

                request_response_time = StateManager.get_global_clock()
                # Request statistics
                for req in batch.reqs:
                    if native_output_before is not None:
                        count_before, finished_before = native_output_before[id(req)]
                        emitted_output = (
                            not finished_before and len(req.output_ids) > count_before
                        )
                    else:
                        emitted_output = len(req.output_ids) != 0
                    if emitted_output:
                        req_stats = request_stats_manager.get_req_stats(req.rid)
                        req_stats.gen_token_latencies.append(
                            request_response_time
                            - req_stats.last_event_time  # queue duration
                        )
                        req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        **(
                            {
                                "forward_mode": simulation_batch.forward_mode,
                                "predicted_forward_latency": predicted_latency,
                            }
                            if simulation_batch.forward_mode is not None
                            else {}
                        ),
                        "requests": simulation_batch.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_blocking_wall_latency": blocked_l2_wall_dur,
                        **hicache_l2_load_stats,
                        "l2_backup_latency": hicache_l2_backup_dur,
                        "preprocess_latency": preprocess_duration,
                        "postprocess_latency": process_batch_result_end
                        - process_batch_result_start,
                        "cpu_overhead": cpu_overhead,
                    }
                )
            else:
                now = time.time()
                StateManager.step_global_clock(
                    max(
                        now
                        - StateManager.get_last_real_time_ts()
                        - getattr(self, _DSPARK_WALL_CREDIT, 0.0),
                        0.0,
                    )
                )
                StateManager.set_last_real_time_ts(now)
                setattr(self, _DSPARK_WALL_CREDIT, 0.0)

            consume_accounting()

            return ret

        def override_profile(
            req, *args, _scheduler=None, _dp_payload=None, _dp_writer=True, **kwargs
        ):
            bridge = C_SchedulerHook.COORDINATOR
            if _dp_payload is None and callable(
                getattr(bridge, "profile_boundary", None)
            ):
                return bridge.profile_boundary(
                    req,
                    snapshot=lambda: {
                        "profile_stats": deepcopy(
                            request_stats_manager.get_all_req_stats()
                        ),
                        "iterations": deepcopy(C_SchedulerHook.ITERATION_STATS),
                        "predictor_metrics": C_SchedulerHook.INFERENCE_PREDICTOR.get_metrics(),
                    },
                    execute=lambda payload, writer: override_profile(
                        req,
                        *args,
                        _scheduler=_scheduler,
                        _dp_payload=payload,
                        _dp_writer=writer,
                        **kwargs,
                    ),
                )
            if getattr(_scheduler, _DSPARK_PENDING_RESULTS, 0):
                ProfileReqOutput = importlib.import_module(
                    "sglang.srt.managers.io_struct"
                ).ProfileReqOutput
                return ProfileReqOutput(
                    success=False,
                    message="DSPark results must drain before resetting profile accounting",
                )
            is_start_profile = req.req_type.name == "START_PROFILE"
            stats: list[RequestStats] = []
            for item in (
                request_stats_manager.get_all_req_stats()
                if _dp_payload is None
                else _dp_payload["profile_stats"]
            ):
                if item.rid is not None and item.input_length > 0:
                    stats.append(deepcopy(item))

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            if _dp_writer:
                os.makedirs(output_dir, exist_ok=True)

            if len(stats) > 0 and _dp_writer:
                min_created_time = stats[0].created_time
                # Align timestamps
                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time

                metrics = calc_metrics(stats)
                metrics["time_cost"] = (
                    time.time() - StateManager.get_last_flush_time_ts()
                )
                metrics["predictor_time_cost"] = (
                    C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST
                )
                metrics.update(
                    calc_iteration_metrics(
                        C_SchedulerHook.ITERATION_STATS
                        if _dp_payload is None
                        else _dp_payload["iterations"],
                        metrics,
                    )
                )
                metrics.update(
                    C_SchedulerHook.INFERENCE_PREDICTOR.get_metrics()
                    if _dp_payload is None
                    else _dp_payload["local_predictor_metrics"][0]
                )
                if _dp_payload is not None:
                    metrics["attention_dp_local_predictor_metrics"] = _dp_payload[
                        "local_predictor_metrics"
                    ]
                memory_budget = ConfigManager.get_memory_budget_metrics()
                if memory_budget is not None:
                    metrics["memory_budget"] = memory_budget

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in (
                            C_SchedulerHook.ITERATION_STATS
                            if _dp_payload is None
                            else _dp_payload["iterations"]
                        ):
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
                    ProfileReqOutput = importlib.import_module(
                        "sglang.srt.managers.io_struct"
                    ).ProfileReqOutput
                    return ProfileReqOutput(
                        success=False,
                        message=f"Failed to save simulation results: {e}",
                    )
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            StateManager.set_last_flush_time_ts(time.time())
            request_stats_manager.reset()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST = 0
            C_SchedulerHook.CURRENT_PREDICTOR_TIME_COST = 0
            C_SchedulerHook.REQ_DISPATCHER.reset()
            C_SchedulerHook.REQ_DISPATCHER.profile_active = is_start_profile
            C_SchedulerHook.INFERENCE_PREDICTOR.reset_metrics()
            # This successful reset starts a new accounting interval. Idle
            # selection/retirement emulation from the previous interval must
            # not reduce the next profile's native CPU overhead.
            setattr(_scheduler, _DSPARK_WALL_CREDIT, 0.0)
            driver = reference_driver()
            if driver is not None:
                driver.take_callback_wall_seconds()

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )
            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(
                success=True,
                message=json.dumps(result),
            )

        def wrapped_init_request_dispatcher(self, *args, **kwargs):
            ret = original_init_request_dispatcher(self, *args, **kwargs)

            _request_dispatcher = getattr(self, "_request_dispatcher", None)

            if _request_dispatcher is not None:
                for ty in _request_dispatcher._mapping.keys():
                    if ty.__name__ == "ProfileReq":
                        _request_dispatcher._mapping[ty] = partial(
                            override_profile, _scheduler=self
                        )
            return ret

        def wrapped_on_idle(self, *args, **kwargs):
            ret = original_on_idle(self, *args, **kwargs)
            if C_SchedulerHook.COORDINATOR is not None:
                C_SchedulerHook.COORDINATOR.on_idle()
            return ret

        target.on_idle = wrapped_on_idle
        target.event_loop_overlap = override_event_loop_overlap
        if original_get_next_batch is not None:
            target.get_next_batch_to_run = wrapped_get_next_batch
        if original_overlap_decision is not None:
            target.is_disable_overlap_for_batch = wrapped_overlap_decision
        if original_release_aborted is not None:
            target._release_aborted_request = wrapped_release_aborted
        target.__init__ = wrapped_init
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target._prefetch_kvcache = wrapped_prefetch_kvcache
        target.init_request_dispatcher = wrapped_init_request_dispatcher

        if original_recv_requests:
            target.recv_requests = wrapped_recv_requests

"""Hooks for collecting requests and synchronized steps from a real SGLang server."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Iterable

from sglang_simulator.hook import BaseHook, install_class_hooks
from sglang_simulator.hook.class_hook_entry import is_class_hook_matched

logger = logging.getLogger(__name__)

COLLECTION_OUTPUT_DIR_ENV = "SGLANG_SIMULATOR_COLLECTION_DIR"


@dataclass
class RequestRecord:
    rid: str = ""
    created_time: float | None = None
    client_created_time: float | None = None
    server_created_time: float | None = None
    queue_start: float = 0.0
    queue_end: float = 0.0
    completed: bool = False
    output_length: int = 0
    input_length: int = 0
    before_adder_device_hit_len: int = 0
    final_device_hit_len: int = 0
    final_host_hit_len: int = 0
    final_disk_hit_len: int = 0
    last_event_time: float = 0.0
    gen_token_latencies: list[float] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)


@dataclass
class CollectionState:
    active: bool = False
    requests: dict[str, RequestRecord] = field(default_factory=dict)
    schedule_batches: list[dict[str, Any]] = field(default_factory=list)

    def start(self) -> None:
        if self.active:
            raise RuntimeError("baseline collection is already active")
        self.requests.clear()
        self.schedule_batches.clear()
        self.active = True

    def get_request(self, rid: str) -> RequestRecord:
        record = self.requests.get(rid)
        if record is None:
            record = RequestRecord(rid=rid)
            self.requests[rid] = record
        return record

    def stop_and_export(self) -> list[Path]:
        if not self.active:
            raise RuntimeError("baseline collection is not active")
        output_dir = os.getenv(COLLECTION_OUTPUT_DIR_ENV)
        if not output_dir:
            raise RuntimeError(
                f"{COLLECTION_OUTPUT_DIR_ENV} is required for baseline collection"
            )

        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        prefix = _rank_prefix()
        exports = {
            "request": (_normalized_request(r) for r in self.requests.values()),
            "raw_request": (asdict(r) for r in self.requests.values()),
            "schedule_batch": iter(self.schedule_batches),
        }
        paths = []
        try:
            for kind, rows in exports.items():
                path = root / f"{prefix}.{kind}.jsonl"
                with path.open("x", encoding="utf-8") as output:
                    paths.append(path)
                    for row in rows:
                        output.write(json.dumps(row) + "\n")
        except (OSError, TypeError, ValueError):
            # Keep in-memory records for retry; never remove an existing file.
            for path in paths:
                try:
                    path.unlink()
                except OSError:
                    logger.warning(
                        "Failed to remove partial baseline file %s", path, exc_info=True
                    )
            raise

        self.active = False
        self.requests.clear()
        self.schedule_batches.clear()
        return paths


def _rank_prefix() -> str:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    prefix = f"TP{parallel.tp_rank}"
    for size_name, rank_name, label in (
        ("dp_size", "dp_rank", "DP"),
        ("pp_size", "pp_rank", "PP"),
        ("moe_ep_size", "moe_ep_rank", "EP"),
    ):
        if getattr(parallel, size_name) > 1:
            prefix += f"-{label}{getattr(parallel, rank_name)}"
    return prefix


def _normalized_request(record: RequestRecord) -> dict[str, Any]:
    timestamp = record.created_time
    if timestamp is None:
        timestamp = record.queue_start
    return {
        "rid": record.rid,
        "timestamp": timestamp * 1000,
        "input_length": record.input_length,
        "output_length": record.output_length,
        "device_cache_hit_length": record.before_adder_device_hit_len,
        "host_cache_hit_length": record.final_host_hit_len - record.final_disk_hit_len,
        "disk_cache_hit_length": record.final_disk_hit_len,
    }


COLLECTION_STATE = CollectionState()


def _custom_params(sampling_params: Any) -> dict[str, Any]:
    if isinstance(sampling_params, dict):
        return sampling_params.get("custom_params") or {}
    return getattr(sampling_params, "custom_params", None) or {}


def _stamp_created_time(tokenized_obj: Any) -> None:
    from sglang.srt.observability.req_time_stats import convert_time_to_realtime

    time_stats = getattr(tokenized_obj, "time_stats", None)
    created_time = getattr(time_stats, "created_time", None)
    if not created_time:
        return
    sampling_params = tokenized_obj.sampling_params
    params = dict(_custom_params(sampling_params))
    # Convert SGLang's perf_counter arrival to Unix time before IPC.
    params["server_created_time"] = convert_time_to_realtime(created_time)
    if isinstance(sampling_params, dict):
        sampling_params["custom_params"] = params
    else:
        sampling_params.custom_params = params


def _iter_received_requests(recv_reqs: Iterable[Any]) -> Iterable[Any]:
    for recv_req in recv_reqs:
        if recv_req.__class__.__name__.startswith("BatchTokenized"):
            yield from recv_req
        else:
            yield recv_req


def _record_received_requests(recv_reqs: Iterable[Any]) -> None:
    if not COLLECTION_STATE.active:
        return
    from sglang.srt.managers.utils import is_health_check_generate_req

    recv_time = time.time()
    for req in _iter_received_requests(recv_reqs):
        # Control messages (abort, profile, RPC, etc.) must not become requests.
        if not hasattr(req, "sampling_params") or not getattr(req, "rid", None):
            continue
        if is_health_check_generate_req(req):
            continue
        record = COLLECTION_STATE.get_request(str(req.rid))
        record.queue_start = recv_time
        params = _custom_params(req.sampling_params)
        record.server_created_time = params.get("server_created_time")
        record.client_created_time = params.get("client_created_time")
        record.created_time = next(
            value
            for value in (
                record.client_created_time,
                record.server_created_time,
                recv_time,
            )
            if value is not None
        )
        record.last_event_time = record.server_created_time or recv_time
        record.input_ids = list(getattr(req, "input_ids", None) or [])
        record.input_length = len(record.input_ids)


def _batch_request_lengths(batch: Any) -> list[tuple[int, int]]:
    """Read prepared batch vectors; Req lengths may lag in decode/mixed batches."""
    if not batch.reqs:
        return []
    if batch.forward_mode.is_decode():
        lengths = [(1, max(length - 1, 0)) for length in batch.seq_lens_cpu.tolist()]
    else:
        lengths = list(zip(batch.extend_lens, batch.prefix_lens, strict=True))
    if len(lengths) != len(batch.reqs):
        raise ValueError("ScheduleBatch length vectors and reqs must have equal size")
    return lengths


def _synchronize_device() -> None:
    # Defer SGLang imports until the class hooks are installed.
    from sglang.srt.platforms import current_platform

    current_platform.synchronize()


def _profile_output(message: str, success: bool = True) -> Any:
    from sglang.srt.managers.io_struct import ProfileReqOutput

    return ProfileReqOutput(success=success, message=message)


def _handle_profile_request(recv_req: Any) -> Any:
    try:
        if recv_req.req_type.name == "START_PROFILE":
            COLLECTION_STATE.start()
            return _profile_output("Baseline collection started")
        paths = COLLECTION_STATE.stop_and_export()
        return _profile_output(
            "Baseline collection exported: " + ", ".join(path.name for path in paths)
        )
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        # Report API errors without terminating the scheduler.
        return _profile_output(str(exc), success=False)


class TokenizerManagerCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target: Any) -> None:
        original_one = target._send_one_request
        original_batch = target._send_batch_request

        @wraps(original_one)
        async def send_one(
            self: Any, tokenized_obj: Any, *args: Any, **kwargs: Any
        ) -> Any:
            _stamp_created_time(tokenized_obj)
            return await original_one(self, tokenized_obj, *args, **kwargs)

        @wraps(original_batch)
        async def send_batch(
            self: Any, tokenized_objs: list[Any], *args: Any, **kwargs: Any
        ) -> Any:
            for obj in tokenized_objs:
                _stamp_created_time(obj)
            return await original_batch(self, tokenized_objs, *args, **kwargs)

        target._send_one_request = send_one
        target._send_batch_request = send_batch


class SchedulerCollectionHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    @classmethod
    def hook(cls, target: Any) -> None:
        from sglang.srt.managers.utils import is_health_check_generate_req

        original_input = target.process_input_requests
        original_run = target.run_batch
        original_result = target.process_batch_result
        original_profiler = target.init_profiler

        @wraps(original_input)
        def process_input(
            self: Any, recv_reqs: list[Any], *args: Any, **kwargs: Any
        ) -> Any:
            _record_received_requests(recv_reqs)
            return original_input(self, recv_reqs, *args, **kwargs)

        @wraps(original_run)
        def run_batch(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
            if not COLLECTION_STATE.active:
                return original_run(self, batch, *args, **kwargs)

            # Workers may mutate the batch, so snapshot before forward.
            requests = tuple(batch.reqs or ())
            scheduled_time = time.time()
            for req in requests:
                record = COLLECTION_STATE.requests.get(str(req.rid))
                if record is not None and record.queue_end == 0:
                    record.queue_end = scheduled_time
                    record.input_ids = list(req.origin_input_ids)
                    record.input_length = len(record.input_ids)
                    # Preserve SGLang's first-chunk counts across later chunks/retractions.
                    device = int(req.cached_tokens_device)
                    host = int(req.cached_tokens_host)
                    storage = int(req.cached_tokens_storage)
                    record.before_adder_device_hit_len = device
                    record.final_device_hit_len = device + host + storage
                    record.final_host_hit_len = host + storage
                    record.final_disk_hit_len = storage

            # A mixed step's latency includes the probe: exclude the entire step.
            if any(is_health_check_generate_req(req) for req in requests):
                return original_run(self, batch, *args, **kwargs)

            lengths = _batch_request_lengths(batch)
            row = {
                "forward_mode": int(batch.forward_mode),
                "request_infos": [
                    {
                        "rid": str(req.rid),
                        "extend_input_len": extend,
                        "prefix_indices_len": past,
                        "output_ids_len": len(req.output_ids),
                    }
                    for req, (extend, past) in zip(requests, lengths, strict=True)
                ],
            }

            _synchronize_device()
            start = time.perf_counter()
            result = original_run(self, batch, *args, **kwargs)
            _synchronize_device()
            end = time.perf_counter()
            row.update(
                start_timestamp=start, end_timestamp=end, iter_latency=end - start
            )
            COLLECTION_STATE.schedule_batches.append(row)
            return result

        @wraps(original_result)
        def process_result(self: Any, batch: Any, *args: Any, **kwargs: Any) -> Any:
            if not COLLECTION_STATE.active:
                return original_result(self, batch, *args, **kwargs)
            requests = [(req, len(req.output_ids)) for req in (batch.reqs or ())]
            result = original_result(self, batch, *args, **kwargs)
            event_time = time.time()
            for req, previous_length in requests:
                record = COLLECTION_STATE.requests.get(str(req.rid))
                if record is None:
                    continue
                record.output_length = len(req.output_ids)
                if record.output_length > previous_length:
                    record.gen_token_latencies.append(
                        max(event_time - record.last_event_time, 0.0)
                    )
                    record.last_event_time = event_time
                if req.finished():
                    record.output_ids = list(req.output_ids)
                    record.completed = True
            return result

        @wraps(original_profiler)
        def init_profiler(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_profiler(self, *args, **kwargs)
            self.profiler_manager._profile = _handle_profile_request
            return result

        target.process_input_requests = process_input
        target.run_batch = run_batch
        target.process_batch_result = process_result
        target.init_profiler = init_profiler


_FRONTEND_HOOKS_INSTALLED = False
_SCHEDULER_HOOKS_INSTALLED = False


def install_frontend_collection_hooks() -> None:
    global _FRONTEND_HOOKS_INSTALLED
    if not _FRONTEND_HOOKS_INSTALLED:
        install_class_hooks(TokenizerManagerCollectionHook)
        _FRONTEND_HOOKS_INSTALLED = True


def install_scheduler_collection_hooks() -> None:
    global _SCHEDULER_HOOKS_INSTALLED
    if not _SCHEDULER_HOOKS_INSTALLED:
        install_class_hooks(SchedulerCollectionHook)
        _SCHEDULER_HOOKS_INSTALLED = True


def validate_collection_hook(hook: type[BaseHook]) -> None:
    if is_class_hook_matched(hook):
        return
    raise RuntimeError(
        "Required SGLang Simulator hook did not match an imported SGLang class: "
        f"{hook.__name__} ({hook.HOOK_MODULE_NAME}.{hook.HOOK_CLASS_NAME}). "
        "The collector must be adapted to this SGLang revision."
    )


def run_collection_scheduler_process(*args: Any, **kwargs: Any) -> Any:
    """Install collection hooks before importing and running Scheduler."""
    install_scheduler_collection_hooks()
    from sglang.srt.managers.scheduler import run_scheduler_process

    # TokenizerManager belongs to the API process; validate only Scheduler here.
    validate_collection_hook(SchedulerCollectionHook)
    return run_scheduler_process(*args, **kwargs)

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Request Time Stats."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    TokenizerMetricsCollector,
)
from sglang.srt.observability.trace import (
    SpanAttributes,
    TraceNullContext,
    TraceReqContext,
    TraceSliceContext,
    get_global_tracing_enabled,
)
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")


logger = logging.getLogger(__name__)

# Reduce system time calls by computing time.time() based on calibrated perf_counter() values.
global_diff_realtime_monotonic = time.time() - time.perf_counter()


def calibrate_time_diff():
    # due to NTP, the diff between time.time() and time.perf_counter() can change
    # periodically calibrate the diff
    global global_diff_realtime_monotonic
    global_diff_realtime_monotonic = time.time() - time.perf_counter()


def real_time():
    return time.time()


def monotonic_time():
    return time.perf_counter()


def convert_time_to_realtime(time_value: float) -> float:
    # note: Within the time scale of a single request's latency,
    # we assume that the diff does not change significantly.
    return time_value + global_diff_realtime_monotonic


def convert_time_to_realtime_ns(time_value: float) -> int:
    return int((time_value + global_diff_realtime_monotonic) * 1e9)


def convert_time_cross_thread(
    time_value: float, old_diff: float, new_diff: float
) -> float:
    # note: precision loss
    return time_value + old_diff - new_diff


@dataclass
class RequestStageConfig:
    stage_name: str
    level: int = 0
    # whether to call metrics_collector.observe_per_stage_req_latency
    metrics_is_observed: bool = False


class RequestStage:
    # Tokenizer/gRPC Server
    TOKENIZE = RequestStageConfig(
        "tokenize",
        level=1,
    )
    API_SERVER_DISPATCH = RequestStageConfig(
        "dispatch",
        level=2,
    )

    # DP controller
    DC_DISPATCH = RequestStageConfig(
        "dc_dispatch",
        level=2,
    )

    # common/non-disaggregation
    REQUEST_PROCESS = RequestStageConfig(
        "request_process",
        level=2,
        metrics_is_observed=True,
    )
    PREFILL_WAITING = RequestStageConfig(
        "prefill_waiting",
        level=1,
        # equal to "observe_queue_time"
        metrics_is_observed=False,
    )
    DECODE_FORWARD = RequestStageConfig(
        "decode_forward",
        level=1,
    )
    DECODE_LOOP = RequestStageConfig(
        "decode_loop",
        level=3,
    )
    PREFILL_FORWARD = RequestStageConfig(
        "prefill_forward",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_CHUNKED_FORWARD = RequestStageConfig(
        "chunked_prefill",
        level=3,
        metrics_is_observed=True,
    )

    # disaggregation prefill
    PREFILL_PREPARE = RequestStageConfig(
        "prefill_prepare",
        level=1,
    )
    PREFILL_BOOTSTRAP = RequestStageConfig(
        "prefill_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_TRANSFER_KV_CACHE = RequestStageConfig(
        "prefill_transfer_kv_cache",
        level=1,
        metrics_is_observed=True,
    )

    # disaggregation decode
    DECODE_PREPARE = RequestStageConfig(
        "decode_prepare",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_BOOTSTRAP = RequestStageConfig(
        "decode_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_WAITING = RequestStageConfig(
        "decode_waiting",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_TRANSFERRED = RequestStageConfig(
        "decode_transferred",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_FAKE_OUTPUT = RequestStageConfig(
        "fake_output",
        level=3,
        metrics_is_observed=True,
    )
    DECODE_QUICK_FINISH = RequestStageConfig(
        "quick_finish",
        level=1,
        metrics_is_observed=True,
    )

    # mini lb
    MINI_LB_LAUNCH = RequestStageConfig(
        "mini_lb_launch",
        level=1,
    )

    WAIT_PD_FINISH = RequestStageConfig(
        "wait_pd_finish",
        level=2,
    )

    # other
    ANONYMOUS = RequestStageConfig("")


@dataclass
class ReqTimeStatsBase:
    enable_metrics: bool = False
    metrics_collector: Optional[
        Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
    ] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = field(
        default_factory=TraceNullContext
    )
    disagg_mode: DisaggregationMode = DisaggregationMode.NULL
    diff_realtime_monotonic: float = 0.0

    @classmethod
    def new_from_obj(cls, obj: ReqTimeStatsBase, *args, **kwargs) -> "ReqTimeStatsBase":
        calibrate_time_diff()
        new_obj = cls(*args, **kwargs)
        if obj is None:
            return new_obj
        for key, value in obj.__dict__.items():
            if hasattr(new_obj, key):
                setattr(new_obj, key, value)

        new_obj.trace_ctx.rebuild_thread_context()

        return new_obj

    def disagg_mode_str(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL:
            return "unified"
        elif self.disagg_mode == DisaggregationMode.DECODE:
            return "decode"
        elif self.disagg_mode == DisaggregationMode.PREFILL:
            return "prefill"
        else:
            return "unknown"

    def set_metrics_collector(
        self, collector: Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
    ):
        if collector:
            self.enable_metrics = True
            self.metrics_collector = collector

    def observe_per_stage_req_latency(self, stage: RequestStageConfig, latency: float):
        if self.enable_metrics and stage.metrics_is_observed:
            self.metrics_collector.observe_per_stage_req_latency(
                stage.stage_name, latency
            )

    def init_trace_ctx(
        self,
        rid: str,
        bootstrap_room: Optional[int],
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.trace_ctx = TraceReqContext(
            rid=rid,
            bootstrap_room=bootstrap_room,
            role=self.disagg_mode_str(),
            module_name="request",
            external_trace_header=external_trace_header,
        )

        if not self.trace_ctx.tracing_enable:
            self.trace_ctx = TraceNullContext()

    def trace_slice(
        self,
        stage: RequestStageConfig,
        start_time: float,
        end_time: float,
        attrs: Optional[Dict] = None,
    ):
        if self.trace_ctx.tracing_enable:
            _slice = TraceSliceContext(
                slice_name=stage.stage_name,
                start_time_ns=convert_time_to_realtime_ns(start_time),
                end_time_ns=convert_time_to_realtime_ns(end_time),
                level=stage.level,
                attrs=attrs,
            )
            self.trace_ctx.trace_slice(_slice)

    def __getstate__(self) -> object:
        # The object is propagated to other processes via serialization and deserialization methods,
        # requiring the metric collector to be reconfigured.
        return {
            "disagg_mode": self.disagg_mode,
            "enable_metrics": False,
            "trace_ctx": self.trace_ctx,
            "diff_realtime_monotonic": global_diff_realtime_monotonic,
        }

    def __setstate__(self, state: object):
        for key in state.keys():
            if key.endswith("time"):
                state[key] = convert_time_cross_thread(
                    state[key],
                    state["diff_realtime_monotonic"],
                    global_diff_realtime_monotonic,
                )
        self.__dict__.update(state)


@dataclass
class APIServerReqTimeStats(ReqTimeStatsBase):
    # get by time.perf_counter()
    created_time: float = 0.0
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    tokenize_finish_time: float = 0.0
    api_server_dispatch_time: float = 0.0
    api_server_dispatch_finish_time: float = 0.0
    response_sent_to_client_time: float = 0.0

    def __getstate__(self) -> object:
        state = {}
        # send to DP controller or Scheduler
        # If necessary, can propagate the timestamp here, for example:
        # state = {
        #    "created_time": self.created_time,
        #    "api_server_dispatch_time": self.api_server_dispatch_time,
        # }
        state.update(super().__getstate__())
        return state

    def set_created_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.created_time = ts

        self.trace_ctx.trace_req_start(convert_time_to_realtime_ns(ts))

    def set_finished_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.finished_time = ts

        self.trace_ctx.trace_req_finish(convert_time_to_realtime_ns(ts))

    def set_first_token_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.first_token_time = ts
        self.last_time = ts

    def set_last_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.last_time = ts

    def set_tokenize_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.tokenize_finish_time = ts

        stage = RequestStage.TOKENIZE
        self.trace_slice(stage, self.created_time, ts)

    def set_api_server_dispatch_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.api_server_dispatch_time = ts

        self.trace_ctx.trace_slice_start(
            RequestStage.API_SERVER_DISPATCH.stage_name,
            RequestStage.API_SERVER_DISPATCH.level,
            convert_time_to_realtime_ns(ts),
        )

    def set_api_server_dispatch_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.api_server_dispatch_finish_time = ts

        self.trace_ctx.trace_slice_end(
            RequestStage.API_SERVER_DISPATCH.stage_name,
            RequestStage.API_SERVER_DISPATCH.level,
            convert_time_to_realtime_ns(ts),
            thread_finish_flag=True,
        )

    def set_response_sent_to_client_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.response_sent_to_client_time = ts

    def get_interval(self):
        return time.perf_counter() - self.last_time

    def get_first_token_latency(self):
        return self.first_token_time - self.created_time

    def get_e2e_latency(self):
        return self.finished_time - self.created_time

    def get_decode_latency(self):
        return self.finished_time - self.first_token_time

    def get_response_sent_to_client_realtime(self):
        return convert_time_to_realtime(self.response_sent_to_client_time)

    def convert_to_output_meta_info(
        self, scheduler_time_stats=None, completion_tokens=0
    ):
        meta_info = {}
        if self.created_time > 0.0:
            meta_info["request_received_ts"] = convert_time_to_realtime(
                self.created_time
            )
        if self.api_server_dispatch_finish_time > 0.0:
            meta_info["api_server_dispatch_finish_ts"] = convert_time_to_realtime(
                self.api_server_dispatch_finish_time
            )
        if self.response_sent_to_client_time > 0.0:
            meta_info["response_sent_to_client_ts"] = convert_time_to_realtime(
                self.response_sent_to_client_time
            )
        if self.finished_time > 0.0:
            meta_info["request_finished_ts"] = convert_time_to_realtime(
                self.finished_time
            )

        if (
            scheduler_time_stats
            and hasattr(scheduler_time_stats, "forward_entry_time")
            and self.finished_time > 0.0
        ):
            meta_info["inference_time"] = (
                self.finished_time - scheduler_time_stats.forward_entry_time
            )

        decode_latency = self.get_decode_latency()
        if decode_latency > 0.0 and completion_tokens > 0:
            meta_info["decode_throughput"] = completion_tokens / decode_latency
        return meta_info

    def convert_to_gen_ai_span_attrs(self):
        span_attrs = {}
        if self.first_token_time and self.created_time:
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN] = (
                self.first_token_time - self.created_time
            )

        if self.finished_time and self.created_time:
            span_attrs[SpanAttributes.GEN_AI_LATENCY_E2E] = (
                self.finished_time - self.created_time
            )

        if self.first_token_time and self.finished_time:
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE] = (
                self.finished_time - self.first_token_time
            )

        if self.api_server_dispatch_finish_time and self.finished_time:
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE] = (
                self.finished_time - self.api_server_dispatch_finish_time
            )

        if self.api_server_dispatch_finish_time and self.first_token_time:
            span_attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL] = (
                self.first_token_time - self.api_server_dispatch_finish_time
            )

        return span_attrs


@dataclass
class DPControllerReqTimeStats(ReqTimeStatsBase):
    # propagated from tokenizer/grpc_server, get by time.perf_counter()
    created_time: float = 0.0
    api_server_dispatch_time: float = 0.0

    # new timestamp, get by time.perf_counter()
    dc_dispatch_time: float = 0.0
    dc_dispatch_finish_time: float = 0.0

    def __getstate__(self) -> object:
        state = {}
        # send to Scheduler
        # If necessary, can propagate the timestamp here, for example:
        # state = {
        #     "created_time": self.created_time,
        #     "api_server_dispatch_time": self.api_server_dispatch_time,
        #     "dc_dispatch_time": self.dc_dispatch_time,
        # }
        state.update(super().__getstate__())
        return state

    def set_dp_dispatch_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.dc_dispatch_time = ts

        self.trace_ctx.trace_slice_start(
            RequestStage.DC_DISPATCH.stage_name,
            RequestStage.DC_DISPATCH.level,
            convert_time_to_realtime_ns(ts),
        )

    def set_dp_dispatch_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.dc_dispatch_finish_time = ts

        self.trace_ctx.trace_slice_end(
            RequestStage.DC_DISPATCH.stage_name,
            RequestStage.DC_DISPATCH.level,
            convert_time_to_realtime_ns(ts),
            thread_finish_flag=True,
        )


@dataclass
class SchedulerReqTimeStats(ReqTimeStatsBase):
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    # Placeholder: not used currently
    # propagated from tokenizer/grpc_server or dp controller
    created_time: float = 0.0
    api_server_dispatch_time: float = 0.0
    dc_dispatch_time: float = 0.0

    # common, get by time.perf_counter()
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    prefill_run_batch_start_time: float = 0.0
    prefill_run_batch_end_time: float = 0.0
    prefill_finished_time: float = 0.0
    completion_time: float = 0.0

    # prefill node, get by time.perf_counter()
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    prefill_kv_transfer_finish_time: float = 0.0

    # decode node, get by time.perf_counter()
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0
    decode_prebuilt_finish_time: float = 0.0

    # only for request tracing
    scheduler_recv_time: float = 0.0
    last_chunked_prefill_finish_time: float = 0.0
    last_decode_finish_time: float = 0.0
    decode_ct: int = 0
    last_decode_scheduled_time: float = 0.0
    last_forward_entry_time: float = 0.0
    last_prefill_finished_time: float = 0.0

    # other
    transfer_speed_gb_s: float = 0.0
    transfer_total_mb: float = 0.0
    # Number of prefill retries for this request
    prefill_retry_count: int = 0

    def __getstate__(self) -> object:
        # send to detokenizer/tokenizer
        if not self.enable_metrics:
            return {}

        state = {
            "wait_queue_entry_time": self.wait_queue_entry_time,
            "forward_entry_time": self.forward_entry_time,
            "prefill_run_batch_start_time": self.prefill_run_batch_start_time,
            "prefill_run_batch_end_time": self.prefill_run_batch_end_time,
            "prefill_finished_time": self.prefill_finished_time,
            "diff_realtime_monotonic": global_diff_realtime_monotonic,
        }
        return state

    def set_scheduler_recv_time(self, ts=None):
        calibrate_time_diff()
        if ts is None:
            ts = time.perf_counter()
        self.scheduler_recv_time = ts

    def set_retract_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        # retract
        self.last_forward_entry_time = 0.0
        self.last_prefill_finished_time = 0.0
        self.last_chunked_prefill_finish_time = 0.0
        self.last_decode_finish_time = 0.0
        self.last_decode_scheduled_time = 0.0

        self.trace_ctx.trace_event("retract", 1, convert_time_to_realtime_ns(ts))

    def set_wait_queue_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        if self.wait_queue_entry_time == 0.0:
            if self.enable_metrics or self.trace_ctx.tracing_enable:
                if self.disagg_mode == DisaggregationMode.PREFILL:
                    stage = RequestStage.PREFILL_BOOTSTRAP
                    slice_start_time = self.prefill_bootstrap_queue_entry_time
                elif self.disagg_mode == DisaggregationMode.DECODE:
                    stage = RequestStage.DECODE_TRANSFERRED
                    slice_start_time = self.decode_transfer_queue_entry_time
                else:
                    stage = RequestStage.REQUEST_PROCESS
                    slice_start_time = self.scheduler_recv_time

                self.observe_per_stage_req_latency(stage, ts - slice_start_time)
                self.trace_slice(stage, slice_start_time, ts)
        else:
            self.set_retract_time(ts)

        self.wait_queue_entry_time = ts

    def set_forward_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        if self.forward_entry_time == 0.0:
            self.forward_entry_time = ts
            self.last_forward_entry_time = ts

            if self.enable_metrics:
                self.metrics_collector.observe_queue_time(self.get_queueing_time())

            if self.enable_metrics or self.trace_ctx.tracing_enable:
                if self.disagg_mode == DisaggregationMode.DECODE:
                    stage = RequestStage.DECODE_WAITING
                else:
                    stage = RequestStage.PREFILL_WAITING
                slice_start_time = self.wait_queue_entry_time

                self.observe_per_stage_req_latency(stage, ts - slice_start_time)
                self.trace_slice(stage, slice_start_time, ts)

                if self.disagg_mode == DisaggregationMode.DECODE:
                    self.trace_ctx.trace_slice_start(
                        RequestStage.DECODE_FORWARD.stage_name,
                        RequestStage.DECODE_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
                else:
                    self.trace_ctx.trace_slice_start(
                        RequestStage.PREFILL_FORWARD.stage_name,
                        RequestStage.PREFILL_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
        elif self.last_forward_entry_time == 0.0:
            self.last_forward_entry_time = ts

    def set_prefill_run_batch_start_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.prefill_run_batch_start_time = ts

    def set_prefill_run_batch_end_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.prefill_run_batch_end_time = ts

    def set_last_chunked_prefill_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        last_time = self.last_chunked_prefill_finish_time
        self.last_chunked_prefill_finish_time = ts

        if last_time == 0.0:
            last_time = self.last_forward_entry_time

        stage = RequestStage.PREFILL_CHUNKED_FORWARD
        self.observe_per_stage_req_latency(stage, ts - last_time)
        self.trace_slice(stage, last_time, ts)

    def set_prefill_finished_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        if self.prefill_finished_time == 0.0:
            self.prefill_finished_time = ts
            self.last_prefill_finished_time = ts

            stage = RequestStage.PREFILL_FORWARD
            self.observe_per_stage_req_latency(stage, ts - self.last_forward_entry_time)

            if self.trace_ctx.tracing_enable:
                if self.last_chunked_prefill_finish_time > 0:
                    self.trace_slice(
                        RequestStage.PREFILL_CHUNKED_FORWARD,
                        self.last_chunked_prefill_finish_time,
                        ts,
                    )

                self.trace_ctx.trace_slice_end(
                    stage.stage_name, stage.level, convert_time_to_realtime_ns(ts)
                )
                if (
                    self.disagg_mode == DisaggregationMode.NULL
                    and self.last_decode_scheduled_time > 0
                ):
                    self.trace_ctx.trace_slice_start(
                        RequestStage.DECODE_FORWARD.stage_name,
                        RequestStage.DECODE_FORWARD.level,
                        convert_time_to_realtime_ns(ts),
                    )
        elif self.last_prefill_finished_time == 0.0:
            # retract
            self.last_prefill_finished_time = ts
            if self.last_chunked_prefill_finish_time > 0:
                self.trace_slice(
                    RequestStage.PREFILL_CHUNKED_FORWARD,
                    self.last_chunked_prefill_finish_time,
                    ts,
                )
            else:
                self.trace_slice(
                    RequestStage.PREFILL_FORWARD, self.last_forward_entry_time, ts
                )

    def set_last_decode_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        last_time = self.last_decode_finish_time
        self.last_decode_finish_time = ts

        if self.enable_metrics or self.trace_ctx.tracing_enable:
            if last_time == 0.0:
                if self.disagg_mode == DisaggregationMode.DECODE:
                    last_time = self.decode_prebuilt_finish_time
                else:
                    if (
                        self.last_decode_scheduled_time
                        < self.last_prefill_finished_time
                    ):
                        last_time = self.last_prefill_finished_time
                    else:
                        last_time = self.last_decode_scheduled_time
            stage = RequestStage.DECODE_LOOP
            self.observe_per_stage_req_latency(stage, ts - last_time)
            attrs = {"decode_ct": self.decode_ct}
            self.trace_slice(stage, last_time, ts, attrs)
            self.decode_ct += 1

    def set_last_scheduled_time(self, forward_mode: ForwardMode, ts=None, attrs=None):
        if ts is None:
            ts = time.perf_counter()

        if self.trace_ctx.tracing_enable:
            if (
                self.disagg_mode == DisaggregationMode.NULL
                and forward_mode.is_decode()
                and self.last_decode_scheduled_time == 0.0
                and self.last_prefill_finished_time > 0
            ):
                self.trace_slice(
                    RequestStage.DECODE_WAITING, self.last_prefill_finished_time, ts
                )
                self.trace_ctx.trace_slice_start(
                    RequestStage.DECODE_FORWARD.stage_name,
                    RequestStage.DECODE_FORWARD.level,
                    convert_time_to_realtime_ns(ts),
                )
                self.last_decode_finish_time = ts

            self.trace_ctx.trace_event(
                "schedule", 3, convert_time_to_realtime_ns(ts), attrs
            )

        if forward_mode.is_decode():
            self.last_decode_scheduled_time = ts

    def set_completion_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.completion_time = ts

        self.trace_ctx.abort()

    def set_quick_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.set_completion_time(ts)
        self.forward_entry_time = ts

    def set_prefill_bootstrap_queue_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.prefill_bootstrap_queue_entry_time = ts

        stage = RequestStage.PREFILL_PREPARE
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)
        self.trace_slice(stage, self.scheduler_recv_time, ts)

    def set_prefill_transfer_queue_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.prefill_transfer_queue_entry_time = ts

    def set_prefill_kv_transfer_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.prefill_kv_transfer_finish_time = ts

        stage = RequestStage.PREFILL_TRANSFER_KV_CACHE
        self.observe_per_stage_req_latency(
            stage, ts - self.prefill_transfer_queue_entry_time
        )
        self.trace_slice(stage, self.prefill_transfer_queue_entry_time, ts)

    def set_decode_prealloc_queue_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.decode_prealloc_queue_entry_time = ts

        stage = RequestStage.DECODE_PREPARE
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)
        self.trace_slice(stage, self.scheduler_recv_time, ts)

    def set_decode_transfer_queue_entry_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.decode_transfer_queue_entry_time = ts

        stage = RequestStage.DECODE_BOOTSTRAP
        self.observe_per_stage_req_latency(
            stage, ts - self.decode_prealloc_queue_entry_time
        )
        self.trace_slice(stage, self.decode_prealloc_queue_entry_time, ts)

    def set_decode_prebuilt_finish_time(self, ts=None):
        if ts is None:
            ts = time.perf_counter()
        self.decode_prebuilt_finish_time = ts

        stage = RequestStage.DECODE_FAKE_OUTPUT
        self.observe_per_stage_req_latency(stage, ts - self.last_forward_entry_time)
        self.trace_slice(stage, self.last_forward_entry_time, ts)

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def get_prefill_waiting_latency(self) -> Optional[float]:
        if self.prefill_run_batch_start_time > 0.0:
            return self.prefill_run_batch_start_time - self.forward_entry_time
        return None

    def get_prefill_launch_latency(self) -> Optional[float]:
        if (
            self.prefill_run_batch_start_time > 0.0
            and self.prefill_run_batch_end_time > 0.0
        ):
            return self.prefill_run_batch_end_time - self.prefill_run_batch_start_time
        return None

    def convert_to_duration(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time:.3f}"
        elif self.disagg_mode == DisaggregationMode.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        bootstrap_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return (
                f"bootstrap_queue_duration({self.format_duration(bootstrap_duration)}) "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start={self.prefill_bootstrap_queue_entry_time:.3f}, "
                f"transfer_speed={self.transfer_speed_gb_s:.2f}GB/s, "
                f"transfer_total={self.transfer_total_mb:.2f}MB, "
                f"#retries={self.prefill_retry_count}"
            )
        elif self.disagg_mode == DisaggregationMode.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )
            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        prealloc_duration >= 0
                        and transfer_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0. {self=}"

            return (
                f"prealloc_queue_duration({self.format_duration(prealloc_duration)}) "
                f"transfer_duration={self.format_duration(transfer_duration)}; "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start={self.decode_prealloc_queue_entry_time:.3f}"
            )
        else:
            return "Unknown Time Stats"

    def convert_to_output_meta_info(self):
        meta_data = {}
        if self.forward_entry_time > 0.0:
            meta_data["forward_entry_time"] = convert_time_to_realtime(
                self.forward_entry_time
            )
        if self.prefill_finished_time > 0.0:
            meta_data["prefill_finished_time"] = convert_time_to_realtime(
                self.prefill_finished_time
            )
        meta_data.update(
            {
                "queue_time": self.get_queueing_time(),
                "prefill_waiting_latency": self.get_prefill_waiting_latency(),
                "prefill_launch_latency": self.get_prefill_launch_latency(),
            }
        )
        return meta_data

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"


def set_schedule_time_batch(batch: ScheduleBatch):
    # only for tracing
    if not get_global_tracing_enabled():
        return

    ts = time.perf_counter()
    bid = uuid.uuid4().hex[:8]
    _attrs = {"bid": bid, "batch_size": len(batch.reqs)}
    if batch.forward_mode.is_decode():
        _attrs["forward_mode"] = "decode"
    elif batch.forward_mode.is_prefill():
        _attrs["forward_mode"] = "prefill"
    elif batch.forward_mode.is_prebuilt():
        _attrs["forward_mode"] = "prebuilt"

    for req in batch.reqs:
        req.time_stats.set_last_scheduled_time(batch.forward_mode, ts, _attrs)


def set_time_batch(reqs: List[Any], set_func: str):
    if reqs is None or len(reqs) == 0:
        return

    ts = time.perf_counter()
    for req in reqs:
        method = getattr(req.time_stats, set_func)
        method(ts)

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
    from sglang.srt.disaggregation.base.conn import KVTransferMetric
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


real_time = time.time
monotonic_time = time.perf_counter


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
    """Configuration for a request pipeline stage.

    Attributes:
        stage_name: Name used for metrics labels and trace span names.
        level: Trace hierarchy depth.
            1 = leaf stages (atomic operations, e.g. TOKENIZE, PREFILL_FORWARD),
            2 = parent/dispatch stages (e.g. API_SERVER_DISPATCH, REQUEST_PROCESS),
            3 = composite/nested stages (e.g. DECODE_LOOP, PREFILL_CHUNKED_FORWARD).
        metrics_is_observed: Whether to call metrics_collector.observe_per_stage_req_latency.
    """

    stage_name: str
    level: int = 0
    metrics_is_observed: bool = False


class RequestStage:
    # Tokenizer/gRPC Server
    TOKENIZE = RequestStageConfig(
        "tokenize",
        level=1,
    )
    API_SERVER_DISPATCH = RequestStageConfig(
        "api_server_dispatch",
        level=2,
    )

    # DP controller
    DPC_DISPATCH = RequestStageConfig(
        "dpc_dispatch",
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

    # speculative decode
    SPEC_DRAFT = RequestStageConfig(
        "spec_draft",
        level=2,
    )

    SPEC_VERIFY = RequestStageConfig(
        "spec_verify",
        level=2,
    )

    SPEC_DRAFT_EXTEND = RequestStageConfig(
        "spec_draft_extend",
        level=3,
    )

    # CPU-side run batch
    RUN_BATCH_CPU = RequestStageConfig(
        "run_batch_cpu",
        level=4,
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

        if new_obj.trace_ctx.tracing_enable:
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
        ts = ts or time.perf_counter()
        self.created_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_req_start(convert_time_to_realtime_ns(ts))

    def set_finished_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.finished_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_req_finish(convert_time_to_realtime_ns(ts))

    def set_first_token_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.first_token_time = ts
        self.last_time = ts

    def set_last_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.last_time = ts

    def set_tokenize_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.tokenize_finish_time = ts

        stage = RequestStage.TOKENIZE
        self.trace_slice(stage, self.created_time, ts)

    def set_api_server_dispatch_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.api_server_dispatch_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_slice_start(
                RequestStage.API_SERVER_DISPATCH.stage_name,
                RequestStage.API_SERVER_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
            )

    def set_api_server_dispatch_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.api_server_dispatch_finish_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_slice_end(
                RequestStage.API_SERVER_DISPATCH.stage_name,
                RequestStage.API_SERVER_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
                thread_finish_flag=True,
            )

    def set_response_sent_to_client_time(self, ts=None):
        ts = ts or time.perf_counter()
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

        decode_latency = self.get_decode_latency()
        if decode_latency > 0.0 and completion_tokens > 1:
            meta_info["decode_throughput"] = (completion_tokens - 1) / decode_latency
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
    dpc_dispatch_time: float = 0.0
    dpc_dispatch_finish_time: float = 0.0

    def __getstate__(self) -> object:
        state = {}
        # send to Scheduler
        # If necessary, can propagate the timestamp here, for example:
        # state = {
        #     "created_time": self.created_time,
        #     "api_server_dispatch_time": self.api_server_dispatch_time,
        #     "dpc_dispatch_time": self.dpc_dispatch_time,
        # }
        state.update(super().__getstate__())
        return state

    def set_dp_dispatch_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.dpc_dispatch_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_slice_start(
                RequestStage.DPC_DISPATCH.stage_name,
                RequestStage.DPC_DISPATCH.level,
                convert_time_to_realtime_ns(ts),
            )

    def set_dp_dispatch_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.dpc_dispatch_finish_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_slice_end(
                RequestStage.DPC_DISPATCH.stage_name,
                RequestStage.DPC_DISPATCH.level,
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
    dpc_dispatch_time: float = 0.0

    # common, get by time.perf_counter()
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
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

    # bootstrap sub-phase tracking (PD disagg)
    bootstrap_done_time: float = 0.0

    # only for request tracing
    scheduler_recv_time: float = 0.0
    last_chunked_prefill_finish_time: float = 0.0
    last_decode_finish_time: float = 0.0
    decode_ct: int = 0
    last_decode_scheduled_time: float = 0.0
    last_forward_entry_time: float = 0.0
    last_prefill_finished_time: float = 0.0
    run_batch_cpu_start_time: float = 0.0

    # speculative decoding
    spec_draft_start_time: float = 0.0
    spec_verify_start_time: float = 0.0
    spec_draft_extend_start_time: float = 0.0

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
            "prefill_finished_time": self.prefill_finished_time,
            "diff_realtime_monotonic": global_diff_realtime_monotonic,
        }
        return state

    def set_scheduler_recv_time(self, ts=None):
        calibrate_time_diff()
        ts = ts or time.perf_counter()
        self.scheduler_recv_time = ts

    def set_spec_draft_start_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.spec_draft_start_time = ts

    def set_spec_draft_end_time(self, ts=None):
        ts = ts or time.perf_counter()

        if self.trace_ctx.tracing_enable:
            stage = RequestStage.SPEC_DRAFT
            self.trace_slice(stage, self.spec_draft_start_time, ts)

    def set_spec_verify_start_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.spec_verify_start_time = ts

    def set_spec_verify_end_time(
        self,
        ts=None,
        num_correct_drafts: int = 0,
        # FIXME: backward-compat alias, remove in next release.
        accepted_tokens: Optional[int] = None,
    ):
        if accepted_tokens is not None:
            num_correct_drafts = accepted_tokens
        ts = ts or time.perf_counter()

        if self.trace_ctx.tracing_enable:
            stage = RequestStage.SPEC_VERIFY
            self.trace_slice(
                stage,
                self.spec_verify_start_time,
                ts,
                {
                    "num_correct_drafts": num_correct_drafts,
                    # FIXME: backward-compat alias, remove in next release.
                    "accepted_tokens": num_correct_drafts,
                },
            )

    def set_spec_draft_extend_start_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.spec_draft_extend_start_time = ts

    def set_spec_draft_extend_end_time(self, ts=None):
        ts = ts or time.perf_counter()

        if self.trace_ctx.tracing_enable:
            stage = RequestStage.SPEC_DRAFT_EXTEND
            self.trace_slice(stage, self.spec_draft_extend_start_time, ts)

    def set_run_batch_cpu_start_time(self, ts=None, attrs=None):
        ts = ts or time.perf_counter()
        self.run_batch_cpu_start_time = ts

    def set_run_batch_cpu_end_time(self, ts=None, attrs=None):
        ts = ts or time.perf_counter()
        if self.run_batch_cpu_start_time > 0.0:
            self.trace_slice(
                RequestStage.RUN_BATCH_CPU, self.run_batch_cpu_start_time, ts, attrs
            )
            self.run_batch_cpu_start_time = 0.0

    def set_retract_time(self, ts=None):
        ts = ts or time.perf_counter()
        # retract
        self.last_forward_entry_time = 0.0
        self.last_prefill_finished_time = 0.0
        self.last_chunked_prefill_finish_time = 0.0
        self.last_decode_finish_time = 0.0
        self.last_decode_scheduled_time = 0.0

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.trace_event("retract", 1, convert_time_to_realtime_ns(ts))

    def set_wait_queue_entry_time(self, ts=None):
        ts = ts or time.perf_counter()
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
        ts = ts or time.perf_counter()
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

    def set_last_chunked_prefill_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        last_time = self.last_chunked_prefill_finish_time
        self.last_chunked_prefill_finish_time = ts

        if last_time == 0.0:
            last_time = self.last_forward_entry_time

        stage = RequestStage.PREFILL_CHUNKED_FORWARD
        self.observe_per_stage_req_latency(stage, ts - last_time)
        self.trace_slice(stage, last_time, ts)

    def set_prefill_finished_time(self, ts=None):
        ts = ts or time.perf_counter()
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
        ts = ts or time.perf_counter()
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
        ts = ts or time.perf_counter()

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
        ts = ts or time.perf_counter()
        self.completion_time = ts

        if self.trace_ctx.tracing_enable:
            self.trace_ctx.abort()

    def compute_and_observe_kv_transfer_metrics(
        self,
        transfer_metric: KVTransferMetric,
    ) -> Optional[dict]:
        """Compute KV transfer metrics and observe them via the metrics collector.

        Returns a dict with latency_ms, total_mb, speed_gb_s if computable, else None.
        """
        result = {}
        if transfer_metric.transfer_total_bytes is None:
            return result if result else None

        # Transfer latency, size, and speed
        if transfer_metric.transfer_latency_s is not None:
            transfer_latency_s = transfer_metric.transfer_latency_s
        else:
            if self.prefill_transfer_queue_entry_time <= 0 or self.completion_time <= 0:
                return result if result else None
            # Note: This only capture the last chunk time
            transfer_latency_s = (
                self.completion_time - self.prefill_transfer_queue_entry_time
            )

        if transfer_latency_s > 0:
            latency_ms = transfer_latency_s * 1000

            total_bytes = transfer_metric.transfer_total_bytes
            total_mb = total_bytes / (1024 * 1024)
            self.transfer_total_mb = total_mb

            speed_gb_s = 0.0
            if transfer_latency_s > 0:
                speed_gb_s = (total_mb / 1024) / transfer_latency_s
                self.transfer_speed_gb_s = speed_gb_s

            result["latency_ms"] = latency_ms
            result["total_mb"] = total_mb
            result["speed_gb_s"] = speed_gb_s

            if self.enable_metrics:
                self.metrics_collector.observe_kv_transfer_metrics(
                    latency_ms=latency_ms,
                    total_mb=total_mb,
                    speed_gb_s=speed_gb_s,
                )

        # Bootstrap and alloc durations
        if (
            self.prefill_bootstrap_queue_entry_time > 0
            and self.bootstrap_done_time > 0
            and self.wait_queue_entry_time > 0
        ):
            bootstrap_ms = (
                self.bootstrap_done_time - self.prefill_bootstrap_queue_entry_time
            ) * 1000
            alloc_ms = (self.wait_queue_entry_time - self.bootstrap_done_time) * 1000

            result["bootstrap_ms"] = bootstrap_ms
            result["alloc_ms"] = alloc_ms

            if self.enable_metrics:
                self.metrics_collector.observe_kv_transfer_bootstrap(
                    bootstrap_ms=bootstrap_ms,
                    alloc_ms=alloc_ms,
                )

        return result if result else None

    def set_quick_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.set_completion_time(ts)
        self.forward_entry_time = ts

    def set_prefill_bootstrap_queue_entry_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.prefill_bootstrap_queue_entry_time = ts

        stage = RequestStage.PREFILL_PREPARE
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)
        self.trace_slice(stage, self.scheduler_recv_time, ts)

    def set_prefill_transfer_queue_entry_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.prefill_transfer_queue_entry_time = ts

    def set_prefill_kv_transfer_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.prefill_kv_transfer_finish_time = ts

        stage = RequestStage.PREFILL_TRANSFER_KV_CACHE
        self.observe_per_stage_req_latency(
            stage, ts - self.prefill_transfer_queue_entry_time
        )
        self.trace_slice(stage, self.prefill_transfer_queue_entry_time, ts)

    def set_decode_prealloc_queue_entry_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.decode_prealloc_queue_entry_time = ts

        stage = RequestStage.DECODE_PREPARE
        self.observe_per_stage_req_latency(stage, ts - self.scheduler_recv_time)
        self.trace_slice(stage, self.scheduler_recv_time, ts)

    def set_decode_transfer_queue_entry_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.decode_transfer_queue_entry_time = ts

        stage = RequestStage.DECODE_BOOTSTRAP
        self.observe_per_stage_req_latency(
            stage, ts - self.decode_prealloc_queue_entry_time
        )
        self.trace_slice(stage, self.decode_prealloc_queue_entry_time, ts)

        if self.enable_metrics and self.bootstrap_done_time > 0:
            bootstrap_ms = (
                self.bootstrap_done_time - self.decode_prealloc_queue_entry_time
            ) * 1000
            alloc_ms = (ts - self.bootstrap_done_time) * 1000
            self.metrics_collector.observe_kv_transfer_bootstrap(
                bootstrap_ms=bootstrap_ms,
                alloc_ms=alloc_ms,
            )

    def set_bootstrap_done_time(self, ts=None):
        ts = ts or time.perf_counter()
        if self.bootstrap_done_time == 0.0:
            self.bootstrap_done_time = ts

    def set_decode_prebuilt_finish_time(self, ts=None):
        ts = ts or time.perf_counter()
        self.decode_prebuilt_finish_time = ts

        stage = RequestStage.DECODE_FAKE_OUTPUT
        self.observe_per_stage_req_latency(stage, ts - self.last_forward_entry_time)
        self.trace_slice(stage, self.last_forward_entry_time, ts)

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def convert_to_duration(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL:
            queue_duration = self.duration_between(
                self.wait_queue_entry_time, self.forward_entry_time
            )
            forward_duration = self.duration_between(
                self.forward_entry_time, self.completion_time
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, entry_time={self.format_wallclock(self.wait_queue_entry_time)}"
        elif self.disagg_mode == DisaggregationMode.PREFILL:
            bootstrap_queue_duration = self.duration_between(
                self.prefill_bootstrap_queue_entry_time, self.wait_queue_entry_time
            )
            queue_duration = self.duration_between(
                self.wait_queue_entry_time, self.forward_entry_time
            )
            forward_duration = self.duration_between(
                self.forward_entry_time, self.completion_time
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        bootstrap_queue_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"bootstrap_queue_duration={bootstrap_queue_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            # Break down bootstrap_queue_duration into sub-phases
            if self.bootstrap_done_time > 0:
                bootstrap_duration = self.duration_between(
                    self.prefill_bootstrap_queue_entry_time, self.bootstrap_done_time
                )
                alloc_wait_duration = self.duration_between(
                    self.bootstrap_done_time, self.wait_queue_entry_time
                )
                if SGLANG_TEST_REQUEST_TIME_STATS:
                    assert (
                        bootstrap_duration >= 0 and alloc_wait_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or alloc_wait_duration={alloc_wait_duration} < 0"
                bootstrap_fields = (
                    f"bootstrap_duration={self.format_duration(bootstrap_duration)}, "
                    f"alloc_wait_duration={self.format_duration(alloc_wait_duration)}, "
                )
            else:
                bootstrap_fields = f"bootstrap_queue_duration={self.format_duration(bootstrap_queue_duration)}, "

            return (
                f"{bootstrap_fields}"
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"entry_time={self.format_wallclock(self.prefill_bootstrap_queue_entry_time)}, "
                f"transfer_speed={self.transfer_speed_gb_s:.2f} GB/s, "
                f"transfer_total={self.transfer_total_mb:.2f} MB, "
                f"#retries={self.prefill_retry_count}"
            )
        elif self.disagg_mode == DisaggregationMode.DECODE:
            prealloc_duration = self.duration_between(
                self.decode_prealloc_queue_entry_time,
                self.decode_transfer_queue_entry_time,
            )
            transfer_duration = self.duration_between(
                self.decode_transfer_queue_entry_time,
                self.wait_queue_entry_time,
            )
            queue_duration = self.duration_between(
                self.wait_queue_entry_time,
                self.forward_entry_time,
            )
            forward_duration = self.duration_between(
                self.forward_entry_time,
                self.completion_time,
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        prealloc_duration >= 0
                        and transfer_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0. {self=}"

            # Break down prealloc_duration into sub-phases
            if self.bootstrap_done_time > 0:
                bootstrap_duration = self.duration_between(
                    self.decode_prealloc_queue_entry_time, self.bootstrap_done_time
                )
                alloc_wait_duration = self.duration_between(
                    self.bootstrap_done_time, self.decode_transfer_queue_entry_time
                )
                if SGLANG_TEST_REQUEST_TIME_STATS:
                    assert (
                        bootstrap_duration >= 0 and alloc_wait_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or alloc_wait_duration={alloc_wait_duration} < 0"
                prealloc_fields = (
                    f"bootstrap_duration={self.format_duration(bootstrap_duration)}, "
                    f"alloc_wait_duration={self.format_duration(alloc_wait_duration)}, "
                )
            else:
                prealloc_fields = f"prealloc_queue_duration={self.format_duration(prealloc_duration)}, "

            return (
                f"{prealloc_fields}"
                f"transfer_duration={self.format_duration(transfer_duration)}, "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"entry_time={self.format_wallclock(self.decode_prealloc_queue_entry_time)}"
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
            }
        )
        return meta_data

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def duration_between(self, start: float, end: float) -> float:
        if start <= 0 or end <= 0:
            return 0.0
        return end - start

    @staticmethod
    def format_wallclock(perf_counter_time: float) -> str:
        return f"{convert_time_to_realtime(perf_counter_time):.3f}"


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


def set_time_batch(
    reqs: List[Any],
    set_func: str,
    trace_only: bool = False,
    attrs: Optional[Dict[str, Any]] = None,
):
    if reqs is None or len(reqs) == 0:
        return
    if trace_only and not get_global_tracing_enabled():
        return

    ts = time.perf_counter()
    for req in reqs:
        method = getattr(req.time_stats, set_func)
        if attrs is None:
            method(ts)
        else:
            method(ts, attrs)

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
"""definition for requests stage timing recorder"""
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sglang.srt.tracing.trace import (
    SglangTraceReqContext,
    get_cur_time_ns,
    get_opentelemetry_initialized,
)


@dataclass
class RequestStageConfig:
    stage_name: str
    level: int = 0
    # whether to call metrics_collector.observe_per_stage_req_latency
    metrics_is_observed: bool = False


class RequestStage:
    # Tokenizer
    TOKENIZE = RequestStageConfig(
        "tokenize",
        level=1,
    )
    TOKENIZER_DISPATCH = RequestStageConfig(
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
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_WAITING = RequestStageConfig(
        "prefill_waiting",
        level=1,
        # equal to "observe_queue_time"
        metrics_is_observed=False,
    )
    DECODE_LOOP = RequestStageConfig(
        "decode_loop",
        level=2,
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
        level=1,
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


class SglangStageContext(SglangTraceReqContext):
    def __init__(
        self,
        rid,
        bootstrap_room,
        module_name,
        server_args,
        metrics_collector=None,
        propagation_context: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        ts: Optional[int] = None,
        external_trace_header: Optional[Dict[str, Any]] = None,
    ):
        self.enable_metrics = getattr(server_args, "enable_metrics", False)
        self.metrics_collector = metrics_collector
        if not metrics_collector:
            self.enable_metrics = False
        self.last_ts_stack = []

        opentelemetry_initialized = get_opentelemetry_initialized()
        self.module_name = module_name
        trace_level = getattr(server_args, "trace_level", 0)
        tracing_enable = (
            True
            if getattr(server_args, "trace_module", None) == module_name
            and trace_level > 0
            and opentelemetry_initialized
            else False
        )
        self.time_record_enable = tracing_enable or self.enable_metrics

        if not tracing_enable:
            super().__init__(str(rid), tracing_enable=False)
            return

        self.disagg_mode = getattr(server_args, "disaggregation_mode", "null")
        if not role:
            role = self.disagg_mode
        super().__init__(
            rid=str(rid),
            bootstrap_room=bootstrap_room,
            role=role,
            tracing_enable=tracing_enable,
            trace_level=trace_level,
        )

        if isinstance(propagation_context, dict):
            super().trace_set_proc_propagate_context(propagation_context)
        else:
            super().trace_req_start(ts, external_trace_header=external_trace_header)

    def metric_trace_slice_start(
        self,
        stage: RequestStageConfig,
        ts: Optional[int] = None,
    ):
        if self.enable_metrics:
            ts = ts or get_cur_time_ns()
            self.last_ts_stack.append(ts)

        super().trace_slice_start(
            stage.stage_name,
            ts=ts,
            anonymous=(stage == RequestStage.ANONYMOUS),
            level=stage.level,
        )

    def metric_trace_slice_end(
        self,
        stage: RequestStageConfig,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        auto_next_anon: bool = False,
        thread_finish_flag: bool = False,
    ):
        if self.enable_metrics and len(self.last_ts_stack) > 0:
            ts = ts or get_cur_time_ns()
            last_ts = self.last_ts_stack.pop()
            lat = (ts - last_ts) / 1e9

            if stage.metrics_is_observed:
                try:
                    self.metrics_collector.observe_per_stage_req_latency(
                        stage.stage_name,
                        lat,
                    )
                except AttributeError:
                    pass

        self.trace_slice_end(
            stage.stage_name,
            ts=ts,
            attrs=attrs,
            auto_next_anon=auto_next_anon,
            thread_finish_flag=thread_finish_flag,
            level=stage.level,
        )

    metric_trace_slice = metric_trace_slice_end


class NoOpTimeRecorder:
    __slots__ = ()

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        pass

    def __delattr__(self, name):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return "<NullObject>"


def metric_trace_slice_batch(
    stage: RequestStageConfig,
    reqs: List,
):
    if not reqs or not reqs[0].stage_context.time_record_enable:
        return

    for req in reqs:
        req.stage_context.metric_trace_slice(
            stage,
            auto_next_anon=not req.finished(),
            thread_finish_flag=req.finished(),
        )


def trace_event_batch(
    name: str,
    reqs: List,
    ts: Optional[int] = None,
    attrs: Dict[str, Any] = {},
):
    if not reqs or not reqs[0].stage_context.tracing_enable:
        return

    bid = uuid.uuid4().hex[:8]
    _attrs = {"bid": bid, "batch_size": len(reqs)}
    _attrs.update(attrs)

    for req in reqs:
        req.stage_context.trace_event(name, ts=ts, attrs=_attrs)


"""
Used when the stage_context cannot be integrated into the request object.

format:
    {
        thread_id: {
            "rid": SglangStageContext
        }
    }
"""
global_stage_context_table: Dict[int, Dict[str, SglangStageContext]] = {}


def global_init_stage_context(
    rid,
    bootstrap_room,
    module_name,
    server_args,
    metrics_collector=None,
    propagation_context: Optional[Dict[str, Any]] = None,
    time_stat_cls=None,
    role: Optional[str] = None,
):
    pid = threading.get_native_id()
    rid = str(rid)
    stage_context = SglangStageContext(
        rid=rid,
        bootstrap_room=bootstrap_room,
        module_name=module_name,
        server_args=server_args,
        metrics_collector=metrics_collector,
        propagation_context=propagation_context,
        time_stat_cls=time_stat_cls,
        role=role,
    )

    global_stage_context_table.setdefault(pid, {})[rid] = stage_context

    return stage_context


def global_get_stage_context(rid) -> Union[SglangStageContext, NoOpTimeRecorder]:
    pid = threading.get_native_id()
    rid = str(rid)
    if pid in global_stage_context_table:
        if rid in global_stage_context_table[pid]:
            return global_stage_context_table[pid][rid]
    return NoOpTimeRecorder()


def global_set_stage_context(stage_context):
    pid = threading.get_native_id()
    rid = stage_context.rid
    global_stage_context_table.setdefault(pid, {})[rid] = stage_context


def global_del_stage_context(rid):
    pid = threading.get_native_id()
    rid = str(rid)
    if pid in global_stage_context_table:
        if rid in global_stage_context_table[pid]:
            del global_stage_context_table[pid][rid]


def trace_inject_propagate_context(obj):
    if hasattr(obj, "stage_context"):
        old_stage_context = obj.stage_context
        obj.stage_context = obj.stage_context.trace_get_proc_propagate_context()
        return old_stage_context
    else:
        return None


def trace_restore_stage_context(obj, old_stage_context):
    if hasattr(obj, "stage_context"):
        obj.stage_context = old_stage_context

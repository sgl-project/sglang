"""Config fields of the ``observability`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``observability`` bag, which is what ``get_observability()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.utils.common import json_list_type


@dataclasses.dataclass
class Observability:
    """Namespace ``observability``."""

    _NS_PATH = "observability"

    # -------------------------------------------------------------------------
    # Logging, metrics, and tracing
    # -------------------------------------------------------------------------
    log_level: A[str, "The logging level of all loggers."] = "info"
    log_level_http: A[
        Optional[str],
        "The logging level of HTTP server. If not set, reuse --log-level by default.",
    ] = None
    log_requests: A[
        bool,
        "Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
    ] = False
    log_requests_level: A[
        int,
        Arg(
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        ),
    ] = 2
    log_requests_format: A[
        str,
        Arg(
            help="Format for request logging: 'text' (human-readable) or 'json' (structured)",
            choices=["text", "json"],
        ),
    ] = "text"
    log_requests_target: A[
        Optional[List[str]],
        "Target(s) for request logging: 'stdout' and/or directory path(s) for file output. Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. ",
    ] = None
    uvicorn_access_log_exclude_prefixes: A[
        List[str],
        Arg(
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. Defaults to empty (disabled). Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
            nargs="*",
        ),
    ] = dataclasses.field(default_factory=list)
    crash_dump_folder: A[
        Optional[str],
        "Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
    ] = None
    show_time_cost: A[bool, "Show time cost of custom marks."] = False
    enable_metrics: A[bool, "Enable log prometheus metrics."] = False
    smg_http_sidecar_port: A[
        Optional[int],
        Arg(
            help="Port for the HTTP sidecar server in legacy SMG gRPC mode (--smg-grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode.",
            aliases=["--grpc-http-sidecar-port"],
        ),
    ] = None
    enable_mfu_metrics: A[
        bool,
        "Enable estimated MFU-related prometheus metrics.",
    ] = False
    enable_metrics_for_all_schedulers: A[
        bool,
        "Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0.",
    ] = False
    load_snapshot_publish_interval: A[
        int,
        "Publish load snapshot to shared memory every N decode iterations. Prefill and idle always publish immediately.",
    ] = 15
    tokenizer_metrics_custom_labels_header: A[
        str, "Specify the HTTP header for passing custom labels for tokenizer metrics."
    ] = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: A[
        Optional[List[str]],
        "The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
    ] = None
    extra_metric_labels: A[
        Optional[Dict[str, str]],
        Arg(
            help='The custom labels for metrics. e.g. \'{"label1": "value1", "label2": "value2"}\'',
            type_parser=json.loads,
        ),
    ] = None
    bucket_time_to_first_token: A[
        Optional[List[float]],
        "The buckets of time to first token, specified as a list of floats.",
    ] = None
    bucket_inter_token_latency: A[
        Optional[List[float]],
        "The buckets of inter-token latency, specified as a list of floats.",
    ] = None
    bucket_e2e_request_latency: A[
        Optional[List[float]],
        "The buckets of end-to-end request latency, specified as a list of floats.",
    ] = None
    prompt_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule of prompt tokens. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
    ] = None
    generation_tokens_buckets: A[
        Optional[List[str]],
        "The buckets rule for generation tokens histogram. "
        "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
        "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
        "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
        "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500').",
    ] = None
    gc_warning_threshold_secs: A[
        float,
        "The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
    ] = 0.0
    decode_log_interval: A[
        int,
        "The log and metrics reporting interval (in decode iterations) for decode batches.",
    ] = 40
    enable_request_time_stats_logging: A[
        bool,
        "Enable per request time stats logging",
    ] = False
    kv_events_config: A[
        Optional[str],
        "Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used. Runtime-load publishing for load-aware routers is a separate opt-in; see --load-publish-endpoint.",
    ] = None
    load_publish_endpoint: A[
        Optional[str],
        "Opt in to the runtime-load PUB socket that load-aware routers subscribe to. Off by default (unset or 'off'). Use 'auto' to reserve the dp_size ports packed after the --kv-events-config range, or a wildcard-host TCP address (e.g. tcp://*:6000) to place it explicitly; rank r binds port+r and /server_info advertises the base under the kv_events block. Requires --kv-events-config to describe a publisher (routers discover the base through /server_info); startup fails if this is set without one, is not bindable, or overlaps the KV range. Note: 'auto' reserves 2*dp_size ports from the KV base — space co-hosted engines accordingly. The router-facing update cadence follows --load-snapshot-publish-interval (shared to avoid double-collecting the snapshot), so a large value there also staleness-caps this feed.",
    ] = None
    enable_forward_pass_metrics: A[
        bool,
        "Enable per-iteration forward pass metrics via ZMQ IPC. External consumers (e.g. Dynamo planner) subscribe to the IPC endpoint exposed in server_args.forward_pass_metrics_ipc_name.",
    ] = False
    forward_pass_metrics_worker_id: A[
        str,
        Arg(help=argparse.SUPPRESS),
    ] = ""
    forward_pass_metrics_ipc_name: A[
        Optional[str],
        Arg(help=argparse.SUPPRESS),
    ] = None
    enable_trace: A[bool, "Enable opentelemetry trace"] = False
    trace_modules: A[
        str,
        "Select the components to trace. Available options are 'request' and 'mooncake'. Format: <module1 name>,<module2 name>,...",
    ] = "request"
    otlp_traces_endpoint: A[
        str,
        "Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
    ] = "localhost:4317"
    # RequestMetricsExporter configuration
    export_metrics_to_file: A[
        bool,
        "Export performance metrics for each request to local file (e.g. for forwarding to external systems).",
    ] = False
    export_metrics_to_file_dir: A[
        Optional[str],
        "Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).",
    ] = None
    # Class-level DI for the five *MetricsCollector classes. Maps collector role
    # (one of: "scheduler", "tokenizer", "storage", "radix_cache", "expert_dispatch")
    # to a subclass of the matching base collector. The five instantiation sites
    # read from this map and fall back to the base class. Class-object only (no
    # CLI surface) since this exists for embedded use cases that pass a Python
    # class directly. Default None preserves existing behavior.
    stat_loggers: Optional[Dict[str, type]] = None

    # -------------------------------------------------------------------------
    # KV canary
    # -------------------------------------------------------------------------
    kv_canary: A[
        str,
        Arg(
            help="KV cache canary mode. 'none' disables the canary (default). 'log' prints them while the server keeps running (production-safe). 'raise' fails the server on the first detected mismatch (CI lane).",
            choices=["none", "log", "raise"],
        ),
    ] = "none"
    kv_canary_real_data: str = "none"
    kv_canary_sweep_interval: A[
        int,
        "Every N forward steps, run a full-pool sweep.",
    ] = 0

    # -------------------------------------------------------------------------
    # Debug tensor dumps
    # -------------------------------------------------------------------------
    debug_tensor_dump_output_folder: A[
        Optional[str],
        "The output folder for dumping tensors. In Eagle mode, tensor outputs from draft and target models are stored in separate subdirectories ('draft' and 'target').",
    ] = None
    # None means dump all layers.
    debug_tensor_dump_layers: A[
        Optional[List[int]], "The layer ids to dump. Dump all layers if not specified."
    ] = None
    # TODO(guoyuhong): clean the old dumper code.
    debug_tensor_dump_input_file: A[
        Optional[str],
        "The input filename for dumping tensors",
    ] = None

    # -------------------------------------------------------------------------
    # Custom hooks, probe, and plugins
    # -------------------------------------------------------------------------
    forward_hooks: A[
        Optional[List[dict[str, Any]]],
        Arg(
            help="JSON-formatted forward hook specifications to attach to the model.",
            type_parser=json_list_type,
        ),
    ] = None
    msprobe_dump_config: A[
        Optional[str],
        "The path of the JSON configuration file for msProbe. If specified, enables msProbe dump.",
    ] = None

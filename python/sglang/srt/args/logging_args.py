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
"""CLI argument definitions for logging."""

import argparse
import json


class LoggingArgs:
    """CLI argument definitions for logging and RequestMetricsExporter."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=ServerArgs.log_requests_level,
            help="0: Log metadata (no sampling parameters). 1: Log metadata and sampling parameters. 2: Log metadata, sampling parameters and partial input/output. 3: Log every input/output.",
            choices=[0, 1, 2, 3],
        )
        parser.add_argument(
            "--log-requests-format",
            type=str,
            default=ServerArgs.log_requests_format,
            choices=["text", "json"],
            help="Format for request logging: 'text' (human-readable) or 'json' (structured)",
        )
        parser.add_argument(
            "--log-requests-target",
            type=str,
            nargs="+",
            default=ServerArgs.log_requests_target,
            help="Target(s) for request logging: 'stdout' and/or directory path(s) for file output. "
            "Can specify multiple targets, e.g., '--log-requests-target stdout /my/path'. ",
        )
        parser.add_argument(
            "--uvicorn-access-log-exclude-prefixes",
            type=str,
            nargs="*",
            default=list(DEFAULT_UVICORN_ACCESS_LOG_EXCLUDE_PREFIXES),
            help="Exclude uvicorn access logs whose request path starts with any of these prefixes. "
            "Defaults to empty (disabled). "
            "Example: --uvicorn-access-log-exclude-prefixes /metrics /health",
        )
        parser.add_argument(
            "--crash-dump-folder",
            type=str,
            default=ServerArgs.crash_dump_folder,
            help="Folder path to dump requests from the last 5 min before a crash (if any). If not specified, crash dumping is disabled.",
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log prometheus metrics.",
        )
        parser.add_argument(
            "--enable-mfu-metrics",
            action="store_true",
            help="Enable estimated MFU-related prometheus metrics.",
        )
        parser.add_argument(
            "--enable-metrics-for-all-schedulers",
            action="store_true",
            help="Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) "
            "to record request metrics separately. This is especially useful when dp_attention is enabled, as "
            "otherwise all metrics appear to come from TP 0.",
        )
        parser.add_argument(
            "--tokenizer-metrics-custom-labels-header",
            type=str,
            default=ServerArgs.tokenizer_metrics_custom_labels_header,
            help="Specify the HTTP header for passing custom labels for tokenizer metrics.",
        )
        parser.add_argument(
            "--tokenizer-metrics-allowed-custom-labels",
            type=str,
            nargs="+",
            default=ServerArgs.tokenizer_metrics_allowed_custom_labels,
            help="The custom labels allowed for tokenizer metrics. The labels are specified via a dict in "
            "'--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': "
            "'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
        )
        parser.add_argument(
            "--extra-metric-labels",
            type=json.loads,
            default=ServerArgs.extra_metric_labels,
            help="The custom labels for metrics. "
            'e.g. \'{"label1": "value1", "label2": "value2"}\'',
        )
        parser.add_argument(
            "--bucket-time-to-first-token",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_time_to_first_token,
            help="The buckets of time to first token, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-inter-token-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_inter_token_latency,
            help="The buckets of inter-token latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--bucket-e2e-request-latency",
            type=float,
            nargs="+",
            default=ServerArgs.bucket_e2e_request_latency,
            help="The buckets of end-to-end request latency, specified as a list of floats.",
        )
        parser.add_argument(
            "--collect-tokens-histogram",
            action="store_true",
            default=ServerArgs.collect_tokens_histogram,
            help="Collect prompt/generation tokens histogram.",
        )
        bucket_rule = (
            "Supports 3 rule types: 'default' uses predefined buckets; 'tse <middle> <base> <count>' "
            "generates two sides exponential distributed buckets (e.g., 'tse 1000 2 8' generates buckets "
            "[984.0, 992.0, 996.0, 998.0, 1000.0, 1002.0, 1004.0, 1008.0, 1016.0]).); 'custom <value1> "
            "<value2> ...' uses custom bucket values (e.g., 'custom 10 50 100 500')."
        )
        parser.add_argument(
            "--prompt-tokens-buckets",
            type=str,
            nargs="+",
            default=ServerArgs.prompt_tokens_buckets,
            help=f"The buckets rule of prompt tokens. {bucket_rule}",
        )
        parser.add_argument(
            "--generation-tokens-buckets",
            type=str,
            nargs="+",
            default=ServerArgs.generation_tokens_buckets,
            help=f"The buckets rule for generation tokens histogram. {bucket_rule}",
        )
        parser.add_argument(
            "--gc-warning-threshold-secs",
            type=float,
            default=ServerArgs.gc_warning_threshold_secs,
            help="The threshold for long GC warning. If a GC takes longer than this, a warning will be logged. Set to 0 to disable.",
        )
        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log and metrics reporting interval (in decode iterations) for decode batches.",
        )
        parser.add_argument(
            "--enable-request-time-stats-logging",
            action="store_true",
            default=ServerArgs.enable_request_time_stats_logging,
            help="Enable per request time stats logging",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=None,
            help="Config in json format for NVIDIA dynamo KV event publishing. Publishing will be enabled if this flag is used.",
        )
        parser.add_argument(
            "--enable-trace",
            action="store_true",
            help="Enable opentelemetry trace",
        )
        parser.add_argument(
            "--otlp-traces-endpoint",
            type=str,
            default="localhost:4317",
            help="Config opentelemetry collector endpoint if --enable-trace is set. format: <ip>:<port>",
        )

        # RequestMetricsExporter configuration
        parser.add_argument(
            "--export-metrics-to-file",
            action="store_true",
            help="Export performance metrics for each request to local file (e.g. for forwarding to external systems).",
        )
        parser.add_argument(
            "--export-metrics-to-file-dir",
            type=str,
            default=ServerArgs.export_metrics_to_file_dir,
            help="Directory path for writing performance metrics files (required when --export-metrics-to-file is enabled).",
        )

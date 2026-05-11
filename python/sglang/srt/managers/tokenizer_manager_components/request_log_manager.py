from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

from sglang.srt.observability.request_metrics_exporter import (
    RequestMetricsExporterManager,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.request_logger import RequestLogger


@dataclass(slots=True, kw_only=True)
class RequestLogManager:
    """Per-request logging + periodic dump + 5-min rolling crash dump."""

    server_args: ServerArgs
    request_logger: RequestLogger
    request_metrics_exporter_manager: RequestMetricsExporterManager
    dump_requests_folder: str = ""
    dump_requests_threshold: int = 1000
    dump_requests_exclude_meta_keys: List[str] = field(
        default_factory=lambda: ["routed_experts", "hidden_states"]
    )
    crash_dump_folder: str = ""
    dump_request_list: List[Tuple] = field(default_factory=list)
    crash_dump_request_list: deque = field(default_factory=deque)
    crash_dump_performed: bool = False

    @classmethod
    def from_server_args(cls, *, server_args: ServerArgs) -> "RequestLogManager":
        request_logger = RequestLogger(
            log_requests=server_args.log_requests,
            log_requests_level=server_args.log_requests_level,
            log_requests_format=server_args.log_requests_format,
            log_requests_target=server_args.log_requests_target,
        )
        _, obj_skip_names, out_skip_names = request_logger.metadata
        request_metrics_exporter_manager = RequestMetricsExporterManager(
            server_args, obj_skip_names, out_skip_names
        )
        return cls(
            server_args=server_args,
            request_logger=request_logger,
            request_metrics_exporter_manager=request_metrics_exporter_manager,
            crash_dump_folder=server_args.crash_dump_folder,
        )

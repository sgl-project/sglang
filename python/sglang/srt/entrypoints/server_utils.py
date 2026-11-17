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
"""HTTP server helpers: Prometheus and uvicorn setup, server info, health-check servers."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
from typing import Any, Dict
from urllib.parse import urlparse

from starlette.routing import Mount

from sglang.srt.environ import envs
from sglang.srt.observability.func_timer import enable_func_timer
from sglang.srt.runtime_context import describe_kv_events_publisher, get_serving
from sglang.srt.server_args import redact_server_args_secrets
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

logger = logging.getLogger(__name__)


# Temporary directory for prometheus multiprocess mode
# Cleaned up automatically when this object is garbage collected
prometheus_multiproc_dir: tempfile.TemporaryDirectory


def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # sglang uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.debug("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.debug(f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")


def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


class UvicornAccessLogFilter(logging.Filter):
    """Filter uvicorn access logs by request path.

    Notes:
    - Uvicorn access records usually provide `request_line` like: "GET /metrics HTTP/1.1".
    - We defensively fall back to parsing `record.getMessage()` if needed.
    """

    def __init__(self, excluded_path_prefixes=None):
        super().__init__()
        excluded_path_prefixes = excluded_path_prefixes or []
        # Normalize once: drop empty prefixes, stringify, keep as tuple (fast iteration, immutable).
        self.excluded_path_prefixes = tuple(str(p) for p in excluded_path_prefixes if p)

    def filter(self, record: logging.LogRecord) -> bool:
        path = None

        request_line = getattr(record, "request_line", None)
        if request_line:
            parts = str(request_line).split()
            if len(parts) >= 2:
                path = parts[1]

        if not path:
            # Fallback for non-standard formatters/records
            try:
                msg = record.getMessage()
            except Exception:
                msg = None
            if msg:
                q1 = msg.find('"')
                q2 = msg.find('"', q1 + 1) if q1 != -1 else -1
                if q1 != -1 and q2 != -1:
                    rl = msg[q1 + 1 : q2]
                    parts = rl.split()
                    if len(parts) >= 2:
                        path = parts[1]

        if not path:
            return True

        # Strip query string for matching
        path = str(path)
        # Some proxies/clients may emit absolute-form request-target in logs:
        # e.g. "GET https://example.com/metrics HTTP/1.1" -> extract "/metrics".
        if "://" in path:
            try:
                path = urlparse(path).path or path
            except Exception:
                # If parsing fails, fall back to the raw value.
                pass
        path = path.split("?", 1)[0]
        return not any(
            path.startswith(prefix) for prefix in self.excluded_path_prefixes
        )


def set_uvicorn_logging_configs(server_args=None):
    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = (
        "[%(asctime)s] %(levelprefix)s %(message)s"
    )
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = (
        '[%(asctime)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    _configure_uvicorn_access_log_filter(LOGGING_CONFIG, server_args)


def _configure_uvicorn_access_log_filter(
    uvicorn_logging_config: dict, server_args=None
):
    """Configure uvicorn access log path filter into uvicorn LOGGING_CONFIG.

    This optionally filters uvicorn access logs (e.g., suppress noisy /metrics polling).

    Args:
        uvicorn_logging_config: The dict-like LOGGING_CONFIG from uvicorn.
        server_args: Parsed server args object that may contain:
            - uvicorn_access_log_exclude_prefixes (list[str] | tuple[str] | None)
    """
    # Optionally filter uvicorn access logs (e.g., suppress noisy /metrics polling).
    if server_args is None:
        return

    filter_name = "sglang_uvicorn_access_path_filter"

    excluded_prefixes = getattr(
        server_args, "uvicorn_access_log_exclude_prefixes", None
    )
    if not excluded_prefixes:
        return

    # Normalize: accept list/tuple; treat a single string as one prefix (not an iterable of chars).
    if isinstance(excluded_prefixes, str):
        excluded_prefixes = [excluded_prefixes]

    # De-duplicate while keeping order; drop empty prefixes.
    excluded_prefixes = [p for p in excluded_prefixes if p]
    excluded_prefixes = list(dict.fromkeys(excluded_prefixes))
    if not excluded_prefixes:
        return

    uvicorn_logging_config.setdefault("filters", {})
    uvicorn_logging_config["filters"][filter_name] = {
        "()": "sglang.srt.entrypoints.server_utils.UvicornAccessLogFilter",
        "excluded_path_prefixes": excluded_prefixes,
    }

    # Attach filter to access handler and/or uvicorn.access logger (best-effort across uvicorn versions).
    handlers = uvicorn_logging_config.get("handlers", {})
    if "access" in handlers:
        filters_list = handlers["access"].setdefault("filters", [])
        if not isinstance(filters_list, list):
            filters_list = list(filters_list)
            handlers["access"]["filters"] = filters_list
        if filter_name not in filters_list:
            filters_list.append(filter_name)

    loggers_cfg = uvicorn_logging_config.get("loggers", {})
    if "uvicorn.access" in loggers_cfg:
        filters_list = loggers_cfg["uvicorn.access"].setdefault("filters", [])
        if not isinstance(filters_list, list):
            filters_list = list(filters_list)
            loggers_cfg["uvicorn.access"]["filters"] = filters_list
        if filter_name not in filters_list:
            filters_list.append(filter_name)


def build_server_info(server_args, scheduler_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build server metadata shared by HTTP and native gRPC.

    Callers add transport-specific fields and serialize the result.
    """
    result = redact_server_args_secrets(server_args.resolved_dict())
    result["launch_command"] = server_args.launch_command
    result.update(scheduler_info)
    result["kv_events"] = describe_kv_events_publisher(server_args)
    return result


def build_model_info_fields(model_config, server_args) -> Dict[str, Any]:
    """The `/model_info` fields that describe the loaded model: its
    modalities, `model_type`, `architectures` and, when the model has an
    embedding spec, its resolved embedding plan.

    Shared by the HTTP server and the Rust frontend's launch configuration.
    """
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.configs.embedding_model_spec import resolved_embedding_plan

    result = {
        "has_image_understanding": model_config.is_image_understandable_model,
        "has_audio_understanding": model_config.is_audio_understandable_model,
        "model_type": getattr(model_config.hf_config, "model_type", None),
        "architectures": getattr(model_config.hf_config, "architectures", None),
        # "hf_config": model_config.hf_config.to_dict(),
    }
    embedding_model_spec = getattr(model_config, "embedding_model_spec", None)
    if embedding_model_spec is not None:
        result["embedding"] = resolved_embedding_plan(
            embedding_model_spec,
            config=resolving_view(server_args),
            model_config=model_config,
        )
    return result


def start_follower_grpc_server(server_args, scheduler_info: Dict[str, Any]):
    """Expose GetServerInfo only, without constructing a TokenizerManager.

    The snapshot is taken after scheduler readiness. This does not launch a
    sidecar or enable inference/control RPCs.
    """
    serving = get_serving()
    if serving.grpc_port is None or serving.smg_grpc_mode or serving.grpc_mode:
        return None
    if envs.SGLANG_RUST_SERVER.get():
        return None

    from sglang.srt.rust_extensions import load_rust_extension
    from sglang.srt.sampling.watermarking.core import redact_watermark_secrets

    grpc_native = load_rust_extension("sglang.srt.rust_extensions._grpc")
    return grpc_native.start_metadata_server(
        host=serving.host,
        port=serving.grpc_port,
        server_info_json=json.dumps(
            msgspec_to_builtins(
                redact_watermark_secrets(build_server_info(server_args, scheduler_info))
            ),
            default=str,
        ),
    )


def launch_dummy_health_check_server(host, port, enable_metrics):
    import asyncio

    import uvicorn
    from fastapi import FastAPI, Response

    from sglang.srt.observability.trace import fastapi_telemetry_disabled_kwargs
    from sglang.srt.utils.network import NetworkAddress

    app = FastAPI(**fastapi_telemetry_disabled_kwargs())

    @app.get("/ping")
    async def ping():
        """Could be used by the checkpoint-engine update script to confirm the server is up."""
        return Response(status_code=200)

    @app.get("/health")
    async def health():
        """Check the health of the http server."""
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        """Check the health of the http server."""
        return Response(status_code=200)

    # Add prometheus middleware
    if enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=envs.SGLANG_TIMEOUT_KEEP_ALIVE.get(),
        loop="auto",
        log_config=None,
        log_level="warning",
    )
    server = uvicorn.Server(config=config)

    # Run server in a background daemon thread with its own event loop
    # This prevents blocking the main thread while still serving health checks
    def run_server():
        try:
            asyncio.run(server.serve())
        except Exception as e:
            logger.error(f"Dummy health check server failed to start: {e}")
            raise
        finally:
            logger.info(
                f"Dummy health check server stopped at {NetworkAddress(host, port).to_host_port_str()}"
            )

    thread = threading.Thread(
        target=run_server, daemon=True, name="health-check-server"
    )
    thread.start()
    logger.info(
        f"Dummy health check server started in background thread at {NetworkAddress(host, port).to_host_port_str()}"
    )

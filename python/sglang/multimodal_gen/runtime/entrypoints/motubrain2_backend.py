# SPDX-License-Identifier: Apache-2.0
"""The explicit `sglang serve --model-type motubrain2` backend."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from sglang.cli.serve_backends import (
    SERVE_BACKEND_API_VERSION,
    ServeBackend,
    ServeRequest,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from sglang.multimodal_gen.runtime.server_args import ServerArgs


def make_backend() -> ServeBackend:
    return ServeBackend(api_version=SERVE_BACKEND_API_VERSION, run=_run)


def create_motubrain2_app(
    server_args: ServerArgs, *, worker_pids: tuple[int, ...] = ()
) -> FastAPI:
    from sglang.multimodal_gen.runtime.entrypoints.action.motubrain2 import router
    from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app

    app = create_app(server_args, worker_pids=worker_pids)
    if server_args.media_gateway_key_file is None:
        app.include_router(router)
    return app


def _run(request: ServeRequest) -> None:
    from sglang.multimodal_gen.runtime.platforms.plugins import apply_plugin_hooks

    apply_plugin_hooks()
    from sglang.multimodal_gen.configs.pipeline_configs.motubrain2 import (
        Motubrain2PipelineConfig,
    )
    from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
        add_multimodal_gen_serve_args,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = argparse.ArgumentParser(
        prog="sglang serve --model-type motubrain2",
        description="Motubrain 2.0 action serving",
    )
    add_multimodal_gen_serve_args(parser)
    argv = list(request.argv)
    if "-h" in argv or "--help" in argv:
        parser.print_help()
        return
    args, unknown = parser.parse_known_args(argv)
    server_args = ServerArgs.from_cli_args(
        args,
        unknown,
        argv=argv,
        default_args={
            "pipeline_class_name": "Motubrain2Pipeline",
            "warmup_mode": "server",
        },
    )
    if server_args.pipeline_class_name != "Motubrain2Pipeline" or not isinstance(
        server_args.pipeline_config, Motubrain2PipelineConfig
    ):
        parser.error("the motubrain2 backend requires Motubrain2Pipeline")
    if server_args.disagg_role != RoleType.MONOLITHIC or server_args.disagg_mode:
        parser.error("Motubrain2 requires a monolithic policy worker")
    if server_args.webui:
        parser.error(
            "Motubrain2 serves robot actions over HTTP and WebSocket; omit --webui"
        )

    _serve(server_args)


def _serve(server_args: ServerArgs) -> None:
    import uvicorn

    from sglang.multimodal_gen.runtime.launch_server import (
        launch_server,
        shutdown_scheduler_processes,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.runtime.utils.trace_wrapper import init_diffusion_tracing

    if "NCCL_NVLS_ENABLE" not in os.environ or server_args.enable_nccl_nvls:
        os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    workers = launch_server(server_args, launch_http_server=False)
    try:
        init_diffusion_tracing(server_args, "DiffHTTPServer")
        set_global_server_args(server_args)
        app = create_motubrain2_app(
            server_args, worker_pids=tuple(p.pid for p in workers)
        )
        uvicorn.run(
            app,
            host=server_args.host,
            port=server_args.port,
            log_level=server_args.log_level,
            use_colors=True,
            ws_per_message_deflate=False,
            timeout_graceful_shutdown=server_args.http_graceful_shutdown_timeout,
        )
    finally:
        shutdown_scheduler_processes(server_args, workers)

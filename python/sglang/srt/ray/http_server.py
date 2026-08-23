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
"""Ray-aware HTTP server launcher."""

from typing import Callable, Optional

from ray.util.placement_group import PlacementGroup

from sglang.srt.entrypoints.engine import (
    init_tokenizer_manager,
    run_detokenizer_process,
    run_scheduler_process,
)
from sglang.srt.server_args import ServerArgs


def launch_engine(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    *,
    placement_group: Optional[PlacementGroup] = None,
):
    """Create RayEngine subprocesses / SchedulerActors."""
    from sglang.srt.ray.engine import RayEngine, _placement_group_context

    with _placement_group_context(placement_group):
        return RayEngine._launch_subprocesses(
            server_args,
            init_tokenizer_manager_func=init_tokenizer_manager_func,
            run_scheduler_process_func=run_scheduler_process_func,
            run_detokenizer_process_func=run_detokenizer_process_func,
        )


def serve_http(
    engine,
    server_args: ServerArgs,
    execute_warmup_func: Optional[Callable] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Block in uvicorn. ``engine`` is the tuple from ``launch_engine``."""
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
    )

    if execute_warmup_func is None:
        execute_warmup_func = _execute_server_warmup

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
        _weight_cache_daemon_procs,
    ) = engine

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )


def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Optional[Callable] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """Launch HTTP server with Ray-based scheduler actors.

    Mirrors http_server.launch_server() but uses RayEngine for scheduler launching.
    """
    serve_http(
        launch_engine(
            server_args,
            init_tokenizer_manager_func=init_tokenizer_manager_func,
            run_scheduler_process_func=run_scheduler_process_func,
            run_detokenizer_process_func=run_detokenizer_process_func,
        ),
        server_args,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )

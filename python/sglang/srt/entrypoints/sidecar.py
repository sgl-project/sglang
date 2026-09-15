# Copyright 2023-2026 SGLang Team
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
"""Lifecycle management for full and telemetry-only local sidecars."""

import argparse
import dataclasses
import importlib
import json
import logging
import multiprocessing as mp
import os

from sglang.srt.entrypoints.sidecar_context import KvEventSource, SidecarContext
from sglang.srt.runtime_context import get_parallel, get_serving
from sglang.srt.utils.common import kill_itself_when_parent_died, kill_process_tree
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.watchdog import SubprocessWatchdog

logger = logging.getLogger(__name__)

SGLANG_GRPC_ENDPOINT_ENV = "SGLANG_GRPC_ENDPOINT"
SGLANG_SIDECAR_CONTEXT_ENV = "SGLANG_SIDECAR_CONTEXT"
_DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT = 45.0


def build_sidecar_context(sources: list[KvEventSource]) -> SidecarContext | None:
    if get_serving().sidecar_scope != "local-telemetry":
        return None
    parallel = get_parallel()
    if any(not 0 <= source.dp_rank < parallel.dp_size for source in sources):
        raise ValueError("Local KV-event publisher rank is outside the DP topology")
    return SidecarContext(
        mode="full" if parallel.node_rank == 0 else "telemetry",
        node_rank=parallel.node_rank,
        nnodes=parallel.nnodes,
        dp_size=parallel.dp_size,
        dist_init_addr=(
            NetworkAddress.parse(parallel.dist_init_addr).to_tcp()
            if parallel.dist_init_addr
            else None
        ),
        kv_event_sources=sources,
    )


def _loopback_host(host: str) -> str:
    if not host or host == "0.0.0.0":
        return "127.0.0.1"
    if host in ("::", "[::]"):
        return "::1"
    return host


def build_sidecar_endpoint(host: str, grpc_port: int) -> str:
    """Both halves are passed in: this is a string helper, and the caller is
    the one that knows where the effective values live."""
    return NetworkAddress(_loopback_host(host), grpc_port).to_url()


def _parse_sidecar_args(args: list[str] | None) -> tuple[list[str], float]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--sidecar-shutdown-timeout",
        type=float,
        default=_DEFAULT_SIDECAR_SHUTDOWN_TIMEOUT,
    )
    parsed, provider_args = parser.parse_known_args(args or [])
    if parsed.sidecar_shutdown_timeout <= 0:
        raise ValueError("--sidecar-shutdown-timeout must be greater than 0.")
    return provider_args, parsed.sidecar_shutdown_timeout


def _run_sidecar(
    module_name: str,
    args: list[str],
    endpoint: str | None,
    context: SidecarContext | None = None,
) -> None:
    kill_itself_when_parent_died()
    if endpoint is None:
        os.environ.pop(SGLANG_GRPC_ENDPOINT_ENV, None)
    else:
        os.environ[SGLANG_GRPC_ENDPOINT_ENV] = endpoint
    if context is None:
        os.environ.pop(SGLANG_SIDECAR_CONTEXT_ENV, None)
    else:
        os.environ[SGLANG_SIDECAR_CONTEXT_ENV] = json.dumps(dataclasses.asdict(context))
    try:
        main = getattr(importlib.import_module(module_name), "main")
    except (AttributeError, ImportError) as e:
        raise RuntimeError(
            f"--sidecar requires importable module {module_name!r} "
            "with a main(argv) function."
        ) from e

    if not callable(main):
        raise RuntimeError(
            f"--sidecar requires module {module_name!r} to expose "
            "a callable main(argv)."
        )

    main(args)


class Sidecar:
    def __init__(
        self,
        proc,
        module_name: str,
        shutdown_timeout: float,
        *,
        allow_clean_exit: bool = True,
    ):
        self.proc = proc
        self.module_name = module_name
        self.shutdown_timeout = shutdown_timeout
        self._stopped = False
        self._watchdog = SubprocessWatchdog(
            processes=[proc],
            process_names=[module_name],
            allow_clean_exit=allow_clean_exit,
        )

    def start(self) -> None:
        self.proc.start()
        self._watchdog.start()
        logger.info(
            "Sidecar module %s started pid=%s",
            self.module_name,
            self.proc.pid,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._watchdog.stop()
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=self.shutdown_timeout)
        else:
            self.proc.join(timeout=0)

        if self.proc.is_alive():
            logger.warning("Sidecar module did not terminate; killing process tree")
            kill_process_tree(self.proc.pid, wait_timeout=self.shutdown_timeout)


def start_sidecar(context: SidecarContext | None = None) -> Sidecar:
    module_name = get_serving().sidecar
    assert module_name is not None
    sidecar_args, shutdown_timeout = _parse_sidecar_args(get_serving().sidecar_args)
    endpoint = (
        None
        if context is not None and context.mode == "telemetry"
        else build_sidecar_endpoint(get_serving().host, get_serving().grpc_port)
    )
    mp_context = mp.get_context("spawn")
    process_args = (module_name, sidecar_args, endpoint)
    lifecycle_args = {}
    if context is not None:
        process_args += (context,)
        lifecycle_args = dict(allow_clean_exit=False)
    proc = mp_context.Process(
        name=f"sglang_sidecar_{module_name}",
        target=_run_sidecar,
        args=process_args,
    )
    sidecar = Sidecar(
        proc,
        module_name,
        shutdown_timeout=shutdown_timeout,
        **lifecycle_args,
    )
    sidecar.start()
    return sidecar

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
"""Ray actor wrapper for SGLang HTTP Server.

IMPORTANT: No sglang imports at module level. This module is imported
on the CPU-only head/driver node.

Usage:
    import ray
    ray.init()
    from sglang.srt.entrypoints.http_server_actor import HttpServerActor

    server = HttpServerActor(model_path="meta-llama/Llama-3-8B", tp_size=8, port=30000, use_ray=True)
    server.wait_until_ready()
    url = server.get_url()
    # ... use url for HTTP requests ...
    server.shutdown()
"""

from __future__ import annotations

import logging

import ray

logger = logging.getLogger(__name__)


def _run_http_server(server_kwargs, placement_groups=None):
    """Module-level function for subprocess (picklable). Runs on ray worker."""
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(**server_kwargs)
    launch_server(server_args, _ray_placement_groups=placement_groups)


@ray.remote
class _InternalHttpServerActor:
    """Internal Ray actor that runs SGLang HTTP server in a subprocess."""

    def __init__(self, **server_kwargs):
        import multiprocessing as mp

        self._pgs = server_kwargs.pop("_ray_placement_groups", None)
        self.port = server_kwargs.get("port", 30000)
        self.node_ip = ray.util.get_node_ip_address()
        ctx = mp.get_context("spawn")
        self._server_process = ctx.Process(
            target=_run_http_server,
            args=(server_kwargs, self._pgs),
            daemon=False,
        )
        self._server_process.start()

        logger.info(
            f"HttpServerActor started on {self.node_ip}:{self.port}, "
            f"pid={self._server_process.pid}"
        )

    def wait_until_ready(self, timeout: int = 600) -> bool:
        """Wait for the server to be ready."""
        import time

        import requests

        url = self.get_url()
        health_url = f"{url}/health"
        start = time.time()

        logger.info(f"Waiting for server at {health_url} to be ready...")

        while time.time() - start < timeout:
            if not self.is_alive():
                logger.error("Server process died while waiting for ready")
                return False
            try:
                r = requests.get(health_url, timeout=5)
                if r.status_code == 200:
                    logger.info(
                        f"Server ready after {time.time() - start:.1f}s"
                    )
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        logger.error(f"Server not ready after {timeout}s timeout")
        return False

    def get_url(self) -> str:
        """Get the server URL."""
        return f"http://{self.node_ip}:{self.port}"

    def get_node_ip(self) -> str:
        """Get the node IP where server is running."""
        return self.node_ip

    def is_alive(self) -> bool:
        """Check if server process is running."""
        return self._server_process is not None and self._server_process.is_alive()

    def get_status(self) -> dict:
        """Get server status in a single RPC call."""
        return {
            "url": self.get_url(),
            "node_ip": self.node_ip,
            "alive": self.is_alive(),
        }

    def shutdown(self):
        """Shutdown the server."""
        if self._server_process is not None and self._server_process.is_alive():
            logger.info("Shutting down HTTP server...")
            self._server_process.terminate()
            self._server_process.join(timeout=30)
            if self._server_process.is_alive():
                logger.warning("Server didn't terminate, killing...")
                self._server_process.kill()
                self._server_process.join()
            logger.info("HTTP server shutdown complete")


class HttpServerActor:
    """Wrapper that manages a Ray actor running SGLang HTTP server.

    Safe to import and instantiate on CPU-only head nodes.

    Usage:
        server = HttpServerActor(model_path="...", tp_size=8, port=30000, use_ray=True)
        server.wait_until_ready()
        url = server.get_url()
        # ... use url for HTTP requests ...
        server.shutdown()
    """

    def __init__(self, **server_kwargs):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from sglang.srt.entrypoints.ray_utils import create_placement_groups

        pgs = create_placement_groups(
            tp_size=server_kwargs.get("tp_size", 1),
            pp_size=server_kwargs.get("pp_size", 1),
            nnodes=server_kwargs.get("nnodes", 1),
        )
        self._placement_groups = pgs

        self._actor = _InternalHttpServerActor.options(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pgs[0],
                placement_group_bundle_index=0,
            ),
        ).remote(_ray_placement_groups=pgs, **server_kwargs)

    def wait_until_ready(self, timeout=600) -> bool:
        return ray.get(self._actor.wait_until_ready.remote(timeout))

    def get_url(self) -> str:
        return ray.get(self._actor.get_url.remote())

    def get_status(self) -> dict:
        return ray.get(self._actor.get_status.remote())

    def shutdown(self):
        if self._actor is not None:
            ray.get(self._actor.shutdown.remote())
            self._actor = None
        if self._placement_groups:
            for pg in self._placement_groups:
                try:
                    ray.util.remove_placement_group(pg)
                except Exception:
                    pass
            self._placement_groups = None

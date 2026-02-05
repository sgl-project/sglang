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

This module provides a Ray actor that wraps the SGLang HTTP server, allowing
sglang to be imported INSIDE the actor (on GPU worker nodes) rather than
on the head node which may not have GPU access.

Usage:
    import ray
    ray.init()

    from sglang.srt.entrypoints.http_server_actor import create_http_server_actor_class

    HttpServerActor = create_http_server_actor_class()
    server = HttpServerActor.options(num_cpus=1).remote(
        model_path="meta-llama/Llama-3-8B",
        tp_size=8,
        port=30000,
        use_ray=True,  # Required for multi-node
    )

    # Wait for server to be ready
    ray.get(server.wait_until_ready.remote())

    # Get the URL
    url = ray.get(server.get_url.remote())
    # Use url for HTTP requests...

    # Shutdown when done
    ray.get(server.shutdown.remote())
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _run_http_server(server_kwargs):
    """Module-level function to run HTTP server (picklable for multiprocessing).

    All sglang imports happen here, on the GPU worker node.

    Args:
        server_kwargs: Dictionary of arguments passed to ServerArgs.
    """
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(**server_kwargs)
    launch_server(server_args)


def create_http_server_actor_class():
    """Factory function to create HttpServerActor class with Ray decorator.

    Returns a Ray actor class that wraps sglang HTTP server. The server is
    started in a subprocess INSIDE the actor, which runs on a GPU worker node.
    This is necessary when the head node doesn't have GPU access or when
    sglang imports would fail on the head node.
    """
    import ray

    @ray.remote
    class HttpServerActor:
        """Ray actor that runs SGLang HTTP server on a GPU worker node.

        The HTTP server is started in a subprocess within this actor, ensuring
        all sglang imports happen on a node with GPU access. This allows the
        head node to remain CPU-only while still orchestrating GPU workloads.
        """

        def __init__(self, **server_kwargs):
            """Initialize and start the HTTP server.

            Args:
                **server_kwargs: Arguments passed to ServerArgs.
                    Required: model_path
                    Recommended: tp_size, port, use_ray=True
            """
            import multiprocessing as mp

            import ray

            self.port = server_kwargs.get("port", 30000)
            self.node_ip = ray.util.get_node_ip_address()
            self._server_kwargs = server_kwargs

            # Use spawn method for clean subprocess
            ctx = mp.get_context("spawn")

            # Start server in subprocess for clean lifecycle management
            # Use module-level function for picklability with spawn context
            self._server_process = ctx.Process(
                target=_run_http_server,
                args=(server_kwargs,),
                daemon=False,  # Allow child processes (SGLang spawns workers internally)
            )
            self._server_process.start()

            logger.info(
                f"HttpServerActor started on {self.node_ip}:{self.port}, "
                f"pid={self._server_process.pid}"
            )

        def wait_until_ready(self, timeout: int = 600) -> bool:
            """Wait for the server to be ready.

            Args:
                timeout: Maximum time to wait in seconds.

            Returns:
                True if server is ready, False if timeout or server died.
            """
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
            """Get the server URL.

            Returns:
                The full URL including protocol, host, and port.
            """
            return f"http://{self.node_ip}:{self.port}"

        def get_node_ip(self) -> str:
            """Get the node IP where server is running.

            Returns:
                The IP address of the worker node.
            """
            return self.node_ip

        def is_alive(self) -> bool:
            """Check if server process is running.

            Returns:
                True if the server subprocess is still alive.
            """
            return self._server_process is not None and self._server_process.is_alive()

        def shutdown(self):
            """Shutdown the server.

            Terminates the server subprocess gracefully, with a fallback to
            forceful kill if the process doesn't respond.
            """
            if self._server_process is not None and self._server_process.is_alive():
                logger.info("Shutting down HTTP server...")
                self._server_process.terminate()
                self._server_process.join(timeout=30)
                if self._server_process.is_alive():
                    logger.warning("Server didn't terminate, killing...")
                    self._server_process.kill()
                    self._server_process.join()
                logger.info("HTTP server shutdown complete")

    return HttpServerActor

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

    from sglang.srt.entrypoints.http_server_actor import create_and_launch_http_server_actor

    server = create_and_launch_http_server_actor(
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
import os

logger = logging.getLogger(__name__)


def _run_http_server(
    server_kwargs,
    ray_cluster_info=None,
):
    """Module-level function to run HTTP server (picklable for multiprocessing).

    All sglang imports happen here, on the GPU worker node.

    Args:
        server_kwargs: Dictionary of arguments passed to ServerArgs.
        ray_cluster_info: Pre-computed RayClusterInfo for Ray multi-node.
    """
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(**server_kwargs)
    launch_server(
        server_args,
        ray_cluster_info=ray_cluster_info,
    )


def create_and_launch_http_server_actor(**server_kwargs):
    """Launch HttpServerActor on rank 0's worker node for optimal IPC communication.

    This function:
    1. Discovers GPU topology in the Ray cluster
    2. Creates per-node placement groups
    3. Schedules HttpServerActor on rank 0's node (ensuring IPC works for ZMQ)
    4. Passes pre-computed topology/PGs through the subprocess to launch_server

    Args:
        **server_kwargs: All arguments passed to ServerArgs.
            Common args: model_path, tp_size, pp_size, port, use_ray, etc.

    Returns:
        Ray actor handle for the HttpServerActor.
    """
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from sglang.srt.utils.ray_cluster_utils import (
        RayClusterInfo,
        create_cluster_topology,
        create_per_node_placement_groups,
        discover_gpu_nodes,
    )

    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    if not ray.is_initialized():
        ray.init()

    tp_size = server_kwargs.get("tp_size", 1)
    pp_size = server_kwargs.get("pp_size", 1)
    world_size = tp_size * pp_size

    gpu_nodes = discover_gpu_nodes()
    if len(gpu_nodes) == 0:
        raise RuntimeError("No GPU nodes found in Ray cluster")

    topology = create_cluster_topology(gpu_nodes, world_size)
    placement_groups = create_per_node_placement_groups(topology)

    logger.info(
        f"Launching HttpServerActor on rank 0's node: "
        f"{topology.nnodes} nodes, {topology.gpus_per_node} GPUs/node, "
        f"world_size={world_size}"
    )

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
                    Internal: ray_cluster_info (RayClusterInfo)
            """
            import multiprocessing as mp

            import ray
            from sglang.srt.utils.ray_cluster_utils import RayClusterInfo

            ray_cluster_info = server_kwargs.pop("ray_cluster_info", None) or RayClusterInfo()

            self.port = server_kwargs.get("port", 30000)
            # This actor runs on PG[0] (rank 0's node), so local IP = rank 0 IP.
            # Pass it through to avoid a probe actor round-trip (~100-500ms).
            self.node_ip = ray.util.get_node_ip_address()
            ray_cluster_info.rank0_node_ip = self.node_ip
            self._server_kwargs = server_kwargs

            # Use spawn method for clean subprocess
            ctx = mp.get_context("spawn")

            # Start server in subprocess for clean lifecycle management
            # Use module-level function for picklability with spawn context
            self._server_process = ctx.Process(
                target=_run_http_server,
                args=(server_kwargs, ray_cluster_info),
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
            """Get the server URL."""
            return f"http://{self.node_ip}:{self.port}"

        def get_node_ip(self) -> str:
            """Get the node IP where server is running."""
            return self.node_ip

        def is_alive(self) -> bool:
            """Check if server process is running."""
            return self._server_process is not None and self._server_process.is_alive()

        def get_status(self) -> dict:
            """Get server status in a single RPC call.

            Returns dict with url, node_ip, and alive fields.
            Use this instead of separate get_url/get_node_ip/is_alive calls
            to avoid multiple round-trips.
            """
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

    server_actor = HttpServerActor.options(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_groups[0],
            placement_group_bundle_index=0,
        ),
    ).remote(
        ray_cluster_info=RayClusterInfo(
            topology=topology,
            placement_groups=placement_groups,
        ),
        **server_kwargs,
    )

    return server_actor

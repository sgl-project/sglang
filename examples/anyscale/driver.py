"""
Multi-Node SGLang Server Driver Script for Anyscale

This script launches SGLang servers across multiple Ray nodes using Ray actors.
Each node runs a sglang.launch_server process with tensor parallelism.

Usage:
    python driver.py [--model MODEL_PATH] [--tp TP_SIZE] [--nnodes NUM_NODES]
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

import ray
import requests


@ray.remote
class SGLangServerActor:
    """Ray actor that runs an SGLang server on a specific node."""

    def __init__(
        self,
        model_path: str,
        tp_size: int,
        pp_size: int,
        nnodes: int,
        node_rank: int,
        host: str = "0.0.0.0",
        port: int = 30000,
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.nnodes = nnodes
        self.node_rank = node_rank
        self.host = host
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self.node_ip = ray.util.get_node_ip_address()

    def get_node_ip(self) -> str:
        """Return the IP address of this node."""
        return self.node_ip

    def start_server(self, dist_init_addr: str) -> dict:
        """Start the SGLang server process.

        Args:
            dist_init_addr: The distributed init address (should be node_rank=0's IP:port).
                           This must be the SAME for all nodes.
        """
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tp",
            str(self.tp_size),
            "--pp-size",
            str(self.pp_size),
            "--dist-init-addr",
            dist_init_addr,
            "--nnodes",
            str(self.nnodes),
            "--node-rank",
            str(self.node_rank),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Calculate GPUs per node: (tp * pp) / nnodes
        gpus_per_node = (self.tp_size * self.pp_size) // self.nnodes

        print(f"[Node {self.node_rank}] Starting SGLang server on {self.node_ip}")
        print(f"[Node {self.node_rank}] TP: {self.tp_size}, PP: {self.pp_size}, GPUs on this node: {gpus_per_node}")
        print(f"[Node {self.node_rank}] dist-init-addr: {dist_init_addr}")
        print(f"[Node {self.node_rank}] Command: {' '.join(cmd)}")

        # Set environment variables for NCCL
        # Each node only uses gpus_per_node GPUs (tp_size / nnodes)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpus_per_node))

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        # Start background thread to stream subprocess output
        def stream_output():
            for line in self.process.stdout:
                print(f"[Node {self.node_rank}] {line}", end="", flush=True)

        self._log_thread = threading.Thread(target=stream_output, daemon=True)
        self._log_thread.start()

        return {
            "node_rank": self.node_rank,
            "node_ip": self.node_ip,
            "pid": self.process.pid,
            "port": self.port,
        }

    def is_alive(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_logs(self, num_lines: int = 50) -> str:
        """Get recent log output from the server."""
        if self.process is None:
            return "Server not started"

        # Read available output (non-blocking)
        logs = []
        try:
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                logs.append(line)
        except Exception:
            pass

        return "".join(logs[-num_lines:])

    def stop_server(self) -> bool:
        """Stop the server process."""
        if self.process is None:
            return True

        print(f"[Node {self.node_rank}] Stopping SGLang server...")
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        return True


def get_node_ips() -> List[str]:
    """Get IP addresses of all Ray nodes."""
    nodes = ray.nodes()
    node_ips = []
    for node in nodes:
        if node["Alive"]:
            node_ip = node["NodeManagerAddress"]
            node_ips.append(node_ip)
    return node_ips


def create_placement_groups(num_nodes: int, gpus_per_node: int):
    """Create placement groups to ensure actors are placed on different nodes."""
    from ray.util.placement_group import placement_group, placement_group_table

    pgs = []
    for i in range(num_nodes):
        # Create a placement group that requires GPUs on a single node
        bundles = [{"GPU": gpus_per_node, "CPU": 1}]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())
        pgs.append(pg)
        print(f"Created placement group {i}: {placement_group_table(pg)}")

    return pgs


def wait_for_server_ready(url: str, timeout: int = 600) -> bool:
    """Wait for the SGLang server to be ready."""
    start_time = time.time()
    health_url = f"{url}/health"

    print(f"Waiting for server at {health_url} to be ready...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                print(f"Server at {url} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(5)
        elapsed = int(time.time() - start_time)
        print(f"Still waiting... ({elapsed}s elapsed)")

    print(f"Timeout waiting for server at {url}")
    return False


def test_server(url: str) -> bool:
    """Test the SGLang server with a simple request."""
    print(f"\n{'='*60}")
    print(f"Testing SGLang server at {url}")
    print(f"{'='*60}")

    # Test generate endpoint
    generate_url = f"{url}/generate"
    payload = {
        "text": "The capital of France is",
        "sampling_params": {
            "max_new_tokens": 32,
            "temperature": 0.0,
        },
    }

    try:
        print(f"\nSending request to {generate_url}")
        print(f"Payload: {payload}")

        response = requests.post(generate_url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        print(f"\nResponse: {result}")
        print(f"\nGenerated text: {result.get('text', 'N/A')}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error testing server: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Node SGLang Server Driver")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model path (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=4,
        help="Total tensor parallelism size across all nodes",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="Pipeline parallelism size",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=2,
        help="Number of nodes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port",
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=20000,
        help="Distributed init port",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests, don't start servers",
    )
    args = parser.parse_args()

    # Initialize Ray (will connect to existing cluster in Anyscale)
    if not ray.is_initialized():
        ray.init()

    # Calculate GPUs per node: (tp * pp) / nnodes
    world_size = args.tp * args.pp_size
    gpus_per_node = world_size // args.nnodes
    if world_size % args.nnodes != 0:
        print(f"Error: world_size ({world_size} = tp*pp) must be divisible by --nnodes ({args.nnodes})")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Multi-Node SGLang Server Driver")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Tensor Parallelism: {args.tp}")
    print(f"Pipeline Parallelism: {args.pp_size}")
    print(f"World Size: {world_size}")
    print(f"Number of Nodes: {args.nnodes}")
    print(f"GPUs per Node: {gpus_per_node}")
    print(f"Port: {args.port}")
    print(f"Dist Init Port: {args.dist_init_port}")

    # Get cluster information
    node_ips = get_node_ips()
    print(f"\nCluster nodes: {node_ips}")

    if len(node_ips) < args.nnodes:
        print(
            f"Error: Need {args.nnodes} nodes but only {len(node_ips)} available"
        )
        sys.exit(1)

    # Create placement groups to ensure actors are on different nodes
    # Each placement group requests gpus_per_node GPUs
    print("\nCreating placement groups...")
    pgs = create_placement_groups(args.nnodes, gpus_per_node)

    # Create actors with placement constraints (without dist_init_addr yet)
    print("\nCreating SGLang server actors...")
    actors = []
    for node_rank in range(args.nnodes):
        # Create actor with specific placement group
        # Each actor gets gpus_per_node GPUs
        actor = SGLangServerActor.options(
            num_gpus=gpus_per_node,
            placement_group=pgs[node_rank],
            placement_group_bundle_index=0,
        ).remote(
            model_path=args.model_path,
            tp_size=args.tp,
            pp_size=args.pp_size,
            nnodes=args.nnodes,
            node_rank=node_rank,
            host="0.0.0.0",
            port=args.port,
        )
        actors.append(actor)

    # Get actual node IPs from actors AFTER placement
    actor_ips = ray.get([actor.get_node_ip.remote() for actor in actors])
    print(f"\nActor node IPs: {actor_ips}")

    # Use actor 0's (node_rank=0) actual IP as dist-init-addr
    # This ensures all nodes use the same address pointing to node 0
    dist_init_addr = f"{actor_ips[0]}:{args.dist_init_port}"
    print(f"Dist init address (from node_rank=0): {dist_init_addr}")

    # Verify all actors are on different nodes
    if len(set(actor_ips)) != args.nnodes:
        print(f"Warning: Some actors might be on the same node! IPs: {actor_ips}")

    # Start all servers with the SAME dist_init_addr
    print("\nStarting SGLang servers on all nodes...")
    start_results = ray.get([
        actor.start_server.remote(dist_init_addr) for actor in actors
    ])

    for result in start_results:
        print(f"  Node {result['node_rank']}: IP={result['node_ip']}, "
              f"PID={result['pid']}, Port={result['port']}")

    # Wait for the master server (node 0) to be ready
    master_ip = actor_ips[0]
    master_url = f"http://{master_ip}:{args.port}"

    print(f"\nMaster server URL: {master_url}")

    if not wait_for_server_ready(master_url, timeout=600):
        print("Failed to start SGLang server")
        # Cleanup
        ray.get([actor.stop_server.remote() for actor in actors])
        sys.exit(1)

    # Test the server
    print("\n" + "=" * 60)
    print("Running server tests...")
    print("=" * 60)

    test_success = test_server(master_url)

    if test_success:
        print("\n" + "=" * 60)
        print("SUCCESS: Multi-node SGLang server is working!")
        print("=" * 60)
        print(f"\nServer is running at: {master_url}")
        print("Press Ctrl+C to stop the server...")

        # Keep the server running
        try:
            while True:
                # Check if all actors are still alive
                alive_status = ray.get(
                    [actor.is_alive.remote() for actor in actors]
                )
                if not all(alive_status):
                    print("Some servers have stopped unexpectedly")
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nShutting down...")

    # Cleanup
    print("\nStopping all servers...")
    ray.get([actor.stop_server.remote() for actor in actors])
    print("Done!")

    sys.exit(0 if test_success else 1)


if __name__ == "__main__":
    main()

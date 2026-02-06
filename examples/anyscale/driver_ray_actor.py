"""
Multi-Node SGLang Driver using Ray Actor Backend

This script tests the unified Ray actor scheduler launching that automatically
discovers GPU nodes and creates per-node placement groups.

Unlike driver.py (which launches separate processes per node), this uses
an EngineActor that imports and creates sglang.Engine INSIDE the actor.
The actor runs on a GPU worker node, so sglang imports work correctly.

Usage:
    python driver_ray_actor.py [--model MODEL_PATH] [--tp TP_SIZE]

Key Design:
    The head node (which runs this script) does NOT import sglang at all.
    Instead, it creates an EngineActor (defined inline) that runs on a GPU worker.
    sglang is imported INSIDE the actor, ensuring GPU access during import.
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Node SGLang Driver using Ray Actor Backend"
    )
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
        help="Total tensor parallelism size (will be distributed across nodes automatically)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallelism size",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port",
    )
    args = parser.parse_args()

    # Import ray (no sglang import on head node!)
    import ray

    # Initialize Ray (connects to existing Anyscale cluster)
    if not ray.is_initialized():
        ray.init()

    # Print cluster info
    print(f"\n{'='*60}")
    print("Multi-Node SGLang Driver (Ray Actor Backend)")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Tensor Parallelism: {args.tp}")
    print(f"Pipeline Parallelism: {args.pp_size}")
    print(f"World Size: {args.tp * args.pp_size}")
    print(f"Port: {args.port}")

    # Print Ray cluster resources
    cluster_resources = ray.cluster_resources()
    print(f"\nRay Cluster Resources:")
    print(f"  Total GPUs: {cluster_resources.get('GPU', 0)}")
    print(f"  Total CPUs: {cluster_resources.get('CPU', 0)}")

    # Print node information
    nodes = ray.nodes()
    gpu_nodes = [n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    print(f"\nGPU Nodes ({len(gpu_nodes)} total):")
    for i, node in enumerate(gpu_nodes):
        gpu_count = node["Resources"].get("GPU", 0)
        node_ip = node["NodeManagerAddress"]
        print(f"  Node {i}: {node_ip} ({int(gpu_count)} GPUs)")

    # Define EngineActor INLINE to avoid any sglang imports on head node
    # sglang is imported INSIDE the actor on a GPU worker node
    print(f"\n{'='*60}")
    print("Creating EngineActor on GPU worker node...")
    print(f"{'='*60}")

    @ray.remote
    class EngineActor:
        """Ray actor that wraps sglang.Engine.

        sglang is imported INSIDE the actor, which runs on a GPU worker node.
        This avoids import errors on the head node which may not have GPU access.
        """

        def __init__(self, **engine_kwargs):
            # Import sglang INSIDE the actor (runs on GPU worker node)
            from sglang import Engine

            print(f"Creating Engine with kwargs: {list(engine_kwargs.keys())}")
            self.engine = Engine(**engine_kwargs)
            print("Engine created successfully")

        def is_ready(self):
            return self.engine is not None

        def generate(self, **kwargs):
            return self.engine.generate(**kwargs)

        def get_server_info(self):
            return self.engine.get_server_info()

        def shutdown(self):
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None

    # Don't request GPUs here - the internal Engine with use_ray=True
    # handles GPU allocation via placement groups for SchedulerActors
    engine_actor = EngineActor.options(
        num_gpus=0,
        num_cpus=1,
    ).remote(
        model_path=args.model_path,
        tp_size=args.tp,
        pp_size=args.pp_size
        ,
        port=args.port,
        use_ray=True,  # Enable Ray actor backend for internal scheduler actors
    )

    # Wait for engine to be ready
    print("Waiting for engine initialization...")
    ray.get(engine_actor.is_ready.remote())

    print(f"\n{'='*60}")
    print("Engine initialized successfully!")
    print(f"{'='*60}")

    # Test the engine with a simple generation
    print("\nRunning test generation...")
    test_prompt = "The capital of France is"

    result = ray.get(
        engine_actor.generate.remote(
            prompt=test_prompt,
            sampling_params={
                "max_new_tokens": 32,
                "temperature": 0.0,
            },
        )
    )

    print(f"\nPrompt: {test_prompt}")
    print(f"Generated: {result['text']}")

    # Run a few more tests
    print(f"\n{'='*60}")
    print("Running additional tests...")
    print(f"{'='*60}")

    test_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a haiku about programming:",
        "What is 2 + 2?",
    ]

    for prompt in test_prompts:
        start_time = time.time()
        result = ray.get(
            engine_actor.generate.remote(
                prompt=prompt,
                sampling_params={
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                },
            )
        )
        elapsed = time.time() - start_time
        print(f"\nPrompt: {prompt}")
        print(f"Generated ({elapsed:.2f}s): {result['text'][:200]}...")

    print(f"\n{'='*60}")
    print("SUCCESS: Multi-node Ray Actor Backend is working!")
    print(f"{'='*60}")

    # Get server info
    server_info = ray.get(engine_actor.get_server_info.remote())
    print(f"\nServer Info:")
    print(f"  Model: {server_info.get('model_path', 'N/A')}")
    print(f"  TP Size: {server_info.get('tp_size', 'N/A')}")
    print(f"  PP Size: {server_info.get('pp_size', 'N/A')}")
    print(f"  Max Total Tokens: {server_info.get('max_total_num_tokens', 'N/A')}")

    # Keep running for manual testing (optional)
    print("\nEngine is ready. Press Ctrl+C to shutdown...")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Cleanup
    ray.get(engine_actor.shutdown.remote())
    print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

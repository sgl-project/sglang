#!/usr/bin/env python3
"""
Test cross-node transfer engine info synchronization on a single machine.

This test simulates a 2-node setup on a single machine with 4 GPUs:
- "Node 0" uses GPUs 0-1 (TP ranks 0-1)
- "Node 1" uses GPUs 2-3 (TP ranks 2-3)
- Total TP=4 spanning both nodes (this is key for cross-node sync to be meaningful)

Usage:
    # Terminal 1 (start node 0 first):
    python test_cross_node_transfer_engine_info.py --node-rank 0

    # Terminal 2 (start node 1):
    python test_cross_node_transfer_engine_info.py --node-rank 1

    # Terminal 3 (after both nodes are up, test the API):
    curl "http://127.0.0.1:30000/get_remote_instance_transfer_engine_info?rank=0"
    curl "http://127.0.0.1:30000/get_remote_instance_transfer_engine_info?rank=1"
    curl "http://127.0.0.1:30000/get_remote_instance_transfer_engine_info?rank=2"  # This should work now!
    curl "http://127.0.0.1:30000/get_remote_instance_transfer_engine_info?rank=3"  # This should work now!

Requirements:
    - 4 GPUs on single machine
    - Model that supports TP=4 (e.g., small models like Qwen2.5-1.5B-Instruct)
"""

import argparse
import subprocess
import sys
import time

import requests


def get_local_ip():
    """Get local IP for dist-init-addr."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def launch_node(node_rank: int, model_path: str, dist_init_addr: str):
    """Launch a simulated node."""
    # Each "node" uses 2 GPUs, but TP spans across nodes (TP=4 total)
    gpus_per_node = 2
    total_tp_size = 4  # TP spans both nodes
    base_gpu_id = node_rank * gpus_per_node
    port = 30000 + node_rank * 100  # 30000 for node 0, 30100 for node 1

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(total_tp_size),  # TP=4 spans both nodes
        "--nnodes",
        "2",
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        dist_init_addr,
        "--base-gpu-id",
        str(base_gpu_id),
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
        # Enable transfer engine mode
        "--remote-instance-weight-loader-start-seed-via-transfer-engine",
    ]

    print(f"[Node {node_rank}] Launching with command:")
    print(" ".join(cmd))
    print()

    # Run in foreground so we can see logs
    subprocess.run(cmd)


def test_api():
    """Test that all ranks are accessible from node 0."""
    base_url = "http://127.0.0.1:30000"

    print("Testing transfer engine info API...")
    print("=" * 60)

    # Wait for server to be ready
    for i in range(60):
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                print("Server is healthy!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server not ready after 120 seconds")
        return False

    # Test all 4 ranks
    all_passed = True
    for rank in range(4):
        try:
            resp = requests.get(
                f"{base_url}/get_remote_instance_transfer_engine_info",
                params={"rank": rank},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"✓ Rank {rank}: SUCCESS - session_id present: {'remote_instance_transfer_engine_info' in data}"
                )
            else:
                print(f"✗ Rank {rank}: FAILED - status {resp.status_code}")
                all_passed = False
        except Exception as e:
            print(f"✗ Rank {rank}: ERROR - {e}")
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - cross-node sync may not be working")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test cross-node transfer engine info sync"
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        choices=[0, 1],
        help="Node rank (0 or 1) to launch. Omit to run API tests only.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model path (should fit in 2 GPUs)",
    )
    parser.add_argument(
        "--dist-init-addr",
        type=str,
        default=None,
        help="Distributed init address (default: auto-detect)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run API tests, don't launch servers",
    )

    args = parser.parse_args()

    if args.test_only:
        success = test_api()
        sys.exit(0 if success else 1)

    if args.node_rank is None:
        parser.print_help()
        print("\n" + "=" * 60)
        print("QUICK START:")
        print(
            "  Terminal 1: python test_cross_node_transfer_engine_info.py --node-rank 0"
        )
        print(
            "  Terminal 2: python test_cross_node_transfer_engine_info.py --node-rank 1"
        )
        print(
            "  Terminal 3: python test_cross_node_transfer_engine_info.py --test-only"
        )
        print("=" * 60)
        sys.exit(0)

    dist_init_addr = args.dist_init_addr or f"{get_local_ip()}:20000"
    print(f"Using dist-init-addr: {dist_init_addr}")

    launch_node(args.node_rank, args.model_path, dist_init_addr)


if __name__ == "__main__":
    main()

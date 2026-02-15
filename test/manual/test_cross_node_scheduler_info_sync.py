#!/usr/bin/env python3
"""
Test cross-node scheduler_infos synchronization for remote weight loading.

Simulates multi-node setups on a single machine using different GPU subsets.
Validates that scheduler_infos are correctly synced across nodes via Gloo.

IMPORTANT: For multi-node tests, start both nodes within a few seconds of each
other to avoid port binding conflicts (they share the same network namespace).

Test cases:
  - tp4_nodes2: TP=4 across 2 nodes, validates basic cross-node sync
  - dp2_single_node: DP=2 with dp_attention on single node
  - dp2_tp2_nodes2: DP=2, TP=4 across 2 nodes with dp_attention

Usage (multi-node):
    Terminal 1: python test_cross_node_scheduler_info_sync.py --test-case tp4_nodes2 --node-rank 0
    Terminal 2: python test_cross_node_scheduler_info_sync.py --test-case tp4_nodes2 --node-rank 1
    Terminal 3: python test_cross_node_scheduler_info_sync.py --test-case tp4_nodes2 --test-only

Usage (single-node):
    Terminal 1: python test_cross_node_scheduler_info_sync.py --test-case dp2_single_node --node-rank 0
    Terminal 2: python test_cross_node_scheduler_info_sync.py --test-case dp2_single_node --test-only

Requirements: 4 GPUs on single machine
"""

import argparse
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List

import requests

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
)


@dataclass
class TestCase:
    name: str
    tp_size: int
    dp_size: int
    nnodes: int
    gpus_per_node: int
    expected_ranks: int
    extra_args: List[str]


TEST_CASES = {
    "tp4_nodes2": TestCase(
        name="tp4_nodes2",
        tp_size=4,
        dp_size=1,
        nnodes=2,
        gpus_per_node=2,
        expected_ranks=4,
        extra_args=[],
    ),
    "dp2_single_node": TestCase(
        name="dp2_single_node",
        tp_size=2,
        dp_size=2,
        nnodes=1,
        gpus_per_node=2,
        expected_ranks=2,
        extra_args=["--enable-dp-attention", "--dp", "2", "--attention-backend", "fa3"],
    ),
    "dp2_tp2_nodes2": TestCase(
        name="dp2_tp2_nodes2",
        tp_size=4,
        dp_size=2,
        nnodes=2,
        gpus_per_node=2,
        expected_ranks=4,
        extra_args=["--enable-dp-attention", "--dp", "2", "--attention-backend", "fa3"],
    ),
}

TEST_CASE_MODELS = {
    "tp4_nodes2": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    "dp2_single_node": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    "dp2_tp2_nodes2": DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
}


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def launch_node(
    test_case: TestCase, node_rank: int, model_path: str, dist_init_addr: str
):
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        str(test_case.tp_size),
        "--port",
        str(30000 + node_rank * 100),
        "--host",
        "0.0.0.0",
        "--remote-instance-weight-loader-start-seed-via-transfer-engine",
    ]
    if test_case.nnodes > 1:
        cmd.extend(
            [
                "--nnodes",
                str(test_case.nnodes),
                "--node-rank",
                str(node_rank),
                "--dist-init-addr",
                dist_init_addr,
                "--base-gpu-id",
                str(node_rank * test_case.gpus_per_node),
            ]
        )
    cmd.extend(test_case.extra_args)
    print(f"[Node {node_rank}] {' '.join(cmd)}")
    subprocess.run(cmd)


def test_api(test_case: TestCase) -> bool:
    base_url = "http://127.0.0.1:30000"
    print(f"Testing {test_case.name}: expecting {test_case.expected_ranks} ranks")

    for _ in range(60):
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("ERROR: Server not ready")
        return False

    all_passed = True
    for rank in range(test_case.expected_ranks):
        try:
            resp = requests.get(
                f"{base_url}/get_remote_instance_transfer_engine_info",
                params={"rank": rank},
                timeout=5,
            )
            status = "✓" if resp.status_code == 200 else "✗"
            print(f"{status} Rank {rank}: {resp.status_code}")
            if resp.status_code != 200:
                all_passed = False
        except Exception as e:
            print(f"✗ Rank {rank}: {e}")
            all_passed = False

    print("PASSED" if all_passed else "FAILED")
    return all_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-case", type=str, choices=list(TEST_CASES.keys()), required=True
    )
    parser.add_argument("--node-rank", type=int, choices=[0, 1])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dist-init-addr", type=str, default=None)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    test_case = TEST_CASES[args.test_case]
    model_path = args.model_path or TEST_CASE_MODELS.get(
        args.test_case, DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    )

    if args.test_only:
        sys.exit(0 if test_api(test_case) else 1)

    if test_case.nnodes == 1:
        launch_node(test_case, 0, model_path, "")
        return

    if args.node_rank is None:
        print(f"Usage: --node-rank 0 or 1, then --test-only in another terminal")
        sys.exit(0)

    dist_init_addr = args.dist_init_addr or f"{get_local_ip()}:20000"
    launch_node(test_case, args.node_rank, model_path, dist_init_addr)


if __name__ == "__main__":
    main()

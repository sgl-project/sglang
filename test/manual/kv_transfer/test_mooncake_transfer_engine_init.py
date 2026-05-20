#!/usr/bin/env python3
"""
Test script for validating Mooncake transfer-engine gating and initialization.
Tests the Mooncake-related branches in the current model-runner flow.

This test verifies:
1. MooncakeTransferEngine initialization conditions
2. Different server argument combinations that trigger mooncake TE
3. Mooncake transfer engine initialization with hostname, gpu_id, and ib_device

Usage:
    # Run from project root on 2 GPUs
    CUDA_VISIBLE_DEVICES=0,1 python test/manual/kv_transfer/test_mooncake_transfer_engine_init.py
"""

import argparse
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch


@dataclass
class ServerArgs:
    """Mock ServerArgs for testing."""

    disaggregation_mode: str = "null"
    disaggregation_transfer_backend: str = "mooncake"
    enable_hierarchical_cache: bool = False
    hicache_storage_backend: str = "mooncake"
    encoder_only: bool = False
    language_only: bool = False
    encoder_transfer_backend: str = "mooncake"
    enable_elastic_expert_backup: bool = False
    elastic_ep_backend: Optional[str] = None
    disaggregation_ib_device: Optional[str] = None
    mooncake_ib_device: Optional[str] = None


def test_mooncake_te_condition(server_args: ServerArgs) -> bool:
    """
    Test the condition logic for using MooncakeTransferEngine.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    dummy_runner = SimpleNamespace(server_args=server_args, gpu_id=0)
    init_called = False

    def _fake_init_mooncake_transfer_engine(*, hostname, gpu_id, ib_device):
        nonlocal init_called
        init_called = True
        return SimpleNamespace(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device,
        )

    with (
        patch(
            "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.init_mooncake_transfer_engine",
            side_effect=_fake_init_mooncake_transfer_engine,
        ),
        patch(
            "sglang.srt.model_executor.model_runner.get_local_ip_auto",
            return_value="127.0.0.1",
        ),
    ):
        ModelRunner.init_shared_mooncake_transfer_engine(dummy_runner)

    return init_called


def run_mooncake_init(
    rank: int,
    world_size: int,
    master_port: int,
    args: argparse.Namespace,
    server_args: ServerArgs,
):
    """Worker function for testing mooncake transfer engine initialization."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Import before try block to avoid NameError in finally
    import torch
    import torch.distributed as dist

    dist_initialized = False

    try:
        # Initialize distributed environment
        print(f"[Rank {rank}] Initializing distributed environment...")
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            init_method=f"tcp://127.0.0.1:{master_port}",
            device_id=rank,
        )
        dist_initialized = True

        # Set device
        torch.cuda.set_device(rank)

        # Sync to ensure all ranks are ready
        dist.barrier()
        print(f"[Rank {rank}] Distributed initialization complete.")

        # Test the condition logic
        use_mooncake_te = test_mooncake_te_condition(server_args)
        print(f"[Rank {rank}] use_mooncake_te = {use_mooncake_te}")

        if use_mooncake_te:
            print(f"[Rank {rank}] Attempting to initialize MooncakeTransferEngine...")

            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )
            from sglang.srt.utils import get_local_ip_auto

            ib_device = (
                server_args.disaggregation_ib_device or server_args.mooncake_ib_device
            )

            print(f"[Rank {rank}] IB device: {ib_device}")

            # Always actually initialize mooncake
            engine = init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=rank,
                ib_device=ib_device,
            )
            print(f"[Rank {rank}] Session ID: {engine.get_session_id()}")
            print(f"[Rank {rank}] MooncakeTransferEngine initialized successfully!")

            dist.barrier()

        print(f"[Rank {rank}] Test completed successfully!")
        sys.exit(0)

    except ImportError as e:
        print(f"[Rank {rank}] Mooncake not available (ImportError): {e}")
        sys.exit(1)

    except Exception as e:
        print(f"[Rank {rank}] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        if dist_initialized and dist.is_initialized():
            dist.destroy_process_group()
        print(f"[Rank {rank}] Process group destroyed.")


def run_test(args: argparse.Namespace, server_args: ServerArgs) -> bool:
    """Run the mooncake transfer engine test."""
    # Set CUDA visible devices
    cuda_devices = args.cuda_visible_devices.split(",")
    world_size = len(cuda_devices)

    if world_size < 2:
        print("ERROR: This test requires at least 2 GPUs.")
        print(
            "Usage: CUDA_VISIBLE_DEVICES=0,1 python test/manual/kv_transfer/test_mooncake_transfer_engine_init.py"
        )
        sys.exit(1)

    # Check GPU availability
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        print(f"ERROR: Requested {world_size} GPUs but only {available_gpus} available")
        sys.exit(1)

    print(f"Testing with {world_size} GPUs: {cuda_devices}")
    print()

    # Print server args configuration
    print("ServerArgs configuration:")
    for key, value in vars(server_args).items():
        print(f"  {key}: {value}")
    print()

    # Check if mooncake should be used
    use_mooncake_te = test_mooncake_te_condition(server_args)
    print(f"use_mooncake_te = {use_mooncake_te}")
    print()

    # Find a free port
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]

    print(f"Using master port: {master_port}")

    # Spawn worker processes
    ctx = multiprocessing.get_context("spawn")
    processes = []

    for rank in range(world_size):
        p = ctx.Process(
            target=run_mooncake_init,
            args=(rank, world_size, master_port, args, server_args),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    success = True
    for i, p in enumerate(processes):
        p.join(timeout=60)
        if p.exitcode != 0:
            print(f"Process {i} failed with exit code: {p.exitcode}")
            success = False

    # Cleanup any remaining processes
    for p in processes:
        if p.is_alive():
            print(f"Process {p.pid} is still alive, terminating...")
            p.terminate()
            p.join(timeout=5)

    return success


def test_condition_logic():
    """Test the condition logic for different server argument combinations."""
    print("=" * 60)
    print("Testing condition logic for use_mooncake_te")
    print("=" * 60)
    print()

    original_hicache_reuse = os.environ.get("SGLANG_HICACHE_MOONCAKE_REUSE_TE")
    passed = 0
    failed = 0

    try:
        test_cases = [
            # (name, env_value, server_args, expected_result)
            (
                "PD disaggregation with mooncake",
                None,
                ServerArgs(
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="mooncake",
                ),
                True,
            ),
            (
                "PD disaggregation without mooncake",
                None,
                ServerArgs(
                    disaggregation_mode="prefill",
                    disaggregation_transfer_backend="other",
                ),
                False,
            ),
            (
                "No disaggregation",
                None,
                ServerArgs(),
                False,
            ),
            (
                "HiCache with mooncake (env=False)",
                "0",
                ServerArgs(
                    enable_hierarchical_cache=True,
                    hicache_storage_backend="mooncake",
                ),
                False,
            ),
            (
                "HiCache with mooncake (env=True)",
                "1",
                ServerArgs(
                    enable_hierarchical_cache=True,
                    hicache_storage_backend="mooncake",
                ),
                True,
            ),
            (
                "Encoder only with mooncake",
                None,
                ServerArgs(encoder_only=True, encoder_transfer_backend="mooncake"),
                True,
            ),
            (
                "Language only with mooncake",
                None,
                ServerArgs(language_only=True, encoder_transfer_backend="mooncake"),
                True,
            ),
            (
                "Elastic expert backup with backend",
                None,
                ServerArgs(
                    enable_elastic_expert_backup=True,
                    elastic_ep_backend="mooncake",
                ),
                True,
            ),
            (
                "Elastic expert backup without backend",
                None,
                ServerArgs(enable_elastic_expert_backup=True, elastic_ep_backend=None),
                False,
            ),
        ]

        for name, env_value, server_args, expected in test_cases:
            if env_value is None:
                os.environ.pop("SGLANG_HICACHE_MOONCAKE_REUSE_TE", None)
            else:
                os.environ["SGLANG_HICACHE_MOONCAKE_REUSE_TE"] = env_value

            result = test_mooncake_te_condition(server_args)
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"{status}: {name}")
            print(f"       Expected: {expected}, Got: {result}")
            print()
    finally:
        if original_hicache_reuse is None:
            os.environ.pop("SGLANG_HICACHE_MOONCAKE_REUSE_TE", None)
        else:
            os.environ["SGLANG_HICACHE_MOONCAKE_REUSE_TE"] = original_hicache_reuse

    print(f"Condition logic tests: {passed} passed, {failed} failed")
    print()

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate Mooncake transfer-engine gating and initialization"
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0,1",
        help="CUDA visible devices (default: 0,1)",
    )
    parser.add_argument(
        "--test-case",
        type=str,
        choices=[
            "pd_disaggregation",
            "hicache",
            "encoder_only",
            "language_only",
            "elastic_ep",
        ],
        default="pd_disaggregation",
        help="Test case to run",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Mooncake Transfer Engine Init Test")
    print("=" * 60)
    print()

    # First run condition logic tests
    condition_passed = test_condition_logic()

    if not condition_passed:
        print("Condition logic tests failed, skipping distributed test.")
        sys.exit(1)

    # Configure server args based on test case
    server_args = ServerArgs()

    if args.test_case == "pd_disaggregation":
        server_args.disaggregation_mode = "prefill"
        server_args.disaggregation_transfer_backend = "mooncake"
    elif args.test_case == "hicache":
        server_args.enable_hierarchical_cache = True
        server_args.hicache_storage_backend = "mooncake"
        os.environ["SGLANG_HICACHE_MOONCAKE_REUSE_TE"] = "1"
    elif args.test_case == "encoder_only":
        server_args.encoder_only = True
        server_args.encoder_transfer_backend = "mooncake"
    elif args.test_case == "language_only":
        server_args.language_only = True
        server_args.encoder_transfer_backend = "mooncake"
    elif args.test_case == "elastic_ep":
        server_args.enable_elastic_expert_backup = True
        server_args.elastic_ep_backend = "mooncake"

    start_time = time.time()
    success = run_test(args, server_args)
    elapsed_time = time.time() - start_time

    print()
    print("=" * 60)
    if success:
        print(f"TEST PASSED (elapsed: {elapsed_time:.2f}s)")
    else:
        print(f"TEST FAILED (elapsed: {elapsed_time:.2f}s)")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

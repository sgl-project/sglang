#!/usr/bin/env python3
"""
Simple RCCL test for multi-GPU communication.
This test verifies that RCCL can initialize and communicate across multiple GPUs.
"""
import os
import sys

import torch
import torch.distributed as dist


def test_rccl_allreduce():
    """Test basic RCCL allreduce operation across all GPUs."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        sys.exit(1)

    # Initialize process group with NCCL (RCCL on AMD)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}/{world_size}] Initialized successfully")

    # Set device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Device: {torch.cuda.get_device_name(device)}")
    print(
        f"[Rank {rank}] Device memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
    )

    # Create a tensor and perform allreduce
    tensor = torch.ones(1000, device=device) * rank
    print(f"[Rank {rank}] Before allreduce: tensor sum = {tensor.sum().item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected_sum = sum(range(world_size)) * 1000
    actual_sum = tensor.sum().item()

    print(
        f"[Rank {rank}] After allreduce: tensor sum = {actual_sum}, expected = {expected_sum}"
    )

    if abs(actual_sum - expected_sum) < 0.1:
        print(f"[Rank {rank}] ✓ RCCL allreduce test PASSED")
        dist.destroy_process_group()
        sys.exit(0)
    else:
        print(f"[Rank {rank}] ✗ RCCL allreduce test FAILED")
        dist.destroy_process_group()
        sys.exit(1)


if __name__ == "__main__":
    test_rccl_allreduce()

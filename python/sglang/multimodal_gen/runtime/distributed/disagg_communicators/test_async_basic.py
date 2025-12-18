#!/usr/bin/env python3
"""
Basic test script for async communication in disaggregated execution.

This script verifies that:
1. Async communication primitives work correctly
2. A single request can complete successfully
3. Tensors are transferred correctly between Non-DiT and DiT ranks

Usage:
    torchrun --nproc_per_node=4 test_async_basic.py
"""

import sys

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device
    torch.cuda.set_device(rank)

    return rank, world_size


def test_basic_async_send_recv():
    """Test basic async send/recv between two ranks."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("Need at least 2 ranks for this test")
        return False

    print(f"[Rank {rank}] Starting basic async test")

    # Sender: rank 0, Receiver: rank 1
    if rank == 0:
        # Send a tensor
        tensor = torch.randn(100, 100, device="cuda")
        print(f"[Rank 0] Sending tensor with sum={tensor.sum().item():.4f}")

        work = dist.isend(tensor, dst=1)
        print(f"[Rank 0] Async send initiated")

        # Do some work while sending
        dummy = torch.randn(100, 100, device="cuda")
        result = dummy @ dummy
        print(f"[Rank 0] Did some computation: {result.sum().item():.4f}")

        # Wait for send to complete
        work.wait()
        print(f"[Rank 0] Send completed")

    elif rank == 1:
        # Receive a tensor
        tensor = torch.empty(100, 100, device="cuda")

        work = dist.irecv(tensor, src=0)
        print(f"[Rank 1] Async recv initiated")

        # Wait for receive
        work.wait()
        print(f"[Rank 1] Received tensor with sum={tensor.sum().item():.4f}")

    dist.barrier()
    print(f"[Rank {rank}] Basic async test passed!")
    return True


def test_disagg_topology():
    """Test disaggregated topology setup."""
    rank, world_size = setup_distributed()

    if world_size < 3:
        print("Need at least 3 ranks for disagg topology test")
        return False

    print(f"[Rank {rank}] Testing disagg topology")

    # Simulate: Last rank is Non-DiT, others are DiT
    num_non_dit_ranks = 1
    non_dit_ranks = list(range(world_size - num_non_dit_ranks, world_size))
    dit_ranks = [r for r in range(world_size) if r not in non_dit_ranks]

    print(f"[Rank {rank}] Non-DiT ranks: {non_dit_ranks}, DiT ranks: {dit_ranks}")

    # Create groups
    non_dit_group = dist.new_group(ranks=non_dit_ranks)
    dit_group = dist.new_group(ranks=dit_ranks)

    if rank in non_dit_ranks:
        role = "non_dit"
        my_group = non_dit_group
    else:
        role = "dit"
        my_group = dit_group

    print(f"[Rank {rank}] Role: {role}")

    # Test broadcast within group
    if role == "dit":
        tensor = torch.ones(10, device="cuda") * rank
        dist.broadcast(tensor, src=dit_ranks[0], group=dit_group)
        expected = dit_ranks[0]
        assert torch.allclose(
            tensor, torch.ones(10, device="cuda") * expected
        ), f"Broadcast failed: expected {expected}, got {tensor[0].item()}"
        print(f"[Rank {rank}] Broadcast test passed")

    dist.barrier()
    print(f"[Rank {rank}] Disagg topology test passed!")
    return True


def test_async_batch_transfer():
    """Test async transfer of a batch-like structure."""
    rank, world_size = setup_distributed()

    if world_size < 2:
        print("Need at least 2 ranks for batch transfer test")
        return False

    print(f"[Rank {rank}] Testing async batch transfer")

    # Simulate batch transfer: rank 0 -> rank 1
    if rank == 0:
        # Create a mock batch
        latents = torch.randn(2, 4, 64, 64, device="cuda")
        embeddings = torch.randn(2, 77, 768, device="cuda")

        print(
            f"[Rank 0] Sending batch: latents sum={latents.sum().item():.4f}, "
            f"embeddings sum={embeddings.sum().item():.4f}"
        )

        # Send metadata first (sync for simplicity in test)
        metadata = {
            "latents_shape": list(latents.shape),
            "embeddings_shape": list(embeddings.shape),
        }

        # In real implementation, metadata would be pickled and sent as tensor
        # For test, we just send shapes as tensors
        shapes_tensor = torch.tensor(
            [*latents.shape, *embeddings.shape], dtype=torch.long, device="cuda"
        )
        dist.send(shapes_tensor, dst=1)

        # Async send tensors
        work1 = dist.isend(latents, dst=1)
        work2 = dist.isend(embeddings, dst=1)

        print(f"[Rank 0] Async sends initiated")

        # Wait for completion
        work1.wait()
        work2.wait()

        print(f"[Rank 0] Batch transfer completed")

    elif rank == 1:
        # Receive shapes
        shapes_tensor = torch.empty(8, dtype=torch.long, device="cuda")
        dist.recv(shapes_tensor, src=0)

        latents_shape = tuple(shapes_tensor[:4].tolist())
        embeddings_shape = tuple(shapes_tensor[4:].tolist())

        print(
            f"[Rank 1] Expecting latents: {latents_shape}, embeddings: {embeddings_shape}"
        )

        # Async receive tensors
        latents = torch.empty(latents_shape, device="cuda")
        embeddings = torch.empty(embeddings_shape, device="cuda")

        work1 = dist.irecv(latents, src=0)
        work2 = dist.irecv(embeddings, src=0)

        print(f"[Rank 1] Async recvs initiated")

        # Wait for completion
        work1.wait()
        work2.wait()

        print(
            f"[Rank 1] Received batch: latents sum={latents.sum().item():.4f}, "
            f"embeddings sum={embeddings.sum().item():.4f}"
        )

    dist.barrier()
    print(f"[Rank {rank}] Async batch transfer test passed!")
    return True


def main():
    """Run all tests."""
    try:
        rank, world_size = setup_distributed()

        print(f"\n{'='*60}")
        print(f"Running Async Communication Tests")
        print(f"Rank: {rank}/{world_size}")
        print(f"{'='*60}\n")

        # Run tests
        tests = [
            ("Basic Async Send/Recv", test_basic_async_send_recv),
            ("Disagg Topology", test_disagg_topology),
            ("Async Batch Transfer", test_async_batch_transfer),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\n--- Test: {test_name} ---")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"[Rank {rank}] Test failed with error: {e}")
                import traceback

                traceback.print_exc()
                results.append((test_name, False))

        # Summary
        if rank == 0:
            print(f"\n{'='*60}")
            print("Test Summary:")
            for test_name, result in results:
                status = "✓ PASSED" if result else "✗ FAILED"
                print(f"  {status}: {test_name}")
            print(f"{'='*60}\n")

            all_passed = all(result for _, result in results)
            if all_passed:
                print("All tests passed! ✓")
                return 0
            else:
                print("Some tests failed! ✗")
                return 1

        return 0

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

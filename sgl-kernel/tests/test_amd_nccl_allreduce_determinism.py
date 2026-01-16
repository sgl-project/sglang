"""
Test to confirm non-determinism of default NCCL all-reduce with batch size invariance.

This test uses the default torch.distributed.all_reduce (NCCL) which can be
NON-DETERMINISTIC due to tree-based reduction algorithms that don't guarantee
fixed accumulation order for bfloat16/float16.

This test compares:
1. Default all-reduce (same batch size) - should be DETERMINISTIC
2. Default all-reduce (different batch size) - typically NON-DETERMINISTIC for bfloat16

Usage:
    python test_ar.py
"""

import multiprocessing as mp
import socket

import torch
import torch.distributed as dist


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(world_size, rank, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    num_trials = 10

    # Matrix sizes similar to real model layers
    # Format: (batch_size, hidden_dim) - typical tensor shape for all-reduce
    BS = 50  # max batch_size (1..BS)
    hidden_dim = 16384  # hidden dimension / intermediate dimension

    # Different seed per rank - each GPU has DIFFERENT input
    torch.manual_seed(42 + rank)

    # Create fixed inputs for all trials
    # Single request: (hidden_dim,)
    base_input = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)
    base_input_rand = torch.randn(hidden_dim, dtype=torch.bfloat16, device=device)

    dist.barrier()

    # =========================================================================
    # TEST 1: Default all-reduce (same batch size) - should be DETERMINISTIC
    # =========================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 1: Default NCCL all_reduce (same batch size)")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = []
    for trial in range(num_trials):
        # Clone the same input
        inp = base_input.clone()

        # Use default NCCL all-reduce
        dist.all_reduce(inp)
        torch.cuda.synchronize()

        # Store checksum
        checksum = inp.view(-1).sum().item()
        first_vals = inp.view(-1)[:5].clone()
        results_allreduce_only.append((checksum, first_vals))

        if rank == 0:
            print(
                f"  Trial {trial+1:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}"
            )

    # Check determinism
    if rank == 0:
        ref_sum, ref_vals = results_allreduce_only[0]
        all_match = True
        for i, (s, vals) in enumerate(results_allreduce_only[1:], 1):
            if abs(ref_sum - s) > 1e-3 or not torch.allclose(ref_vals, vals, rtol=1e-3):
                all_match = False
                print(f"  Trial {i+1} DIFFERS! ref_sum={ref_sum:.6f}, got={s:.6f}")

        if all_match:
            print("  ✓ DEFAULT ALL_REDUCE (fixed BS): DETERMINISTIC (as expected)")
        else:
            print("  ✗ DEFAULT ALL_REDUCE (fixed BS): NON-DETERMINISTIC (unexpected!)")

    dist.barrier()

    # =========================================================================
    # TEST 2: Default all-reduce (different batch size) - typically NON-DETERMINISTIC
    # [a], [a, x], [a, x, x], ...
    # =========================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 2: Default NCCL all_reduce (different batch size)")
        print("Batches: [a], [a,x], [a,x,x], ...")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = {trial: [] for trial in range(num_trials)}
    for trial in range(num_trials):
        for bs in range(1, BS + 1):
            # Construct batch: (batch_size, hidden_dim)
            # First element is base_input, rest are base_input_rand
            batch = torch.stack([base_input] + [base_input_rand] * (bs - 1), dim=0)
            # Shape: (bs, hidden_dim)

            # Flatten for all-reduce: (bs * hidden_dim,)
            batch_flat = batch.view(-1)

            # Use default NCCL all-reduce
            dist.all_reduce(batch_flat)
            torch.cuda.synchronize()

            # Reshape back to (bs, hidden_dim)
            batch_out = batch_flat.view(bs, hidden_dim)

            # Only compare output corresponding to first request
            out_first_req = batch_out[0].clone()
            checksum = out_first_req.sum().item()
            first_vals = out_first_req[:5].clone()
            results_allreduce_only[trial].append((bs, checksum, first_vals))

            if rank == 0:
                print(
                    f"  Batch size {bs:2d}: sum={checksum:.6f}, first5={first_vals.tolist()}"
                )

    # Check determinism
    if rank == 0:
        for trial in range(num_trials):
            results = results_allreduce_only[trial]

            _, ref_sum, ref_vals = results[0]
            all_match = True
            for _, s, vals in results[1:]:
                if abs(ref_sum - s) > 1e-3 or not torch.allclose(
                    ref_vals, vals, rtol=1e-3
                ):
                    all_match = False

        if all_match:
            print("  ✓ DEFAULT ALL_REDUCE (variant BS): DETERMINISTIC")
        else:
            print("  ✗ DEFAULT ALL_REDUCE (variant BS): NON-DETERMINISTIC")

    dist.barrier()

    dist.destroy_process_group()


def main():
    world_size = 8
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("Default NCCL All-Reduce Determinism Test")
    print("=" * 70)
    print(f"Available GPUs: {available_gpus}")
    print(f"Using world_size: {world_size}")

    if available_gpus < world_size:
        print(
            f"WARNING: Only {available_gpus} GPUs available, using {available_gpus} instead"
        )
        world_size = available_gpus

    if world_size < 2:
        print("ERROR: Need at least 2 GPUs for this test")
        return

    mp.set_start_method("spawn", force=True)
    port = get_open_port()

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(world_size, rank, port))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    main()

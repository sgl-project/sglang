"""
Test deterministic custom all-reduce kernel behavior with batch size invariance.

This test uses the 1-stage all-reduce kernel which is inherently deterministic
due to fixed accumulation ordering (each GPU reads all data from all GPUs and
reduces locally in a fixed order - no atomics, no race conditions).

Note: This is NOT a reduce-scatter + all-gather (RS+AG) approach.

This test compares:
1. Deterministic kernel (same batch size)
2. Deterministic kernel (different batch size)

It also checks inputs larger than the staging buffer the kernel reduces
through, which are all-reduced in pieces.

Usage:
    pytest test_amd_deterministic_custom_allreduce.py
"""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="sgl-kernel-unit", runner_config="2-gpu-amd")


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(world_size, rank, port):
    envs.SGLANG_USE_1STAGE_ALLREDUCE.set("1")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    # Try to import and use deterministic kernel
    try:
        from torch.distributed import new_group

        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        # Create gloo group for custom AR
        dist.barrier()
        ar_group = new_group(backend="gloo")
        dist.barrier()

        custom_ar = CustomAllreduce(group=ar_group, device=device)

        if custom_ar is None or custom_ar.disabled:
            if rank == 0:
                print("✗ Custom AR not available or disabled")
            dist.destroy_process_group()
            return
    except Exception as e:
        if rank == 0:
            print(f"✗ Failed to initialize deterministic kernel: {e}")
            import traceback

            traceback.print_exc()
        dist.destroy_process_group()
        return

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

    # Check if inputs fit in buffer
    # Buffer size is max_size bytes, input size is numel * element_size bytes
    input_size_bytes = base_input.numel() * base_input.element_size()
    if input_size_bytes > custom_ar.max_size and rank == 0:
        print(
            f"Warning: Input size ({input_size_bytes/(1024*1024):.1f} MB) exceeds buffer size ({custom_ar.max_size/(1024*1024):.1f} MB)"
        )
        print("  Using unregistered mode (will copy to buffer)")

    dist.barrier()

    # =========================================================================
    # TEST 1: Deterministic kernel (same batch size) - should be DETERMINISTIC
    # =========================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 1: Deterministic kernel (same batch size)")
        print(f"{'='*70}")
    dist.barrier()

    results_allreduce_only = []
    for trial in range(num_trials):
        # Clone the same input
        inp = base_input.clone()

        result = custom_ar.custom_all_reduce(inp)
        torch.cuda.synchronize()

        # Store checksum
        checksum = result.view(-1).sum().item()
        first_vals = result.view(-1)[:5].clone()
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
            print("  ✓ DETERMINISTIC KERNEL (fixed BS): DETERMINISTIC (as expected)")
        else:
            print(
                "  ✗ DETERMINISTIC KERNEL (fixed BS): NON-DETERMINISTIC (unexpected!)"
            )

    dist.barrier()

    # =========================================================================
    # TEST 2: Deterministic kernel (different batch size) - should be DETERMINISTIC
    # [a], [a, x], [a, x, x], ...
    # =========================================================================
    if rank == 0:
        print(f"\n{'='*70}")
        print("TEST 2: Deterministic kernel (different batch size)")
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

            result_flat = custom_ar.custom_all_reduce(batch_flat)
            torch.cuda.synchronize()

            # Reshape back to (bs, hidden_dim)
            batch_out = result_flat.view(bs, hidden_dim)

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
            print("  ✓ DETERMINISTIC KERNEL (variant BS): DETERMINISTIC")
        else:
            print("  ✗ DETERMINISTIC KERNEL (variant BS): NON-DETERMINISTIC")

    dist.barrier()

    dist.destroy_process_group()


# The staging buffer the deterministic kernel reduces through holds `max_size`
# bytes, and a small one makes the behaviour of larger inputs cheap to check.
OVERSIZED_MAX_SIZE = 256 * 1024


def oversized_worker(world_size, rank, port):
    """All-reduce inputs that do not fit in the staging buffer, repeatedly.

    Such an input used to be registered on every call, and registrations are
    never released, so a long-running server eventually died with "Rank data
    buffer is overflowed".
    """
    envs.SGLANG_USE_1STAGE_ALLREDUCE.set("1")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    from torch.distributed import new_group

    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    dist.barrier()
    ar_group = new_group(backend="gloo")
    dist.barrier()

    custom_ar = CustomAllreduce(
        group=ar_group, device=device, max_size=OVERSIZED_MAX_SIZE
    )
    if custom_ar.disabled or not custom_ar.use_amd_deterministic_impl:
        if rank == 0:
            print("✗ Deterministic custom AR not available")
        dist.destroy_process_group()
        return

    def check(numel, num_calls):
        # Small integers, exact in fp32, repeating with a period that does not
        # divide the piece size, so a misplaced piece changes the result.
        pattern = (torch.arange(numel, device=device, dtype=torch.float32) % 1021) + 1
        inp = pattern * (rank + 1)
        expected = pattern * (world_size * (world_size + 1) // 2)
        for call in range(num_calls):
            result = custom_ar.custom_all_reduce(inp)
            assert result is not None, "custom all-reduce declined the input"
            if call in (0, num_calls - 1):
                torch.cuda.synchronize()
                wrong = result != expected
                if wrong.any():
                    idx = int(wrong.to(torch.uint8).argmax())
                    raise AssertionError(
                        f"rank {rank}: call {call + 1} on {numel} elements is "
                        f"wrong in {int(wrong.sum())} places, first at index "
                        f"{idx}: got {result[idx].item()}, "
                        f"want {expected[idx].item()}"
                    )

    # Control: an input that fits, so a failure below is about the size of the
    # input and not about the kernel or the harness.
    buffer_elements = OVERSIZED_MAX_SIZE // 4
    check(buffer_elements // 2, num_calls=1)
    if rank == 0:
        print("  ✓ input that fits in the staging buffer: correct")

    # One registration per call used to take a `RankData` slot, and `rank_data`
    # holds `max_size / sizeof(RankData)` of them; 16 bytes deliberately
    # under-estimates the 64-byte struct, so this overruns the budget either
    # way. 2.5 buffers of input also leaves a partial piece at the end.
    num_calls = OVERSIZED_MAX_SIZE // 16 + 1
    check(buffer_elements * 5 // 2, num_calls=num_calls)
    if rank == 0:
        print(f"  ✓ {num_calls} all-reduces of an oversized input: correct")

    dist.barrier()
    dist.destroy_process_group()


def run_workers(target, title):
    world_size = 8
    available_gpus = torch.cuda.device_count()

    print("=" * 70)
    print(title)
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
        p = mp.Process(target=target, args=(world_size, rank, port))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    failed = {rank: p.exitcode for rank, p in enumerate(procs) if p.exitcode != 0}
    if failed:
        raise AssertionError(f"worker processes failed, exit codes by rank: {failed}")


def main():
    run_workers(worker, "Deterministic Kernel All-Reduce Determinism Test")
    run_workers(oversized_worker, "Deterministic Kernel Oversized Input Test")


requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs",
)


@requires_two_gpus
def test_deterministic_custom_allreduce():
    """Test that deterministic custom all-reduce produces consistent results."""
    run_workers(worker, "Deterministic Kernel All-Reduce Determinism Test")


@requires_two_gpus
def test_deterministic_custom_allreduce_oversized_input():
    """Test inputs larger than the staging buffer, repeated."""
    run_workers(oversized_worker, "Deterministic Kernel Oversized Input Test")


if __name__ == "__main__":
    main()
